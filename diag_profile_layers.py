"""
Profile each candidate quantization-boundary tensor in the transformer to
identify which ones cause the most int8 noise. Output ranks all annotated
activations and saves a JSON of {"int16_node_names": [...]} that the
compile script will consume.

Per-node metrics:
  range_full   = max-abs over ALL samples
  range_per    = mean of per-sample max-abs (a low value here means rare outliers)
  outlier_ratio= range_full / range_per (high → static scale must span outlier;
                 typical values get squashed)
  concentration= (per-channel max-abs) skew: top1/median ratio
  predicted_int8_snr_db ≈ 20 * log10(typical_std * 444 / range_full)

Selection rule: a node goes to int16 if predicted SNR < 20 dB OR outlier_ratio > 5.
"""
import copy
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from export_flux2_klein_qnn import (
    Flux2TransformerWrapper,
    build_transformer_inputs,
    configure_local_tooling,
    load_pipeline,
)

configure_local_tooling()

from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
    QcomChipset,
)
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("profile")


def per_node_stats(prepared, sample_inputs_list):
    """For each activation_post_process_*, gather raw tensor stats per sample."""
    stats = defaultdict(lambda: {
        "samples_max": [],
        "samples_std": [],
        "samples_mean_abs": [],
        "per_channel_max_accum": None,  # max over samples, per-channel (last dim)
        "shape": None,
        "producer_target": None,
        "producer_module_stack": None,
    })

    # Build map observer_name -> producing FX node, and capture metadata
    obs_to_producer = {}
    for node in prepared.graph.nodes:
        if node.op == "call_module" and node.target.startswith("activation_post_process_"):
            producer = node.args[0]
            obs_to_producer[node.target] = producer
            v = producer.meta.get("val")
            shape = tuple(v.shape) if v is not None and hasattr(v, "shape") else None
            stk = producer.meta.get("nn_module_stack", {})
            mstack = "<?>"
            if stk:
                last = list(stk.values())[-1]
                if isinstance(last, tuple):
                    mstack = last[0]
            stats[node.target]["shape"] = shape
            stats[node.target]["producer_target"] = str(producer.target)
            stats[node.target]["producer_module_stack"] = mstack

    # Replace observers' forward to capture tensor stats
    handles = []
    for name, mod in prepared.named_modules():
        if not name.startswith("activation_post_process_"):
            continue
        if not hasattr(mod, "min_val"):
            continue

        def make_hook(obs_name):
            def hook(_m, inputs, out):
                x = inputs[0]
                if not torch.is_tensor(x):
                    return out
                if not x.is_floating_point():
                    return out
                xa = x.detach().abs()
                stats[obs_name]["samples_max"].append(float(xa.max()))
                stats[obs_name]["samples_std"].append(float(x.detach().std()))
                stats[obs_name]["samples_mean_abs"].append(float(xa.mean()))
                # per-channel max along last dim
                if x.ndim >= 2:
                    flat = xa.reshape(-1, x.shape[-1])
                    pc = flat.max(dim=0).values
                    cur = stats[obs_name]["per_channel_max_accum"]
                    if cur is None:
                        stats[obs_name]["per_channel_max_accum"] = pc.clone()
                    elif cur.shape == pc.shape:
                        stats[obs_name]["per_channel_max_accum"] = torch.maximum(cur, pc)
                    # else: observer hit by varying-shape tensors; skip per-channel
                return out
            return hook

        handles.append(mod.register_forward_hook(make_hook(name)))

    # Run all calibration samples
    log.info("Running %d calibration samples through hooks ...", len(sample_inputs_list))
    with torch.no_grad():
        for i, cal in enumerate(sample_inputs_list):
            if not isinstance(cal, tuple):
                cal = (cal,)
            prepared(*cal)
            if (i + 1) % 5 == 0:
                log.info("  %d/%d", i + 1, len(sample_inputs_list))

    for h in handles:
        h.remove()
    return stats


def score_nodes(stats):
    """Compute per-node int8 quality predictor and selection."""
    rows = []
    for name, s in stats.items():
        if not s["samples_max"]:
            continue
        rmax = max(s["samples_max"])
        rmean = sum(s["samples_max"]) / len(s["samples_max"])
        std = max(s["samples_std"]) if s["samples_std"] else 0.0
        outlier_ratio = rmax / max(rmean, 1e-12)

        # per-channel concentration
        pc = s["per_channel_max_accum"]
        concentration = float("nan")
        if pc is not None and pc.numel() > 1:
            top1 = float(pc.max())
            median = float(pc.median())
            concentration = top1 / max(median, 1e-12)

        # predicted int8 SNR (rough)
        if rmax > 0 and std > 0:
            # int8 step = 2*rmax/256; quant noise rms ≈ step/sqrt(12) ≈ rmax/443
            predicted_snr_db = 20.0 * math.log10(std * 443.0 / rmax)
        else:
            predicted_snr_db = float("nan")

        rows.append({
            "name": name,
            "shape": str(s["shape"]),
            "op": s["producer_target"],
            "module": s["producer_module_stack"],
            "range_max": rmax,
            "range_mean": rmean,
            "std_max": std,
            "outlier_ratio": outlier_ratio,
            "concentration": concentration,
            "predicted_int8_snr_db": predicted_snr_db,
        })
    rows.sort(key=lambda r: (r["predicted_int8_snr_db"]
                              if not math.isnan(r["predicted_int8_snr_db"])
                              else 999.0))
    return rows


def select_int16_nodes(rows, snr_threshold=20.0, outlier_threshold=5.0):
    """Pick nodes where int8 is predicted to fail."""
    selected = []
    for r in rows:
        snr = r["predicted_int8_snr_db"]
        ratio = r["outlier_ratio"]
        if not math.isnan(snr) and snr < snr_threshold:
            selected.append(r)
        elif ratio > outlier_threshold:
            selected.append(r)
    return selected


def main():
    log.info("Loading pipeline ...")
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
    sample_inputs = build_transformer_inputs(
        pipe, 512, 512, 512, dtype=torch.float32, num_img2img_images=0,
    )

    log.info("Pre-export ...")
    captured = torch.export.export(model, sample_inputs, strict=True).module()

    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=QcomChipset.SM8850,
    )
    quantizer.set_default_quant_config(
        QuantDtype.use_8a8w,
        is_conv_per_channel=True,
        is_linear_per_channel=True,
    )
    log.info("prepare_pt2e ...")
    prepared = prepare_pt2e(captured, quantizer)

    cal = torch.load(
        str(Path(__file__).parent / "calibration_data" / "calibration_transformer.pt"),
        weights_only=False,
    )
    log.info("Loaded %d calibration samples", len(cal))

    log.info("Profiling ...")
    stats = per_node_stats(prepared, cal)
    log.info("Got stats for %d observers", len(stats))

    rows = score_nodes(stats)
    log.info("=== TOP 50 (worst predicted int8 SNR) ===")
    for r in rows[:50]:
        log.info(
            "  snr=%-6.1f outlier=%-6.1f conc=%-6.1f rmax=%-9.1f rmean=%-7.2f shape=%-25s op=%-30s mod=%s",
            r["predicted_int8_snr_db"], r["outlier_ratio"], r["concentration"],
            r["range_max"], r["range_mean"], r["shape"], r["op"], r["module"],
        )

    selected = select_int16_nodes(rows)
    log.info("=== SELECTED %d nodes for 16a8w (snr<20dB OR outlier_ratio>5) ===",
             len(selected))
    selected_names = [r["name"] for r in selected]

    # Need to map observer name -> producer FX node name (so we can match
    # by `node.name` in the export script's predicate)
    log.info("=== mapping observer_name → producer_node_name ===")
    name_map = {}
    for node in prepared.graph.nodes:
        if (node.op == "call_module"
                and node.target.startswith("activation_post_process_")
                and node.target in selected_names):
            producer = node.args[0]
            name_map[node.target] = producer.name

    out = {
        "all_rows": rows[:200],  # don't blow up the file
        "int16_observer_names": selected_names,
        "int16_producer_node_names": list(name_map.values()),
    }
    out_path = Path(__file__).parent / "int16_profile.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info("Wrote %s with %d int16 nodes (%d unique producers)",
             out_path, len(selected_names), len(set(name_map.values())))


if __name__ == "__main__":
    main()
