"""Build a VTCM-safe lean int16 profile and host-test PT2E SNR.

Strategy:
  1. Load all_rows from int16_profile.json (already has SNR + shape per observer).
  2. Filter: only candidates with numel < 2M (fits V81 8 MB VTCM at int16).
  3. Take the top-K worst by predicted int8 SNR.
  4. The 16a8w predicate matches FX nodes by (op_name, shape) — no observer
     mapping needed, since the only VTCM-safe shape is (1,512,3072) and we
     enumerate the worst per-op cases.
  5. Run a small calibration + convert + forward to measure mixed PT2E SNR.
"""
import argparse
import ast
import copy
import json
import logging
import math
import sys
from pathlib import Path

import torch
from torchao.quantization.pt2e.observer import HistogramObserver

sys.path.insert(0, str(Path(__file__).parent))
from export_flux2_klein_qnn import (
    Flux2TransformerWrapper,
    build_transformer_inputs,
    configure_local_tooling,
    load_pipeline,
)

configure_local_tooling()

from executorch.backends.qualcomm.quantizer.quantizer import (
    QnnQuantizer, QuantDtype, ModuleQConfig,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType, QcomChipset,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("lean")

INT16_OVERFLOW_NUMEL = 2_000_000  # ~4 MB at int16 (conservative)


def diff(name, ref, q):
    ref = ref.detach().float(); q = q.detach().float()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0), q.flatten().unsqueeze(0)).item()
    snr_db = 20 * torch.log10(ref.norm() / ((ref - q).norm() + 1e-12)).item()
    log.info("%-60s max=%.4f cos=%.5f SNR=%.2fdB",
             name, (ref - q).abs().max().item(), cos, snr_db)


def numel_from_shape(shape_str):
    t = ast.literal_eval(shape_str)
    n = 1
    for d in t: n *= int(d)
    return n


def shape_tuple(s):
    return tuple(ast.literal_eval(s))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top_k", type=int, default=30,
                   help="How many worst VTCM-safe nodes to promote to 16a8w")
    p.add_argument("--out", default="int16_profile_lean.json")
    args = p.parse_args()

    # 1) Pick lean candidates from the existing profile
    profile = json.loads((Path(__file__).parent / "int16_profile.json").read_text())
    rows = profile["all_rows"]

    candidates = []
    skipped_overflow = []
    for r in rows:
        snr = r["predicted_int8_snr_db"]
        outr = r["outlier_ratio"]
        bad = (not math.isnan(snr) and snr < 20.0) or outr > 5.0
        if not bad:
            continue
        n = numel_from_shape(r["shape"])
        if n >= INT16_OVERFLOW_NUMEL:
            skipped_overflow.append(r)
            continue
        candidates.append({**r, "numel": n})

    candidates.sort(key=lambda r: r["predicted_int8_snr_db"])
    lean_top = candidates[:args.top_k]

    log.info("=== %d VTCM-safe; %d skipped (overflow at int16) ===",
             len(candidates), len(skipped_overflow))
    log.info("=== Top %d to promote ===", len(lean_top))
    for r in lean_top:
        log.info("  snr=%6.2f outlier=%5.2f shape=%-20s op=%-25s mod=%s",
                 r["predicted_int8_snr_db"], r["outlier_ratio"],
                 r["shape"], r["op"], r["module"])

    # Build the predicate target set: (target_op_name, shape_tuple, module_prefix)
    predicate_targets = set()
    for r in lean_top:
        predicate_targets.add((r["op"], shape_tuple(r["shape"]), r["module"]))
    log.info("Predicate target tuples: %d", len(predicate_targets))

    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        "vtcm_overflow_threshold_numel": INT16_OVERFLOW_NUMEL,
        "top_k_target": args.top_k,
        "predicate_targets": [
            {"op": op, "shape": list(sh), "module": mod}
            for (op, sh, mod) in predicate_targets
        ],
        "details": lean_top,
        "skipped_overflow_examples": [
            {"snr": r["predicted_int8_snr_db"], "shape": r["shape"],
             "op": r["op"], "module": r["module"]}
            for r in skipped_overflow[:30]
        ],
    }, indent=2))
    log.info("Wrote %s", out_path)

    # 2) Load model and run host PT2E test with the lean config
    log.info("Loading pipeline ...")
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
    sample_inputs = build_transformer_inputs(
        pipe, 512, 512, 512, dtype=torch.float32, num_img2img_images=0)
    cal = torch.load(
        str(Path(__file__).parent / "calibration_data" / "calibration_transformer.pt"),
        weights_only=False)[:5]
    probe = cal[0] if isinstance(cal[0], tuple) else (cal[0],)

    log.info("fp32 ref ...")
    with torch.no_grad():
        ref = model(*probe)
    if isinstance(ref, tuple): ref = ref[0]

    log.info("torch.export ...")
    captured = torch.export.export(model, probe, strict=True).module()

    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=QcomChipset.SM8850,
    )
    quantizer.set_default_quant_config(
        QuantDtype.use_8a8w,
        is_conv_per_channel=True, is_linear_per_channel=True,
        act_observer=HistogramObserver,
    )

    matched_count = [0]  # mutable for closure
    def _is_int16(node):
        # Match by op target + shape + (optional) module path
        target = str(node.target) if hasattr(node, "target") else ""
        v = node.meta.get("val") if hasattr(node, "meta") else None
        if v is None or not hasattr(v, "shape"):
            return False
        sh = tuple(int(d) for d in v.shape)
        # module path
        stk = node.meta.get("nn_module_stack", {}) if hasattr(node, "meta") else {}
        mod = "<?>"
        if stk:
            last = list(stk.values())[-1]
            if isinstance(last, tuple):
                mod = last[0]
        if (target, sh, mod) in predicate_targets:
            matched_count[0] += 1
            return True
        return False

    quantizer.set_submodule_qconfig_list([
        (_is_int16, ModuleQConfig(
            quant_dtype=QuantDtype.use_16a8w,
            is_conv_per_channel=True, is_linear_per_channel=True,
            act_observer=HistogramObserver,
        )),
    ])

    log.info("prepare ...")
    prepared = prepare_pt2e(captured, quantizer)
    log.info("Predicate matched %d FX nodes during prepare", matched_count[0])
    log.info("calibrating with %d ...", len(cal))
    with torch.no_grad():
        for c in cal:
            if not isinstance(c, tuple): c = (c,)
            prepared(*c)
    log.info("convert ...")
    converted = convert_pt2e(prepared)
    log.info("forward ...")
    with torch.no_grad():
        out = converted(*probe)
    if isinstance(out, tuple): out = out[0]
    diff(f"LEAN top-{args.top_k} VTCM-safe (matched={matched_count[0]})", ref, out)


if __name__ == "__main__":
    main()
