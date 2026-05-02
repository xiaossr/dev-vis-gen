"""Stage 1 host PT2E SNR test: promote context_embedder to a16w8 on top of
the linear-only-discard a8w8 baseline.

Baseline: +3.6 dB / cos 0.82 (diag_linear_only_quant.py with full discard).
Hypothesis: context_embedder is the single biggest local int8 outlier
(7.15 dB local SNR vs next-worst 12.8 dB). Promoting just it should give a
meaningful jump.

If gain >= +0.5 dB: proceed to compile.
If not: run with --top5 to also promote the 5 worst output projections.
"""
import argparse, copy, json, logging, sys
from pathlib import Path

# Add vendored torchao (with pt2e submodule) ahead of installed torchao so that
# the local executorch (HEAD detached at v1.2.0, expects torchao>=0.17 with pt2e)
# imports cleanly. Installed torchao 0.10 lacks pt2e.
_REPO = Path(__file__).resolve().parent
_VENDORED_AO = _REPO / "executorch" / "third-party" / "ao"
if _VENDORED_AO.exists():
    sys.path.insert(0, str(_VENDORED_AO))

import torch

sys.path.insert(0, str(Path(__file__).parent))
from export_flux2_klein_qnn import (
    Flux2TransformerWrapper, configure_local_tooling, load_pipeline,
)
configure_local_tooling()

from torchao.quantization.pt2e.observer import HistogramObserver
from executorch.backends.qualcomm.quantizer.quantizer import (
    QnnQuantizer, QuantDtype, ModuleQConfig,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType, QcomChipset,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("ce_prom")


def diff(name, ref, q):
    ref = ref.detach().float(); q = q.detach().float()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0), q.flatten().unsqueeze(0)).item()
    snr_db = 20 * torch.log10(ref.norm() / ((ref - q).norm() + 1e-12)).item()
    log.info("[%s]  max=%.4f  cos=%.5f  SNR=%.2fdB",
             name, (ref - q).abs().max().item(), cos, snr_db)
    return {"name": name, "cos": cos, "snr_db": snr_db}


# Top-5 worst output projections (by err_norm, after context_embedder)
TOP5_OUTPROJ = [
    "transformer.transformer_blocks.4.ff_context.linear_out",
    "transformer.single_transformer_blocks.19.attn.to_out",
    "transformer.transformer_blocks.3.ff_context.linear_out",
    "transformer.single_transformer_blocks.17.attn.to_out",
    "transformer.single_transformer_blocks.18.attn.to_out",
]

# All "problem" Linears identified by per_linear_snr.json: every output
# projection (attn.to_out + ff*.linear_out) plus context_embedder. These
# clustered at 7-25 dB local SNR; the remaining 70+ Linears were >30 dB.
ALL_PROBLEM_LINEARS = (
    [f"transformer.single_transformer_blocks.{i}.attn.to_out" for i in range(20)]
    + [f"transformer.transformer_blocks.{i}.ff_context.linear_out" for i in range(5)]
    + [f"transformer.transformer_blocks.{i}.ff.linear_out" for i in range(5)]
)


def make_predicate(prefixes, counter):
    """Match nodes whose nn_module_stack contains any of `prefixes` exactly
    (must equal or be the start of the module path, terminated by '.' or end).
    """
    def pred(node):
        stk = node.meta.get("nn_module_stack", {}) if hasattr(node, "meta") else {}
        if not stk:
            return False
        for entry in stk.values():
            mod = entry[0] if isinstance(entry, tuple) else str(entry)
            for prefix in prefixes:
                # exact match or prefix followed by '.'
                if mod == prefix or mod.startswith(prefix + "."):
                    counter[0] += 1
                    return True
        return False
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top5", action="store_true",
                   help="Also promote 5 worst output projections")
    p.add_argument("--all-problem", action="store_true",
                   help="Promote context_embedder + every output projection "
                        "(20 single.attn.to_out + 5 ff_context.linear_out + 5 ff.linear_out)")
    p.add_argument("--ncal", type=int, default=5)
    p.add_argument("--probe-only", action="store_true",
                   help="Just dump nn_module_stack entries that match candidate prefixes and exit")
    args = p.parse_args()

    prefixes = ["transformer.context_embedder"]
    if args.top5:
        prefixes += TOP5_OUTPROJ
    if args.all_problem:
        prefixes += ALL_PROBLEM_LINEARS
    log.info("Promoting %d prefixes", len(prefixes))
    for p_ in prefixes:
        log.info("  %s", p_)

    log.info("Loading pipeline ...")
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    cal = torch.load(
        str(Path(__file__).parent / "calibration_data" / "calibration_transformer.pt"),
        weights_only=False)[:args.ncal]
    probe = cal[0] if isinstance(cal[0], tuple) else (cal[0],)

    model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()

    log.info("fp32 ref ...")
    with torch.no_grad():
        ref = model(*probe)
    if isinstance(ref, tuple): ref = ref[0]

    log.info("export ...")
    captured = torch.export.export(model, probe, strict=True).module()

    if args.probe_only:
        # Dump unique module paths that contain "context_embedder" or any of TOP5
        seen = set()
        for node in captured.graph.nodes:
            stk = node.meta.get("nn_module_stack", {}) if hasattr(node, "meta") else {}
            for entry in stk.values():
                mod = entry[0] if isinstance(entry, tuple) else str(entry)
                if "context_embedder" in mod or "ff_context.linear_out" in mod or "attn.to_out" in mod:
                    seen.add(mod)
        for s in sorted(seen):
            log.info("MOD: %s", s)
        return

    # Build the linear-only-discard a8w8 baseline quantizer
    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=QcomChipset.SM8850,
    )
    quantizer.set_default_quant_config(
        QuantDtype.use_8a8w,
        is_conv_per_channel=True, is_linear_per_channel=True,
        act_observer=HistogramObserver,
    )

    aten = torch.ops.aten
    keep = {aten.linear.default, aten.conv2d.default, aten.conv1d.default}
    aggressive = [op for op in quantizer.quant_ops if op not in keep]
    quantizer.add_discard_ops(aggressive)
    log.info("Discarded %d non-linear/conv ops (linear-only-discard baseline)",
             len(aggressive))

    counter = [0]
    pred = make_predicate(prefixes, counter)

    quantizer.set_submodule_qconfig_list([
        (pred, ModuleQConfig(
            quant_dtype=QuantDtype.use_16a8w,
            is_conv_per_channel=True, is_linear_per_channel=True,
            act_observer=HistogramObserver,
        )),
    ])

    log.info("prepare ...")
    prepared = prepare_pt2e(captured, quantizer)
    log.info("Predicate matched %d FX nodes", counter[0])
    if counter[0] == 0:
        log.error("Predicate matched 0 nodes! Aborting before calibration.")
        return
    log.info("calibrate %d ...", len(cal))
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
    suffix = ""
    if args.top5: suffix += "+top5"
    if args.all_problem: suffix += "+allproblem"
    label = "context_embedder@16a8w" + suffix
    res = diff(label, ref, out)

    fname_suf = ""
    if args.top5: fname_suf += "_top5"
    if args.all_problem: fname_suf += "_all"
    out_path = Path(__file__).parent / f"promote_ce{fname_suf}_results.json"
    out_path.write_text(json.dumps(res, indent=2))
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
