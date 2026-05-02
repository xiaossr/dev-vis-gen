"""Reproduce PyTorch's torchao recipe on QNN: quantize only linears.

PyTorch's `Int8DynamicActivationInt8WeightConfig` only quantizes `nn.Linear`
(109 modules in FLUX). Every other op (mul, add, LN, softmax, cat, view,
permute) stays at fp. Result: 9.9 dB SNR static / 17.4 dB SNR dynamic.

QNN PT2E with HistogramObserver gives -2.6 dB because it annotates EVERY op.
Test: tell QnnQuantizer to discard quant on every non-linear op.
"""
import argparse, copy, json, logging, sys
from pathlib import Path

import torch
from torchao.quantization.pt2e.observer import HistogramObserver

sys.path.insert(0, str(Path(__file__).parent))
from export_flux2_klein_qnn import (
    Flux2TransformerWrapper, build_transformer_inputs,
    configure_local_tooling, load_pipeline,
)
configure_local_tooling()

from executorch.backends.qualcomm.quantizer.quantizer import (
    QnnQuantizer, QuantDtype,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType, QcomChipset,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("linonly")


def diff(name, ref, q):
    ref = ref.detach().float(); q = q.detach().float()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0), q.flatten().unsqueeze(0)).item()
    snr_db = 20 * torch.log10(ref.norm() / ((ref - q).norm() + 1e-12)).item()
    log.info("[%-30s]  max=%.4f  cos=%.5f  SNR=%.2fdB",
             name, (ref - q).abs().max().item(), cos, snr_db)
    return {"name": name, "cos": cos, "snr_db": snr_db}


def run(probe, calib, model, discard_ops, label):
    log.info("\n############ %s ############", label)
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
    if discard_ops:
        # Filter to ops actually in quant_ops to avoid KeyError on missing
        valid = [op for op in discard_ops if op in quantizer.quant_ops]
        skipped = [op for op in discard_ops if op not in quantizer.quant_ops]
        if skipped:
            log.info("Skipping (not in quant_ops): %s", [str(op) for op in skipped])
        quantizer.add_discard_ops(valid)
        log.info("Discarding %d ops from quant: %s",
                 len(valid), [str(op) for op in valid])
    log.info("prepare ...")
    prepared = prepare_pt2e(captured, quantizer)
    log.info("calibrate ...")
    with torch.no_grad():
        for c in calib:
            if not isinstance(c, tuple): c = (c,)
            prepared(*c)
    log.info("convert ...")
    converted = convert_pt2e(prepared)
    log.info("forward ...")
    with torch.no_grad():
        out = converted(*probe)
    if isinstance(out, tuple): out = out[0]
    return out


def main():
    log.info("Loading ...")
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    cal = torch.load(
        str(Path(__file__).parent / "calibration_data" / "calibration_transformer.pt"),
        weights_only=False)[:5]
    probe = cal[0] if isinstance(cal[0], tuple) else (cal[0],)

    model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()

    log.info("fp32 ref ...")
    with torch.no_grad():
        ref = model(*probe)
    if isinstance(ref, tuple): ref = ref[0]

    results = []

    # All non-linear ops we know about (avoid touching linear.default,
    # conv2d.default, bmm.default, matmul.default).
    aten = torch.ops.aten
    NONLIN = [
        aten.mul.Tensor, aten.add.Tensor, aten.sub.Tensor, aten.div.Tensor,
        aten.layer_norm.default, aten.rsqrt.default, aten.mean.dim,
        aten._softmax.default, aten._safe_softmax.default,
        aten.cat.default, aten.silu.default, aten.gelu.default,
        aten.tanh.default, aten.sigmoid.default, aten.neg.default,
        aten.relu.default,
    ]

    out = run(probe, cal, model, [], "BASELINE no discard")
    results.append(diff("baseline", ref, out))

    out = run(probe, cal, model, NONLIN, "discard non-linear ops (1st pass)")
    results.append(diff("linear_only_partial", ref, out))

    # Aggressive: discard EVERYTHING except aten.linear.default and conv*
    log.info("\n=== Building maximal discard list ===")
    # snapshot a fresh quant_ops set
    from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer as QQ
    tmp = QQ(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=QcomChipset.SM8850,
    )
    keep = {aten.linear.default, aten.conv2d.default, aten.conv1d.default}
    aggressive = [op for op in tmp.quant_ops if op not in keep]
    log.info("Aggressive discard list: %d ops (keeping only linear+conv)",
             len(aggressive))

    out = run(probe, cal, model, aggressive, "discard ALL except linear/conv")
    results.append(diff("linear_only_aggressive", ref, out))

    log.info("\n=================== SUMMARY ===================")
    for r in results:
        log.info("  %-25s SNR=%6.2f dB  cos=%.5f", r["name"], r["snr_db"], r["cos"])

    out_path = Path(__file__).parent / "linear_only_results.json"
    out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
