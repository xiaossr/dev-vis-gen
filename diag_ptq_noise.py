"""
Host-side PTQ-noise diagnostic. Quantifies how much error pure post-training
quantization introduces, *before* QNN ever touches the model. Compares three
configs against fp32 reference:
  A) per-channel weights (our current export)
  B) per-tensor weights  (April's setup)
  C) per-channel weights + softmax discarded (kept in fp)

Also dumps the top-50 observer activation ranges from config A.

Runs on this machine; takes ~10-15 min. No phone needed.
"""
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from export_flux2_klein_qnn import (  # noqa: E402
    Flux2TransformerWrapper,
    build_transformer_inputs,
    configure_local_tooling,
    load_pipeline,
)

configure_local_tooling()

from executorch.backends.qualcomm.quantizer.quantizer import (  # noqa: E402
    QnnQuantizer,
    QuantDtype,
)
from executorch.backends.qualcomm.serialization.qc_schema import (  # noqa: E402
    QnnExecuTorchBackendType,
    QcomChipset,
)
from torchao.quantization.pt2e.quantize_pt2e import (  # noqa: E402
    convert_pt2e,
    prepare_pt2e,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("diag")


def diff(name, ref, q):
    ref = ref.detach().float()
    q = q.detach().float()
    abs_diff = (ref - q).abs()
    rel = abs_diff / (ref.abs() + 1e-8)
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0), q.flatten().unsqueeze(0)
    ).item()
    snr_db = 20 * torch.log10(
        ref.norm() / ((ref - q).norm() + 1e-12)
    ).item()
    log.info(
        "%-40s max=%.4f mean=%.5f rel_mean=%.4f cos=%.5f SNR=%.2fdB ref_max=%.3f",
        name,
        abs_diff.max().item(),
        abs_diff.mean().item(),
        rel.mean().item(),
        cos,
        snr_db,
        ref.abs().max().item(),
    )


def quantize_and_run(captured, sample_inputs, calibration_data,
                     is_per_channel, discard_softmax, discard_layernorm=False):
    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=QcomChipset.SM8850,
    )
    quantizer.set_default_quant_config(
        QuantDtype.use_8a8w,
        is_conv_per_channel=is_per_channel,
        is_linear_per_channel=is_per_channel,
    )
    discards = []
    if discard_softmax:
        discards += [
            torch.ops.aten._safe_softmax.default,
            torch.ops.aten.softmax.int,
        ]
    if discard_layernorm:
        discards += [torch.ops.aten.native_layer_norm.default]
    if discards:
        quantizer.add_discard_ops(discards)

    log.info("prepare_pt2e ...")
    prepared = prepare_pt2e(captured, quantizer)

    log.info("calibrating with %d samples ...", len(calibration_data))
    with torch.no_grad():
        for i, cal in enumerate(calibration_data):
            if not isinstance(cal, tuple):
                cal = (cal,)
            prepared(*cal)
            if (i + 1) % 5 == 0:
                log.info("  cal %d/%d", i + 1, len(calibration_data))

    log.info("convert_pt2e ...")
    converted = convert_pt2e(prepared)

    log.info("running converted (fake-quant) forward on probe ...")
    with torch.no_grad():
        out = converted(*sample_inputs)
    return out, prepared


def dump_observer_ranges(prepared, top_k=50):
    ranges = []
    for name, mod in prepared.named_modules():
        if hasattr(mod, "min_val") and hasattr(mod, "max_val"):
            mn = float(mod.min_val) if mod.min_val.numel() == 1 else float(mod.min_val.min())
            mx = float(mod.max_val) if mod.max_val.numel() == 1 else float(mod.max_val.max())
            ranges.append((mx - mn, mn, mx, name))
    ranges.sort(reverse=True)
    log.info("=== TOP %d OBSERVER RANGES ===", top_k)
    for r, mn, mx, name in ranges[:top_k]:
        log.info("  range=%.3f  [%.3f, %.3f]  %s", r, mn, mx, name)
    log.info("=== BOTTOM %d (smallest, possibly degenerate) ===", 20)
    for r, mn, mx, name in ranges[-20:]:
        log.info("  range=%.6f  [%.6f, %.6f]  %s", r, mn, mx, name)


def main():
    log.info("Loading FLUX.2-klein pipeline (fp32) ...")
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()

    sample_inputs = build_transformer_inputs(
        pipe, 512, 512, 512, dtype=torch.float32, num_img2img_images=0,
    )

    log.info("Loading real calibration samples ...")
    calibration_data = torch.load(
        str(Path(__file__).parent / "calibration_data" / "calibration_transformer.pt"),
        weights_only=False,
    )
    # 5 samples per config × 5 configs ≈ 12 min total. Plenty for relative SNR.
    calibration_data = calibration_data[:5]
    log.info("Using %d calibration samples per config", len(calibration_data))

    # Use the FIRST calibration sample as our probe so it's in-distribution.
    probe = calibration_data[0]
    if not isinstance(probe, tuple):
        probe = (probe,)
    log.info("Probe shapes: %s", [tuple(x.shape) for x in probe])

    log.info("=== fp32 reference forward ===")
    t0 = time.time()
    with torch.no_grad():
        ref_out = model(*probe)
    log.info("ref forward took %.1fs", time.time() - t0)
    if isinstance(ref_out, tuple):
        ref_out = ref_out[0]

    log.info("=== capturing pre-export ===")
    t0 = time.time()
    captured = torch.export.export(model, probe, strict=True).module()
    log.info("export took %.1fs", time.time() - t0)
    # We need a separate captured graph per quantizer (prepare_pt2e mutates).
    # Cheapest: re-export when needed. But that's slow. Instead, deepcopy the captured GM.
    import copy

    log.info("\n############ CONFIG A: per-channel weights ############")
    cap_a = copy.deepcopy(captured)
    out_a, prep_a = quantize_and_run(
        cap_a, probe, calibration_data,
        is_per_channel=True, discard_softmax=False,
    )
    if isinstance(out_a, tuple):
        out_a = out_a[0]
    diff("A: per-channel", ref_out, out_a)
    dump_observer_ranges(prep_a, top_k=30)
    del cap_a, prep_a

    log.info("\n############ CONFIG B: per-tensor weights (April's) ############")
    cap_b = copy.deepcopy(captured)
    out_b, _ = quantize_and_run(
        cap_b, probe, calibration_data,
        is_per_channel=False, discard_softmax=False,
    )
    if isinstance(out_b, tuple):
        out_b = out_b[0]
    diff("B: per-tensor", ref_out, out_b)
    del cap_b

    log.info("\n############ CONFIG C: per-channel + softmax discarded ############")
    cap_c = copy.deepcopy(captured)
    out_c, _ = quantize_and_run(
        cap_c, probe, calibration_data,
        is_per_channel=True, discard_softmax=True,
    )
    if isinstance(out_c, tuple):
        out_c = out_c[0]
    diff("C: per-channel + no-softmax-q", ref_out, out_c)
    del cap_c

    log.info("\n############ CONFIG D: per-channel + LN discarded (our device path) ############")
    cap_d = copy.deepcopy(captured)
    out_d, _ = quantize_and_run(
        cap_d, probe, calibration_data,
        is_per_channel=True, discard_softmax=False, discard_layernorm=True,
    )
    if isinstance(out_d, tuple):
        out_d = out_d[0]
    diff("D: per-channel + no-LN-q  ", ref_out, out_d)
    del cap_d

    log.info("\n############ CONFIG E: per-channel + softmax+LN discarded ############")
    cap_e = copy.deepcopy(captured)
    out_e, _ = quantize_and_run(
        cap_e, probe, calibration_data,
        is_per_channel=True, discard_softmax=True, discard_layernorm=True,
    )
    if isinstance(out_e, tuple):
        out_e = out_e[0]
    diff("E: per-channel + no-sm+no-LN", ref_out, out_e)
    del cap_e

    log.info("\n=== SUMMARY (cosine to fp32) ===")
    log.info("Higher SNR / cos closer to 1.0 = better. Big drop = where noise is.")


if __name__ == "__main__":
    main()
