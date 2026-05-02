"""
Round 2 of PTQ-noise diagnostic. Tests the leverage points that *should* help
given Round 1's finding (residual-stream outliers ±6000 crush int8 dynamic range):

  F: per-channel weights + HistogramObserver (percentile-style outlier clipping)
  G: 16a8w everywhere (control: int16 activations, big enough to span outliers)
  H: mixed — 16a8w on deep single-transformer-blocks 10-23, 8a8w elsewhere

Reuses 5 calibration samples per config. ~12 min.
"""
import copy
import logging
import sys
import time
from pathlib import Path

import torch
from torchao.quantization.pt2e.observer import HistogramObserver

sys.path.insert(0, str(Path(__file__).parent))
from export_flux2_klein_qnn import (  # noqa: E402
    Flux2TransformerWrapper,
    build_transformer_inputs,
    configure_local_tooling,
    load_pipeline,
)

configure_local_tooling()

from executorch.backends.qualcomm.quantizer.quantizer import (  # noqa: E402
    ModuleQConfig,
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
log = logging.getLogger("diag2")


def diff(name, ref, q):
    ref = ref.detach().float()
    q = q.detach().float()
    abs_diff = (ref - q).abs()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0), q.flatten().unsqueeze(0)
    ).item()
    snr_db = 20 * torch.log10(
        ref.norm() / ((ref - q).norm() + 1e-12)
    ).item()
    log.info(
        "%-50s max=%.4f mean=%.5f cos=%.5f SNR=%.2fdB",
        name, abs_diff.max().item(), abs_diff.mean().item(), cos, snr_db,
    )


def is_deep_single_block(node):
    """Match nodes inside transformer.single_transformer_blocks.{10..23}."""
    stack = node.meta.get("nn_module_stack", {})
    for entry in stack.values():
        if isinstance(entry, tuple):
            qual = entry[0]
            if "single_transformer_blocks" in qual:
                parts = qual.split(".")
                try:
                    idx = parts.index("single_transformer_blocks")
                    if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                        if int(parts[idx + 1]) >= 10:
                            return True
                except ValueError:
                    pass
    return False


def quantize_and_run(captured, sample_inputs, calibration_data,
                     dtype, act_observer=None, mixed_precision_predicate=None):
    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=QcomChipset.SM8850,
    )
    if act_observer is not None:
        quantizer.set_default_quant_config(
            dtype, is_conv_per_channel=True, is_linear_per_channel=True,
            act_observer=act_observer,
        )
    else:
        quantizer.set_default_quant_config(
            dtype, is_conv_per_channel=True, is_linear_per_channel=True,
        )
    if mixed_precision_predicate is not None:
        quantizer.set_submodule_qconfig_list([
            (mixed_precision_predicate,
             ModuleQConfig(
                 quant_dtype=QuantDtype.use_16a8w,
                 is_conv_per_channel=True,
                 is_linear_per_channel=True,
             )),
        ])

    log.info("prepare_pt2e ...")
    prepared = prepare_pt2e(captured, quantizer)

    log.info("calibrating with %d samples ...", len(calibration_data))
    with torch.no_grad():
        for i, cal in enumerate(calibration_data):
            if not isinstance(cal, tuple):
                cal = (cal,)
            prepared(*cal)
            log.info("  cal %d/%d", i + 1, len(calibration_data))

    log.info("convert_pt2e ...")
    converted = convert_pt2e(prepared)

    log.info("running converted (fake-quant) forward on probe ...")
    with torch.no_grad():
        out = converted(*sample_inputs)
    if isinstance(out, tuple):
        out = out[0]
    return out


def main():
    log.info("Loading pipeline ...")
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()

    sample_inputs = build_transformer_inputs(
        pipe, 512, 512, 512, dtype=torch.float32, num_img2img_images=0,
    )

    calibration_data = torch.load(
        str(Path(__file__).parent / "calibration_data" / "calibration_transformer.pt"),
        weights_only=False,
    )[:5]
    log.info("Using %d calibration samples per config", len(calibration_data))

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

    log.info("\n############ F: 8a8w + HistogramObserver (percentile clipping) ############")
    try:
        cap_f = copy.deepcopy(captured)
        out_f = quantize_and_run(cap_f, probe, calibration_data,
                                  dtype=QuantDtype.use_8a8w,
                                  act_observer=HistogramObserver)
        diff("F: 8a8w + HistogramObserver        ", ref_out, out_f)
        del cap_f
    except Exception as e:
        log.error("Config F failed: %r", e)

    log.info("\n############ I: 16a8w + HistogramObserver (combined) ############")
    try:
        cap_i = copy.deepcopy(captured)
        out_i = quantize_and_run(cap_i, probe, calibration_data,
                                  dtype=QuantDtype.use_16a8w,
                                  act_observer=HistogramObserver)
        diff("I: 16a8w + HistogramObserver       ", ref_out, out_i)
        del cap_i
    except Exception as e:
        log.error("Config I failed: %r", e)

    log.info("\n=== SUMMARY ===")
    log.info("Higher SNR / cos closer to 1.0 = better.")
    log.info("Recall: A from prior run was SNR=-2.50 dB, cos=-0.15 (pure noise).")


if __name__ == "__main__":
    main()
