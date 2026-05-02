"""
Test the timestep-hypothesis. Compare:
  L: calibrate with t=0.0 samples only;   probe at t=0.0  (matched)
  M: calibrate with t=1.0 samples only;   probe at t=1.0  (matched)
  N: calibrate with t=0.0 samples only;   probe at t=1.0  (mismatch - control)

If L and M are both clean (>~16 dB SNR) and N is noise, this proves that
per-timestep static scales recover the signal — the same scale just can't
serve all timesteps simultaneously.
"""
import copy
import logging
import sys
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
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("ts")


def diff(name, ref, q):
    ref = ref.detach().float()
    q = q.detach().float()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0), q.flatten().unsqueeze(0)
    ).item()
    snr_db = 20 * torch.log10(
        ref.norm() / ((ref - q).norm() + 1e-12)
    ).item()
    log.info(
        "%-50s max=%.4f cos=%.5f SNR=%.2fdB",
        name, (ref - q).abs().max().item(), cos, snr_db,
    )


def quantize_and_run(captured, probe, calib):
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
    log.info("calibrating with %d samples ...", len(calib))
    with torch.no_grad():
        for c in calib:
            if not isinstance(c, tuple):
                c = (c,)
            prepared(*c)
    log.info("convert_pt2e ...")
    converted = convert_pt2e(prepared)
    log.info("forward ...")
    with torch.no_grad():
        out = converted(*probe)
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
    cal_all = torch.load(
        str(Path(__file__).parent / "calibration_data" / "calibration_transformer.pt"),
        weights_only=False,
    )

    cal_t0 = [c for c in cal_all if c[2].item() == 0.0]
    cal_t1 = [c for c in cal_all if c[2].item() == 1.0]
    log.info("t=0.0 samples: %d, t=1.0 samples: %d", len(cal_t0), len(cal_t1))

    probe_t0 = cal_t0[0]
    probe_t1 = cal_t1[0]

    log.info("=== fp32 ref @ t=0.0 ===")
    with torch.no_grad():
        ref_t0 = model(*probe_t0)
    if isinstance(ref_t0, tuple):
        ref_t0 = ref_t0[0]

    log.info("=== fp32 ref @ t=1.0 ===")
    with torch.no_grad():
        ref_t1 = model(*probe_t1)
    if isinstance(ref_t1, tuple):
        ref_t1 = ref_t1[0]

    log.info("=== capturing pre-export ===")
    captured = torch.export.export(model, probe_t0, strict=True).module()

    log.info("\n############ L: calib t=0.0 only, probe t=0.0 (matched) ############")
    cap_l = copy.deepcopy(captured)
    out_l = quantize_and_run(cap_l, probe_t0, cal_t0[:5])
    diff("L: cal=t0 probe=t0 (matched)", ref_t0, out_l)
    del cap_l

    log.info("\n############ M: calib t=1.0 only, probe t=1.0 (matched) ############")
    cap_m = copy.deepcopy(captured)
    out_m = quantize_and_run(cap_m, probe_t1, cal_t1[:5])
    diff("M: cal=t1 probe=t1 (matched)", ref_t1, out_m)
    del cap_m

    log.info("\n############ N: calib t=0.0 only, probe t=1.0 (MISMATCH control) ############")
    cap_n = copy.deepcopy(captured)
    out_n = quantize_and_run(cap_n, probe_t1, cal_t0[:5])
    diff("N: cal=t0 probe=t1 (mismatch)", ref_t1, out_n)
    del cap_n


if __name__ == "__main__":
    main()
