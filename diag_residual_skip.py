"""
Test: keep 8a8w everywhere EXCEPT discard quantization on aten.add.Tensor
(the residual writes). The data shows residual ADDs are the outlier-heavy
tensors. If we leave them in fp, can we keep 8 bits everywhere else?
"""
import copy
import logging
import sys
import time
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

from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
    QcomChipset,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("res_skip")


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


def quantize_and_run(captured, probe, calib, *, dtype, discard_ops=None,
                     act_observer=None):
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
    if discard_ops:
        quantizer.add_discard_ops(discard_ops)

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
    cal = torch.load(
        str(Path(__file__).parent / "calibration_data" / "calibration_transformer.pt"),
        weights_only=False,
    )[:5]
    probe = cal[0] if isinstance(cal[0], tuple) else (cal[0],)

    log.info("=== fp32 reference forward ===")
    with torch.no_grad():
        ref_out = model(*probe)
    if isinstance(ref_out, tuple):
        ref_out = ref_out[0]

    log.info("=== capturing pre-export ===")
    captured = torch.export.export(model, probe, strict=True).module()

    log.info("\n############ J: 8a8w + Histogram + skip-add (residual unquantized) ############")
    cap_j = copy.deepcopy(captured)
    out_j = quantize_and_run(
        cap_j, probe, cal, dtype=QuantDtype.use_8a8w,
        act_observer=HistogramObserver,
        discard_ops=[torch.ops.aten.add.Tensor],
    )
    diff("J: 8a8w + Hist + skip-add ", ref_out, out_j)
    del cap_j

    log.info("\n############ K: 8a8w + Histogram + skip-add+mul (modulation also in fp) ############")
    cap_k = copy.deepcopy(captured)
    out_k = quantize_and_run(
        cap_k, probe, cal, dtype=QuantDtype.use_8a8w,
        act_observer=HistogramObserver,
        discard_ops=[torch.ops.aten.add.Tensor, torch.ops.aten.mul.Tensor],
    )
    diff("K: 8a8w + Hist + skip-add+mul", ref_out, out_k)
    del cap_k


if __name__ == "__main__":
    main()
