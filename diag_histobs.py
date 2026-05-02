"""Just config F (HistogramObserver), full traceback."""
import logging
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.ao.quantization.observer import HistogramObserver

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
log = logging.getLogger("histobs")


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

    log.info("export ...")
    captured = torch.export.export(model, probe, strict=True).module()

    log.info("Building QnnQuantizer with HistogramObserver ...")
    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=QcomChipset.SM8850,
    )
    try:
        quantizer.set_default_quant_config(
            QuantDtype.use_8a8w,
            is_conv_per_channel=True,
            is_linear_per_channel=True,
            act_observer=HistogramObserver,
        )
    except Exception:
        log.error("set_default_quant_config failed:")
        traceback.print_exc()
        return

    try:
        log.info("prepare_pt2e ...")
        prepared = prepare_pt2e(captured, quantizer)
    except Exception:
        log.error("prepare_pt2e failed:")
        traceback.print_exc()
        return

    log.info("calibrating ...")
    with torch.no_grad():
        for i, c in enumerate(cal):
            if not isinstance(c, tuple):
                c = (c,)
            try:
                prepared(*c)
            except Exception:
                log.error("forward(cal=%d) failed:", i)
                traceback.print_exc()
                return

    log.info("convert_pt2e ...")
    try:
        converted = convert_pt2e(prepared)
    except Exception:
        log.error("convert_pt2e failed:")
        traceback.print_exc()
        return

    log.info("DONE — HistogramObserver run completed without error.")


if __name__ == "__main__":
    main()
