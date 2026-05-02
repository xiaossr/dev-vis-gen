"""Sweep activation observer choices on QNN PT2E to find the 12 dB gap
vs PyTorch naive static w8a8.

Naive PyTorch static w8a8 = 9.9 dB. QNN PT2E w8a8 (HistogramObserver) = -2.6 dB.
12 dB gap is QNN-specific. Try alternative observers to triangulate.
"""
import argparse, copy, json, logging, sys
from pathlib import Path

import torch

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
from torchao.quantization.pt2e.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    PerChannelMinMaxObserver,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("obs")


def diff(name, ref, q):
    ref = ref.detach().float(); q = q.detach().float()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0), q.flatten().unsqueeze(0)).item()
    snr_db = 20 * torch.log10(ref.norm() / ((ref - q).norm() + 1e-12)).item()
    log.info("[%-30s]  max=%.4f  cos=%.5f  SNR=%.2fdB",
             name, (ref - q).abs().max().item(), cos, snr_db)
    return {"name": name, "cos": cos, "snr_db": snr_db}


def run(model, probe, calib, observer_cls, label):
    log.info("\n############ %s ############", label)
    captured = torch.export.export(model, probe, strict=True).module()
    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=QcomChipset.SM8850,
    )
    if observer_cls is None:
        # use the QNN default observer (whatever it is)
        quantizer.set_default_quant_config(
            QuantDtype.use_8a8w,
            is_conv_per_channel=True, is_linear_per_channel=True,
        )
    else:
        quantizer.set_default_quant_config(
            QuantDtype.use_8a8w,
            is_conv_per_channel=True, is_linear_per_channel=True,
            act_observer=observer_cls,
        )
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

    out = run(model, probe, cal, HistogramObserver, "HistogramObserver (baseline)")
    results.append(diff("Histogram", ref, out))

    out = run(model, probe, cal, MinMaxObserver, "MinMaxObserver")
    results.append(diff("MinMax", ref, out))

    out = run(model, probe, cal, MovingAverageMinMaxObserver, "MovingAverageMinMaxObserver")
    results.append(diff("MovingAvgMinMax", ref, out))

    # QNN default (don't override act_observer at all)
    out = run(model, probe, cal, None, "QNN default observer")
    results.append(diff("QnnDefault", ref, out))

    log.info("\n=================== SUMMARY ===================")
    for r in results:
        log.info("  %-25s SNR=%6.2f dB  cos=%.5f", r["name"], r["snr_db"], r["cos"])

    out_path = Path(__file__).parent / "observer_sweep_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
