"""Linear-only-discard with use_16a8w as the GLOBAL default (instead of
use_8a8w). Tests whether bit-depth promotion across ALL Linears improves
SNR — distinguishes 'bit-depth doesn't help these 31 Linears' from
'bit-depth doesn't help any Linear'.

Baseline (a8w8 lin-only-discard): +3.60 dB / cos 0.82
Hypothesis: a16w8 lin-only-discard should give a big jump if quant noise
is dominated by Linear input/output 8-bit rounding.
"""
import logging, sys, json
from pathlib import Path

_REPO = Path(__file__).resolve().parent
_VENDORED_AO = _REPO / "executorch" / "third-party" / "ao"
if _VENDORED_AO.exists():
    sys.path.insert(0, str(_VENDORED_AO))

import torch

sys.path.insert(0, str(_REPO))
from export_flux2_klein_qnn import (
    Flux2TransformerWrapper, configure_local_tooling, load_pipeline,
)
configure_local_tooling()

from torchao.quantization.pt2e.observer import HistogramObserver
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType, QcomChipset,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("a16w8")


def diff(name, ref, q):
    ref = ref.detach().float(); q = q.detach().float()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0), q.flatten().unsqueeze(0)).item()
    snr = 20 * torch.log10(ref.norm() / ((ref - q).norm() + 1e-12)).item()
    log.info("[%s]  cos=%.5f  SNR=%.2fdB", name, cos, snr)
    return {"name": name, "cos": cos, "snr_db": snr}


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

    captured = torch.export.export(model, probe, strict=True).module()

    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=QcomChipset.SM8850,
    )
    quantizer.set_default_quant_config(
        QuantDtype.use_16a8w,
        is_conv_per_channel=True, is_linear_per_channel=True,
        act_observer=HistogramObserver,
    )
    aten = torch.ops.aten
    keep = {aten.linear.default, aten.conv2d.default, aten.conv1d.default}
    discard = [op for op in quantizer.quant_ops if op not in keep]
    quantizer.add_discard_ops(discard)
    log.info("Discarded %d ops; default = use_16a8w (ALL Linears 16-bit act, 8-bit weight)",
             len(discard))

    log.info("prepare ...")
    prepared = prepare_pt2e(captured, quantizer)
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
    res = diff("lin_only_a16w8_global", ref, out)

    out_path = Path(__file__).parent / "lin_only_a16w8_results.json"
    out_path.write_text(json.dumps(res, indent=2))
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
