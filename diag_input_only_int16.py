"""Test: promote ONLY encoder_hidden_states (graph input) to 16a8w; rest stays
8a8w. If this alone recovers SNR significantly, encoder input was the
bottleneck and we can implement a host-side rescale preproc instead."""
import logging, sys
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
    QnnQuantizer, QuantDtype, ModuleQConfig,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType, QcomChipset,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("input16")


def diff(name, ref, q):
    ref = ref.detach().float(); q = q.detach().float()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0), q.flatten().unsqueeze(0)).item()
    snr_db = 20 * torch.log10(ref.norm() / ((ref - q).norm() + 1e-12)).item()
    log.info("[%s]  max=%.4f  cos=%.5f  SNR=%.2fdB",
             name, (ref - q).abs().max().item(), cos, snr_db)


def main():
    log.info("Loading ...")
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
    sample = build_transformer_inputs(pipe, 512, 512, 512, dtype=torch.float32, num_img2img_images=0)
    cal = torch.load(
        str(Path(__file__).parent / "calibration_data" / "calibration_transformer.pt"),
        weights_only=False)[:5]
    probe = cal[0] if isinstance(cal[0], tuple) else (cal[0],)

    log.info("fp32 ref ...")
    with torch.no_grad():
        ref = model(*probe)
    if isinstance(ref, tuple): ref = ref[0]

    log.info("export ...")
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

    matched = [0]
    def is_target(node):
        if node.name == "encoder_hidden_states":
            matched[0] += 1
            return True
        # also catch its direct downstream activation_post_process which
        # otherwise falls back to default 8a8w
        return False

    quantizer.set_submodule_qconfig_list([
        (is_target, ModuleQConfig(
            quant_dtype=QuantDtype.use_16a8w,
            is_conv_per_channel=True, is_linear_per_channel=True,
            act_observer=HistogramObserver,
        )),
    ])

    log.info("prepare ...")
    prepared = prepare_pt2e(captured, quantizer)
    log.info("matched %d nodes", matched[0])
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
    diff("encoder_hidden_states-16a8w only", ref, out)


if __name__ == "__main__":
    main()
