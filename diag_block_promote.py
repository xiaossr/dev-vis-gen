"""Promote only D00-D04 double-stream blocks to 16a8w; rest stays 8a8w.

Predicate matches by nn_module_stack containing
'transformer_blocks.{0,1,2,3,4}' (NOT single_transformer_blocks).
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
    QnnQuantizer, QuantDtype, ModuleQConfig,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType, QcomChipset,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("blkprom")


def diff(name, ref, q):
    ref = ref.detach().float(); q = q.detach().float()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0), q.flatten().unsqueeze(0)).item()
    snr_db = 20 * torch.log10(ref.norm() / ((ref - q).norm() + 1e-12)).item()
    log.info("[%s]  max=%.4f  cos=%.5f  SNR=%.2fdB",
             name, (ref - q).abs().max().item(), cos, snr_db)


def make_predicate(target_module_prefixes):
    """Return a predicate that returns True iff node's nn_module_stack contains
    any of the target prefixes (and NOT a single_transformer_blocks if the
    'transformer_blocks' prefix is in target — to avoid name collision)."""
    def pred(node):
        stk = node.meta.get("nn_module_stack", {}) if hasattr(node, "meta") else {}
        if not stk:
            return False
        for entry in stk.values():
            mod = entry[0] if isinstance(entry, tuple) else str(entry)
            for prefix in target_module_prefixes:
                if mod.startswith(prefix):
                    # Avoid double-stream prefix accidentally matching single
                    # block — both share 'transformer_blocks' substring but
                    # single is 'single_transformer_blocks.X', double is
                    # 'transformer_blocks.X'.
                    return True
        return False
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", nargs="+", default=["0", "1", "2", "3", "4"],
                   help="indices of double-stream blocks to promote to 16a8w")
    p.add_argument("--ncal", type=int, default=5)
    args = p.parse_args()

    target_prefixes = [
        f"transformer.transformer_blocks.{i}" for i in args.blocks
    ]
    log.info("Promoting modules with prefix: %s", target_prefixes)

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

    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=QcomChipset.SM8850,
    )
    quantizer.set_default_quant_config(
        QuantDtype.use_8a8w,
        is_conv_per_channel=True, is_linear_per_channel=True,
        act_observer=HistogramObserver,
    )

    pred = make_predicate(target_prefixes)
    matched = [0]
    def counting_pred(n):
        ok = pred(n)
        if ok: matched[0] += 1
        return ok

    quantizer.set_submodule_qconfig_list([
        (counting_pred, ModuleQConfig(
            quant_dtype=QuantDtype.use_16a8w,
            is_conv_per_channel=True, is_linear_per_channel=True,
            act_observer=HistogramObserver,
        )),
    ])

    log.info("prepare ...")
    prepared = prepare_pt2e(captured, quantizer)
    log.info("Predicate matched %d FX nodes", matched[0])
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
    diff(f"D{'+'.join(args.blocks)}@16a8w (matched={matched[0]})", ref, out)


if __name__ == "__main__":
    main()
