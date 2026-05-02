"""
Map activation_post_process_N observer names to their actual FX node target.
Tells us WHAT op produces each insane-range activation. Fast (~3 min).
"""
import logging
import sys
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
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("obs_map")


def main():
    log.info("Loading pipeline ...")
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
    sample_inputs = build_transformer_inputs(
        pipe, 512, 512, 512, dtype=torch.float32, num_img2img_images=0,
    )
    log.info("Pre-export ...")
    captured = torch.export.export(model, sample_inputs, strict=True).module()

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

    # Calibrate with 1 sample so observers populate min/max.
    cal = torch.load(
        str(Path(__file__).parent / "calibration_data" / "calibration_transformer.pt"),
        weights_only=False,
    )[0]
    if not isinstance(cal, tuple):
        cal = (cal,)
    log.info("calibrating with 1 sample ...")
    with torch.no_grad():
        prepared(*cal)

    # Build map: observer module name -> the node that PRODUCES the value
    # being observed (the call_module observer reads from its first input).
    log.info("Mapping observer name -> producing node ...")
    obs_to_producer = {}
    for node in prepared.graph.nodes:
        if node.op == "call_module" and node.target.startswith("activation_post_process_"):
            # The node being observed is node.args[0].
            producer = node.args[0]
            obs_to_producer[node.target] = producer

    # Collect (range, mn, mx, obs_name, producer_target, producer_meta) for non-degenerate.
    rows = []
    for name, mod in prepared.named_modules():
        if hasattr(mod, "min_val") and hasattr(mod, "max_val"):
            mn = float(mod.min_val) if mod.min_val.numel() == 1 else float(mod.min_val.min())
            mx = float(mod.max_val) if mod.max_val.numel() == 1 else float(mod.max_val.max())
            r = mx - mn
            producer = obs_to_producer.get(name)
            target = "<unknown>"
            shape = "<?>"
            stack = "<?>"
            if producer is not None:
                target = str(producer.target)
                if "val" in producer.meta:
                    val = producer.meta["val"]
                    if hasattr(val, "shape"):
                        shape = str(tuple(val.shape))
                # Get nn_module_stack for context (which submodule this came from)
                stk = producer.meta.get("nn_module_stack", {})
                if stk:
                    last = list(stk.values())[-1]
                    if isinstance(last, tuple):
                        stack = last[0]
            rows.append((r, mn, mx, name, target, shape, stack))

    rows.sort(reverse=True)
    log.info("=== TOP 40 RANGES ===")
    for r, mn, mx, name, target, shape, stack in rows[:40]:
        log.info(
            "  range=%-9.2f [%9.3f, %9.3f]  %-30s  shape=%-25s  op=%-30s  in=%s",
            r, mn, mx, name, shape, target, stack,
        )


if __name__ == "__main__":
    main()
