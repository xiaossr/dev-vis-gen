"""Verify that set_submodule_qconfig_list with use_16a8w actually changes the
input observer's bit-width on a promoted Linear. Sanity test before chasing
deeper hypotheses.

Approach: build the linear-only-discard a8w8 baseline + promote
context_embedder to use_16a8w via set_submodule_qconfig_list. Run prepare_pt2e
to insert observers. Walk the prepared graph, find the fake-quant placed on
context_embedder's input edge, dump its quant_min/quant_max.

If quant_max == 127  -> override didn't apply (still int8)
If quant_max == 32767 -> override applied (int16)
"""
import logging, sys
from pathlib import Path

# Vendored torchao first (matches diag_promote_context_embedder.py setup)
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
from executorch.backends.qualcomm.quantizer.quantizer import (
    QnnQuantizer, QuantDtype, ModuleQConfig,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType, QcomChipset,
)
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("verify")


def build_quantizer(promote_ce=False):
    q = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=QcomChipset.SM8850,
    )
    q.set_default_quant_config(
        QuantDtype.use_8a8w,
        is_conv_per_channel=True, is_linear_per_channel=True,
        act_observer=HistogramObserver,
    )
    aten = torch.ops.aten
    keep = {aten.linear.default, aten.conv2d.default, aten.conv1d.default}
    discard = [op for op in q.quant_ops if op not in keep]
    q.add_discard_ops(discard)

    if promote_ce:
        def pred(node):
            stk = node.meta.get("nn_module_stack", {}) if hasattr(node, "meta") else {}
            for entry in stk.values():
                mod = entry[0] if isinstance(entry, tuple) else str(entry)
                if mod == "transformer.context_embedder" or \
                   mod.startswith("transformer.context_embedder."):
                    return True
            return False
        q.set_submodule_qconfig_list([
            (pred, ModuleQConfig(
                quant_dtype=QuantDtype.use_16a8w,
                is_conv_per_channel=True, is_linear_per_channel=True,
                act_observer=HistogramObserver,
            )),
        ])
    return q


def find_context_embedder_linear_node(gm):
    """Find the FX node that is the context_embedder's aten.linear call."""
    for node in gm.graph.nodes:
        if node.op != "call_function": continue
        if node.target.__name__ != "linear.default": continue
        stk = node.meta.get("nn_module_stack", {})
        for entry in stk.values():
            mod = entry[0] if isinstance(entry, tuple) else str(entry)
            if mod == "transformer.context_embedder" or \
               mod.startswith("transformer.context_embedder."):
                return node
    return None


def dump_observer_for_node(gm, node, label):
    """The Linear node's first input is the activation. Walk to find the
    activation observer (FakeQuantize / observer) attached to that edge."""
    log.info("=== %s ===", label)
    log.info("  Linear node: %s", node.name)
    if not node.args:
        log.info("  no args")
        return

    act_input = node.args[0]
    log.info("  Activation input: %s (op=%s, target=%s)", act_input.name, act_input.op,
             getattr(act_input, "target", None))

    # In prepare_pt2e, observers are inserted as call_module nodes whose
    # name typically contains 'activation_post_process' or fake_quant.
    # They sit *between* the producer and consumer.

    def get_observer_module(module_node):
        """If module_node is a call_module pointing to an observer, return it."""
        if module_node.op != "call_module":
            return None
        sub = gm
        for part in module_node.target.split("."):
            sub = getattr(sub, part)
        return sub

    obs = get_observer_module(act_input)
    if obs is None:
        # Maybe upstream of act_input is an observer — search backward
        log.info("  act_input is not an observer node; checking its predecessors ...")
        # Find any call_module nodes feeding into the linear that are observers
        for n in gm.graph.nodes:
            if n.op == "call_module" and node in n.users:
                obs = get_observer_module(n)
                if obs is not None:
                    log.info("  found observer feeding linear: %s", n.name)
                    break
    if obs is None:
        log.info("  NO observer found on input edge")
        return

    log.info("  observer class: %s", type(obs).__name__)
    for attr in ["quant_min", "quant_max", "dtype", "qscheme",
                 "ch_axis", "is_dynamic"]:
        if hasattr(obs, attr):
            log.info("  %s = %s", attr, getattr(obs, attr))


def main():
    log.info("Loading ...")
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    cal = torch.load(
        str(Path(__file__).parent / "calibration_data" / "calibration_transformer.pt"),
        weights_only=False)[:1]
    probe = cal[0] if isinstance(cal[0], tuple) else (cal[0],)

    model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()

    log.info("torch.export ...")
    captured = torch.export.export(model, probe, strict=True).module()

    for label, promote in [("BASELINE a8w8", False), ("PROMOTED context_embedder a16w8", True)]:
        log.info("\n############ %s ############", label)
        # Re-export each time to keep modules fresh
        cap = torch.export.export(model, probe, strict=True).module()
        q = build_quantizer(promote_ce=promote)
        prepared = prepare_pt2e(cap, q)
        node = find_context_embedder_linear_node(prepared)
        if node is None:
            log.error("Could not find context_embedder linear node")
            continue
        dump_observer_for_node(prepared, node, label)


if __name__ == "__main__":
    main()
