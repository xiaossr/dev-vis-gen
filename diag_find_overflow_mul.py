"""Find which aten.mul.Tensor in the exported graph has 4.5M+ elements
(i.e. would overflow VTCM at int16). Print module stack for each, plus
the operands' shapes."""
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("find")

def main():
    log.info("Loading ...")
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
    sample = build_transformer_inputs(pipe, 512, 512, 512, dtype=torch.float32, num_img2img_images=0)

    log.info("export ...")
    captured = torch.export.export(model, sample, strict=True).module()

    candidates = []
    n_mul = 0
    for node in captured.graph.nodes:
        if node.op != "call_function":
            continue
        if "mul" not in str(node.target):
            continue
        n_mul += 1
        v = node.meta.get("val")
        if v is None or not hasattr(v, "shape"):
            continue
        sh = tuple(int(d) for d in v.shape)
        n = 1
        for d in sh: n *= d
        if n < 2_000_000:
            continue
        # operand shapes
        op_shapes = []
        for a in node.args:
            if hasattr(a, "meta"):
                av = a.meta.get("val")
                if av is not None and hasattr(av, "shape"):
                    op_shapes.append(tuple(int(d) for d in av.shape))
                else:
                    op_shapes.append("scalar")
            else:
                op_shapes.append(repr(a))
        stk = node.meta.get("nn_module_stack", {})
        mod = "<?>"
        if stk:
            last = list(stk.values())[-1]
            if isinstance(last, tuple): mod = last[0]
        candidates.append({
            "name": node.name,
            "target": str(node.target),
            "out_shape": sh,
            "out_numel": n,
            "args_shapes": op_shapes,
            "module": mod,
        })

    log.info("Total mul-like ops: %d", n_mul)
    log.info("Mul ops with output >=2M elem: %d", len(candidates))

    # focus on the early ones (likely culprits for first-block overflow)
    for i, c in enumerate(candidates[:80]):
        log.info("[%3d] %-30s out=%-25s args=%s mod=%s",
                 i, c["name"], c["out_shape"], c["args_shapes"], c["module"])

    # Check specifically for q-shape (1, ?, 24, 128) or (1, 24, ?, 128) - SDPA scale
    log.info("\n=== Q-SHAPE candidates (likely SDPA scale or RMSNorm rescale) ===")
    for c in candidates:
        sh = c["out_shape"]
        if 24 in sh and 128 in sh:
            log.info("%-30s out=%-25s args=%s mod=%s",
                     c["name"], sh, c["args_shapes"], c["module"])

if __name__ == "__main__":
    main()
