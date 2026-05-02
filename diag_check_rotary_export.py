"""Check if rotary chunking actually changed the exported graph shapes."""
import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from export_flux2_klein_qnn import (
    Flux2TransformerWrapper, build_transformer_inputs,
    configure_local_tooling, load_pipeline,
)
configure_local_tooling()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("c")

def main():
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    sample = build_transformer_inputs(pipe, 512, 512, 512, dtype=torch.float32, num_img2img_images=0)

    for split in (1, 2):
        os.environ["FLUX_ROTARY_HEAD_SPLIT"] = str(split)
        m = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
        ep = torch.export.export(m, sample, strict=True)
        gm = ep.module()
        # count muls by output shape
        from collections import Counter
        shape_count = Counter()
        for n in gm.graph.nodes:
            if n.op == "call_function" and "mul" in str(n.target):
                v = n.meta.get("val")
                if v is not None and hasattr(v, "shape"):
                    shape_count[tuple(int(d) for d in v.shape)] += 1
        log.info("--- split=%d ---", split)
        for sh, cnt in sorted(shape_count.items(), key=lambda x: -x[1])[:10]:
            log.info("  shape=%-30s count=%d", sh, cnt)

if __name__ == "__main__":
    main()
