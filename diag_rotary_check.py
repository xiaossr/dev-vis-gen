"""Quick sanity: ensure rotary head-split produces identical output to non-split."""
import logging
import os
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("rot_chk")

def main():
    configure_local_tooling()
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    sample = build_transformer_inputs(pipe, 512, 512, 512, dtype=torch.float32, num_img2img_images=0)

    # No-split baseline
    os.environ["FLUX_ROTARY_HEAD_SPLIT"] = "1"
    model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
    with torch.no_grad():
        out_nosplit = model(*sample)

    # 2-way split
    os.environ["FLUX_ROTARY_HEAD_SPLIT"] = "2"
    with torch.no_grad():
        out_split = model(*sample)

    diff = (out_nosplit - out_split).abs().max().item()
    rel = (out_nosplit - out_split).norm().item() / out_nosplit.norm().item()
    log.info("max abs diff: %.6e, rel: %.6e", diff, rel)
    if diff < 1e-4:
        log.info("OK: rotary split mathematically equivalent")
    else:
        log.error("MISMATCH! split changed math")

if __name__ == "__main__":
    main()
