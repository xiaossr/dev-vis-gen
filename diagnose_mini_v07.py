#!/usr/bin/env python3
"""
Fast diagnostic: try to export a mini-transformer slice on ET 0.7.0 path.

Use to reproduce/diagnose w8a8 compile issues on V81 without waiting hours
for a full transformer export.

Usage:
    .venv-et07/bin/python diagnose_mini_v07.py --num_single 2
"""

import argparse
import logging
import os
import torch
import torch.nn as nn

# trigger env setup
from export_flux2_klein_qnn_v07 import (  # noqa: F401
    export_component_to_qnn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("diag_mini_v07")


class MiniFluxTransformer(nn.Module):
    """Trim the block lists but keep the full forward — real modulation fanout."""

    def __init__(self, src, num_double: int = 1, num_single: int = 2):
        super().__init__()
        self.transformer = src
        src.transformer_blocks = nn.ModuleList(list(src.transformer_blocks)[:num_double])
        src.single_transformer_blocks = nn.ModuleList(
            list(src.single_transformer_blocks)[:num_single]
        )

    def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids):
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )[0]


def build_inputs(transformer, height=512, width=512, max_text_len=512):
    vae_sf = 8
    patch_h = (height // vae_sf) // 2
    patch_w = (width // vae_sf) // 2
    num_tokens = patch_h * patch_w
    in_channels = transformer.config.in_channels
    joint_dim = transformer.config.joint_attention_dim

    torch.manual_seed(0)
    hidden_states = torch.randn(1, num_tokens, in_channels, dtype=torch.float32) * 0.7
    encoder_hidden_states = torch.randn(1, max_text_len, joint_dim, dtype=torch.float32) * 0.02
    timestep = torch.tensor([1.0], dtype=torch.float32)
    img_ids = torch.zeros(1, num_tokens, 4, dtype=torch.float32)
    txt_ids = torch.zeros(1, max_text_len, 4, dtype=torch.float32)
    log.info("inputs: hs=%s ehs=%s", list(hidden_states.shape), list(encoder_hidden_states.shape))
    return (hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_double", type=int, default=1)
    ap.add_argument("--num_single", type=int, default=2)
    ap.add_argument("--soc", default="SM8850")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--quant_dtype", default="8a8w")
    ap.add_argument("--output_path", default="./mini_v07.pte")
    args = ap.parse_args()

    from export_flux2_klein_xnnpack import load_pipeline
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    mini = MiniFluxTransformer(
        pipe.transformer, num_double=args.num_double, num_single=args.num_single,
    ).eval().cpu()
    sample_inputs = build_inputs(pipe.transformer)

    log.info(
        "Exporting mini: %d double + %d single blocks, quant=%s, soc=%s",
        args.num_double, args.num_single,
        "fp16" if not args.quantize else args.quant_dtype, args.soc,
    )
    export_component_to_qnn(
        mini, sample_inputs, args.output_path,
        soc_model=args.soc, quantize=args.quantize,
        quant_dtype=args.quant_dtype, num_calibration=1,
    )
    log.info("SUCCESS: %s (%.1f MB)",
             args.output_path, os.path.getsize(args.output_path) / 1e6)


if __name__ == "__main__":
    main()
