#!/usr/bin/env python3
"""
Reproduce the QNN HTP compile failure on a small slice of the FLUX transformer.

Full-transformer export takes ~3h to hit the failing partition. This script
builds a mini-transformer containing only the shared modulation chain and
a few transformer blocks — enough to create the same "modulation + block"
partition mix that trips RouterX86 on V81 — and runs it through the same
QNN export path. Expected wall time: ~3-5 minutes.

Usage:
    python diagnose_mini_transformer.py --num_single_blocks 2
    python diagnose_mini_transformer.py --num_single_blocks 4
    python diagnose_mini_transformer.py --use_fp16
"""
import argparse
import logging
import os
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("diag_mini")


class MiniFluxTransformer(nn.Module):
    """
    Full Flux2Transformer2DModel with trimmed block lists to reproduce the
    modulation-broadcast partition behavior in a small, fast-to-compile model.

    Keeps everything the real forward needs: embedders, pos/time/modulation,
    a handful of blocks, norm_out, proj_out. The shared-modulation fanout
    pattern (one modulation tensor -> every block) is preserved, which is
    what reproduces the RouterX86 `q::ForceFormat_Crouton` failure.
    """

    def __init__(self, src_transformer, num_double: int = 1, num_single: int = 2):
        super().__init__()
        self.transformer = src_transformer
        self._original_double = list(src_transformer.transformer_blocks)
        self._original_single = list(src_transformer.single_transformer_blocks)
        src_transformer.transformer_blocks = nn.ModuleList(
            self._original_double[:num_double]
        )
        src_transformer.single_transformer_blocks = nn.ModuleList(
            self._original_single[:num_single]
        )

    def forward(self, hidden_states, encoder_hidden_states, timestep,
                img_ids, txt_ids):
        # Klein: guidance is always None (the 4B distilled variant).
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )[0]


def build_inputs(transformer, height=512, width=512, max_text_len=512):
    """Construct valid inputs for the mini transformer."""
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

    log.info(
        "inputs: hs=%s ehs=%s ts=%s iid=%s tid=%s",
        list(hidden_states.shape), list(encoder_hidden_states.shape),
        list(timestep.shape), list(img_ids.shape), list(txt_ids.shape),
    )
    return (hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_double_blocks", type=int, default=1)
    ap.add_argument("--num_single_blocks", type=int, default=2)
    ap.add_argument("--soc_model", default="SM8850")
    ap.add_argument("--use_fp16", action="store_true",
                    help="Skip PTQ, use fp16 QNN HTP mode")
    ap.add_argument("--quant_dtype", default="8a8w")
    ap.add_argument("--num_calibration_passes", type=int, default=1)
    ap.add_argument("--output_path", default="./mini_transformer.pte")
    ap.add_argument("--transformer_shards", type=int, default=1)
    ap.add_argument("--discard_softmax", action="store_true",
                    help="Keep _safe_softmax / softmax in fp16 on HTP (unquantized)")
    ap.add_argument("--discard_rms_norm", action="store_true",
                    help="Keep rms_norm in fp16 on HTP (unquantized)")
    ap.add_argument("--skip_ln_decomp", action="store_true",
                    help="Do NOT decompose LayerNorm (let it fall back to CPU)")
    ap.add_argument("--online_prepare", action="store_true",
                    help="Use on-device graph preparation (sidesteps RouterX86)")
    args = ap.parse_args()

    # Lazy imports so the script can syntax-check without executorch.
    from export_flux2_klein_qnn import (
        load_pipeline, export_component_to_qnn, get_qcom_chipset,
        _patch_apply_rotary_emb_for_qnn, configure_local_tooling,
    )

    configure_local_tooling(allow_reexec=True)
    _patch_apply_rotary_emb_for_qnn()

    if args.skip_ln_decomp:
        import export_flux2_klein_qnn as qnn_mod
        def _noop(*a, **kw):
            log.info("SKIPPED: _decompose_layer_norm (skip_ln_decomp=True)")
        qnn_mod._decompose_layer_norm = _noop

    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    mini = MiniFluxTransformer(
        pipe.transformer,
        num_double=args.num_double_blocks,
        num_single=args.num_single_blocks,
    ).eval().cpu()
    sample_inputs = build_inputs(pipe.transformer)

    total = args.num_double_blocks + args.num_single_blocks
    log.info(
        "exporting mini transformer: %d double + %d single blocks, quant=%s, soc=%s",
        args.num_double_blocks, args.num_single_blocks,
        "fp16" if args.use_fp16 else args.quant_dtype,
        args.soc_model,
    )
    discard_ops = []
    if args.discard_softmax:
        discard_ops.append(torch.ops.aten._safe_softmax.default)
        discard_ops.append(torch.ops.aten.softmax.int)
    if args.discard_rms_norm:
        discard_ops.append(torch.ops.aten.rms_norm.default)

    export_component_to_qnn(
        mini,
        sample_inputs,
        args.output_path,
        soc_chipset=get_qcom_chipset(args.soc_model),
        num_calibration_passes=args.num_calibration_passes,
        online_prepare=args.online_prepare,
        quant_dtype=None if args.use_fp16 else args.quant_dtype,
        use_fp16=args.use_fp16,
        calibration_data=None,
        num_shards=args.transformer_shards,
        num_double_layers=args.num_double_blocks,
        total_layers=total,
        discard_quant_ops=discard_ops if discard_ops else None,
    )
    log.info("SUCCESS: %s (%.1f MB)",
             args.output_path,
             os.path.getsize(args.output_path) / 1e6)


if __name__ == "__main__":
    main()
