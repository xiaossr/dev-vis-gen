#!/usr/bin/env python3
"""
Export FLUX.2-klein-4B to ExecuTorch XNNPACK (.pte) for fully on-device CPU inference.

Pipeline components (from model_index.json):
  - Text encoder : Qwen3ForCausalLM         → embeddings/*.pt  (pre-computed)
  - Transformer  : Flux2Transformer2DModel   → transformer.pte
  - VAE decoder  : AutoencoderKLFlux2        → vae_decoder.pte
  - VAE encoder  : AutoencoderKLFlux2        → vae_encoder.pte  (for img2img)
  - Scheduler    : FlowMatchEulerDiscreteScheduler (pure Python at inference)

The Qwen3 text encoder is too large to torch.export on most machines
(OOM during tracing/serialisation).  Instead, we run it in PyTorch at
export time and save the resulting embeddings as .pt files.  The
inference script loads these directly — no text_encoder.pte needed.

Guidance
--------
The Klein pipeline ALWAYS passes ``guidance=None`` to the transformer;
the guidance tensor parameter exists in the architecture but is unused.
For non-distilled Klein variants (e.g. FLUX.2-klein-base-9B), classifier-
free guidance (CFG) is implemented by running the transformer **twice**
(once with the positive prompt, once with an empty/negative prompt) and
interpolating the predictions.  The 4B distilled model skips CFG entirely
(is_distilled=True, guidance_scale ignored).

Image-to-image
--------------
When a reference image is provided the pipeline:
  1. Encodes it with the VAE encoder → patchifies → BN-normalises → packs.
  2. Creates image positional IDs with time-offsets (T=10,20,…).
  3. Concatenates image tokens with noise tokens before the transformer.
  4. After the transformer, only noise-token predictions are kept.
For this to work with ExecuTorch's static-shape requirement, the
transformer must be exported with a combined sequence length that covers
noise + image tokens.  Use ``--num_img2img_images`` to account for this.

Requirements
------------
    pip install -r requirements_export.txt

Usage
-----
    # Export all components (fp32, 512×512 target resolution):
    python export_flux2_klein_xnnpack.py --output_dir ./exported_models

    # With int8 quantization for smaller .pte and faster CPU inference:
    python export_flux2_klein_xnnpack.py --output_dir ./exported_models --quantize

    # Export only one component:
    python export_flux2_klein_xnnpack.py --component transformer
    python export_flux2_klein_xnnpack.py --component vae
    python export_flux2_klein_xnnpack.py --component embeddings
    python export_flux2_klein_xnnpack.py --component vae_encoder

    # Pre-compute embeddings for specific prompts:
    python export_flux2_klein_xnnpack.py --component embeddings \
        --prompts "a cat on a windowsill" "a dog in a field"

    # Export with image-to-image support (1 reference image at same res):
    python export_flux2_klein_xnnpack.py --num_img2img_images 1

Notes
-----
- XNNPACK operates in float32.  bfloat16 weights are cast automatically.
- The transformer alone is ~4 B params (~16 GB fp32, ~4 GB int8).
- torch.export may not support every op in the model; unsupported ops fall
  back to ExecuTorch's portable CPU kernels (slower but functional).
- Text embeddings are pre-computed in PyTorch and saved as .pt files.
  The Qwen3 text encoder is too large to torch.export (OOM during
  tracing).  To use a new prompt, re-run with ``--component embeddings``.

Platform compatibility
----------------------
- **macOS (Apple Silicon / x86)**: ExecuTorch + XNNPACK works natively.
  XNNPACK has ARM NEON kernels for M-series chips.  int8 recommended;
  the full pipeline in fp32 needs ~20+ GB RAM.
- **Snapdragon 8 Gen 5 (Samsung)**: ExecuTorch is designed for this.
  XNNPACK provides ARM NEON / i8mm / dot-product kernels.  However RAM
  is the bottleneck — flagship phones have 12-16 GB with ~4-6 GB used by
  the OS.  int8 quantization is mandatory; even then the full pipeline
  (text encoder + transformer + VAE) may need ~8-10 GB.  Consider also
  the QNN delegate for Hexagon NPU acceleration, or exporting only the
  transformer to .pte and running text encoder + VAE in PyTorch.
"""

import argparse
import gc
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("flux2_export")


# ============================================================================
# 1.  Export-friendly wrapper modules
# ============================================================================

class Qwen3TextEncoderWrapper(nn.Module):
    """Wraps Qwen3ForCausalLM for export as a text encoder.

    The FLUX.2-klein pipeline extracts hidden states from intermediate
    layers (default: 9, 18, 27), stacks them, and reshapes to produce
    prompt embeddings of shape (B, seq_len, num_layers * hidden_dim).

    KV-cache is disabled.  Input is fixed-length padded token IDs
    with an attention mask.
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        hidden_states_layers: tuple = (9, 18, 27),
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.hidden_states_layers = list(hidden_states_layers)

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, seq_len)  int64
        attention_mask: torch.Tensor,   # (B, seq_len)  int64
    ) -> torch.Tensor:
        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        # Stack selected hidden-state layers: (B, num_layers, seq_len, hidden_dim)
        out = torch.stack(
            [output.hidden_states[k] for k in self.hidden_states_layers], dim=1
        )
        # Reshape to (B, seq_len, num_layers * hidden_dim) = e.g. (B, seq_len, 15360)
        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_channels * hidden_dim
        )
        return prompt_embeds


class Flux2TransformerWrapper(nn.Module):
    """Thin wrapper around Flux2Transformer2DModel.

    * Accepts only positional tensor arguments (no **kwargs / dicts).
    * Returns a plain tensor instead of a dataclass.
    * ``guidance=None`` is ALWAYS correct for Klein (both distilled and
      non-distilled).  The guidance_embeds architecture param exists but
      Klein never uses it.  Classifier-free guidance for non-distilled
      Klein is done via two separate forward passes, not a guidance tensor.
    * Positional IDs are batched (B, seq_len, 4) using 4D (T,H,W,L) coords.
    * For image-to-image, hidden_states contains [noise_tokens, image_tokens]
      concatenated along the sequence dimension, and img_ids contains the
      matching concatenated positional IDs.
    """

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,           # (B, img_seq_len, in_channels)
        encoder_hidden_states: torch.Tensor,    # (B, txt_seq_len, joint_attn_dim)
        timestep: torch.Tensor,                 # (B,)
        img_ids: torch.Tensor,                  # (B, img_seq_len, 4)
        txt_ids: torch.Tensor,                  # (B, txt_seq_len, 4)
    ) -> torch.Tensor:
        out = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        return out[0]


class VAEEncoderWrapper(nn.Module):
    """Wraps AutoencoderKLFlux2 for encode-only export (image-to-image).

    Returns the argmax latent (deterministic, no sampling), which is
    what the pipeline uses for reference-image conditioning.
    Output shape: (B, latent_channels, H // vae_sf, W // vae_sf).
    """

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        dist = self.vae.encode(pixel_values, return_dict=False)[0]
        return dist.mode()


class VAEDecoderWrapper(nn.Module):
    """Wraps AutoencoderKLFlux2 for decode-only export.

    The caller is expected to handle batch-norm un-normalisation and
    un-patchification before calling this wrapper.  This wrapper simply
    invokes the VAE decoder on spatial latents (B, C, H, W).
    """

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents, return_dict=False)[0]


# ============================================================================
# 2.  Helpers
# ============================================================================

def _free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_pipeline(model_id: str, dtype: torch.dtype = torch.float32):
    """Load Flux2KleinPipeline from HuggingFace Hub (or a local path)."""
    from diffusers import Flux2KleinPipeline

    logger.info("Loading pipeline '%s' (dtype=%s) …", model_id, dtype)
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)
    logger.info("Pipeline loaded.")
    return pipe


def _get_vae_scale_factor(pipe) -> int:
    """Return the spatial down-scale factor of the VAE."""
    if hasattr(pipe, "vae_scale_factor"):
        return pipe.vae_scale_factor
    vae_cfg = pipe.vae.config
    if hasattr(vae_cfg, "block_out_channels"):
        return 2 ** (len(vae_cfg.block_out_channels) - 1)
    return 8  # safe default for FLUX-family models


def _prepare_latent_ids_klein(latent_h: int, latent_w: int,
                              batch: int = 1) -> torch.Tensor:
    """Build 4D positional IDs (T, H, W, L) for image latent tokens.

    Matches Flux2KleinPipeline._prepare_latent_ids: after patchification
    the spatial dims are halved, so pass the *post-patchify* H and W.
    Returns shape (B, H*W, 4) as int64.
    """
    t = torch.arange(1)           # [0]
    h = torch.arange(latent_h)
    w = torch.arange(latent_w)
    l = torch.arange(1)           # [0]
    latent_ids = torch.cartesian_prod(t, h, w, l)          # (H*W, 4)
    return latent_ids.unsqueeze(0).expand(batch, -1, -1)   # (B, H*W, 4)


def _prepare_text_ids_klein(seq_len: int, batch: int = 1) -> torch.Tensor:
    """Build 4D positional IDs (T, H, W, L) for text tokens.

    Matches Flux2KleinPipeline._prepare_text_ids.
    Returns shape (B, seq_len, 4) as int64.
    """
    out_ids = []
    for _ in range(batch):
        t = torch.arange(1)       # [0]
        h = torch.arange(1)       # [0]
        w = torch.arange(1)       # [0]
        seq = torch.arange(seq_len)
        coords = torch.cartesian_prod(t, h, w, seq)       # (seq_len, 4)
        out_ids.append(coords)
    return torch.stack(out_ids)   # (B, seq_len, 4)


def build_text_encoder_inputs(max_text_len: int, batch: int = 1):
    """Create dummy token-level inputs for the Qwen3 text encoder."""
    input_ids      = torch.ones(batch, max_text_len, dtype=torch.long)
    attention_mask  = torch.ones(batch, max_text_len, dtype=torch.long)
    logger.info(
        "Text encoder sample shapes:\n"
        "  input_ids      : %s\n"
        "  attention_mask  : %s",
        input_ids.shape, attention_mask.shape,
    )
    return (input_ids, attention_mask)


def _compute_latent_dims(height: int, width: int, vae_sf: int):
    """Compute post-patchify latent dims matching the pipeline exactly.

    Pipeline does: ``height = 2 * (int(height) // (vae_scale_factor * 2))``
    to ensure H,W are divisible by 2 (required for 2×2 patchification).
    Returns (patch_h, patch_w) — the spatial dims after patchification.
    """
    latent_h = 2 * (height // (vae_sf * 2))
    latent_w = 2 * (width  // (vae_sf * 2))
    return latent_h // 2, latent_w // 2


def build_transformer_inputs(pipe, height: int, width: int, max_text_len: int,
                             dtype: torch.dtype = torch.float32,
                             num_img2img_images: int = 0):
    """Create dummy tensors that match the transformer's expected shapes.

    FLUX.2-klein uses patchification (2×2) so the actual latent spatial
    dims fed to the transformer are halved vs. the VAE output.  Latents
    are packed to (B, H*W, C) with C = in_channels (128 post-patch).
    Positional IDs are batched (B, seq_len, 4) with 4D (T, H, W, L).
    ``guidance=None`` always for Klein (not passed to wrapper).

    When *num_img2img_images* > 0 the hidden_states sequence length is
    extended to include reference-image tokens (same spatial resolution
    per image, concatenated after the noise tokens).
    """
    t_cfg = pipe.transformer.config
    in_channels = t_cfg.in_channels                       # 128 (post-patchify)
    joint_dim   = t_cfg.joint_attention_dim               # 15360
    vae_sf = _get_vae_scale_factor(pipe)

    patch_h, patch_w = _compute_latent_dims(height, width, vae_sf)
    num_noise_tokens = patch_h * patch_w
    num_img_tokens   = num_noise_tokens * num_img2img_images
    total_img_tokens = num_noise_tokens + num_img_tokens

    batch = 1

    hidden_states            = torch.randn(batch, total_img_tokens, in_channels, dtype=dtype)
    encoder_hidden_states    = torch.randn(batch, max_text_len, joint_dim, dtype=dtype)
    timestep                 = torch.full((batch,), 0.5, dtype=dtype)

    # Positional ids for noise tokens
    noise_ids = _prepare_latent_ids_klein(patch_h, patch_w, batch)

    if num_img2img_images > 0:
        # Reference image IDs have time-offsets T=10, 20, … per image
        ref_id_list = []
        for img_idx in range(num_img2img_images):
            t_offset = 10 + 10 * img_idx
            single = _prepare_latent_ids_klein(patch_h, patch_w, batch)
            single[:, :, 0] = t_offset  # set T coordinate
            ref_id_list.append(single)
        ref_ids = torch.cat(ref_id_list, dim=1)
        img_ids = torch.cat([noise_ids, ref_ids], dim=1).to(dtype)
    else:
        img_ids = noise_ids.to(dtype)

    txt_ids = _prepare_text_ids_klein(max_text_len, batch).to(dtype)

    logger.info(
        "Transformer sample shapes:\n"
        "  hidden_states         : %s  (noise=%d + img2img=%d)\n"
        "  encoder_hidden_states : %s\n"
        "  timestep              : %s\n"
        "  img_ids               : %s\n"
        "  txt_ids               : %s",
        hidden_states.shape, num_noise_tokens, num_img_tokens,
        encoder_hidden_states.shape,
        timestep.shape, img_ids.shape, txt_ids.shape,
    )
    return (hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids)


def build_vae_inputs(pipe, height: int, width: int,
                     dtype: torch.dtype = torch.float32):
    """Create a dummy latent tensor for the VAE decoder.

    The VAE receives un-patchified latents of shape (B, C, H, W) where
    C = latent_channels and H, W match the full VAE latent resolution
    (before patchification).
    """
    vae_cfg = pipe.vae.config
    latent_ch = getattr(vae_cfg, "latent_channels", 32)
    vae_sf = _get_vae_scale_factor(pipe)
    patch_h, patch_w = _compute_latent_dims(height, width, vae_sf)
    # Un-patchified dims are 2× the patchified dims
    latent_h, latent_w = patch_h * 2, patch_w * 2

    latents = torch.randn(1, latent_ch, latent_h, latent_w, dtype=dtype)
    logger.info("VAE decoder sample shape: %s", latents.shape)
    return (latents,)


def build_vae_encoder_inputs(height: int, width: int,
                             dtype: torch.dtype = torch.float32):
    """Create a dummy pixel-space image for the VAE encoder (img2img)."""
    pixel_values = torch.randn(1, 3, height, width, dtype=dtype)
    logger.info("VAE encoder sample shape: %s", pixel_values.shape)
    return (pixel_values,)


# ============================================================================
# 3.  Core export routine
# ============================================================================

def export_component_to_xnnpack(
    model: nn.Module,
    sample_inputs: tuple,
    output_path: str,
    quantize: bool = False,
):
    """torch.export → XNNPACK partitioning → ExecuTorch serialisation.

    If *quantize* is True, PT2E int8-symmetric quantization is applied
    before lowering (requires ``torchao``).
    """
    from torch.export import export
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    model.eval()

    # ---- optional int8 quantization (two-stage export) -----------------
    if quantize:
        logger.info("Applying PT2E int8 symmetric quantization …")
        try:
            from torchao.quantization.pt2e.quantize_pt2e import (
                convert_pt2e,
                prepare_pt2e,
            )
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer,
                get_symmetric_quantization_config,
            )

            pre_ep = export(model, sample_inputs)
            model = pre_ep.module()

            quantizer = XNNPACKQuantizer()
            qconfig = get_symmetric_quantization_config(is_per_channel=True)
            quantizer.set_global(qconfig)

            model = prepare_pt2e(model, quantizer)

            logger.info("Running calibration forward passes …")
            num_calibration_passes = 10
            with torch.no_grad():
                for cal_i in range(num_calibration_passes):
                    # Use varied representative inputs: different random seeds
                    # and a spread of timestep values to cover the activation
                    # range the model will see at inference time.
                    cal_inputs = []
                    for j, inp in enumerate(sample_inputs):
                        if inp.is_floating_point():
                            if inp.ndim == 1 and inp.shape[0] == 1:
                                # Timestep — sweep across sigma range
                                cal_inputs.append(
                                    torch.full_like(inp, (cal_i + 1) / (num_calibration_passes + 1))
                                )
                            else:
                                cal_inputs.append(torch.randn_like(inp))
                        else:
                            cal_inputs.append(inp)
                    model(*cal_inputs)
                    logger.info("  calibration %d/%d", cal_i + 1, num_calibration_passes)

            model = convert_pt2e(model)
            logger.info("Quantization complete.")
        except ImportError as exc:
            logger.warning(
                "Quantization deps unavailable (%s); falling back to fp32.", exc
            )
            quantize = False

    # ---- export --------------------------------------------------------
    logger.info("torch.export.export() …")
    exported_program = export(model, sample_inputs)

    # ---- lower to XNNPACK ----------------------------------------------
    logger.info("Lowering to XNNPACK backend …")
    edge_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    )

    # ---- serialise to .pte ---------------------------------------------
    logger.info("Serialising to .pte …")
    et_program = edge_program.to_executorch()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Saved %s  (%.1f MB)", output_path, size_mb)


# ============================================================================
# 4.  Tokenizer helper
# ============================================================================

def copy_tokenizer(pipe, output_dir: str):
    """Save the tokenizer to disk so it can be shipped with the .pte files."""
    tok_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    pipe.tokenizer.save_pretrained(tok_dir)
    logger.info("Tokenizer saved to %s/", tok_dir)


def precompute_embeddings(pipe, prompts: list, max_text_len: int,
                          output_dir: str, is_distilled: bool = True):
    """Run the text encoder in PyTorch and save embeddings as .pt files.

    This avoids the need to export the (very large) Qwen3 text encoder
    to .pte.  Each prompt gets a separate file in ``embeddings/``.
    For non-distilled models, an empty-string prompt is included for CFG.
    """
    emb_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    # Include empty prompt only for non-distilled models (CFG)
    if is_distilled:
        all_prompts = list(dict.fromkeys(prompts))  # deduplicate
    else:
        all_prompts = list(dict.fromkeys([""] + prompts))  # empty for CFG

    for idx, prompt in enumerate(all_prompts):
        logger.info("Encoding prompt %d/%d: %r", idx + 1, len(all_prompts),
                     prompt[:80] + ("…" if len(prompt) > 80 else ""))

        result = pipe.encode_prompt(
            prompt=prompt,
            max_sequence_length=max_text_len,
        )
        # encode_prompt returns (prompt_embeds, pooled_prompt_embeds, text_ids)
        prompt_embeds = result[0].detach().cpu().float()

        # Save with a sanitised filename
        safe_name = f"prompt_{idx}.pt"
        save_path = os.path.join(emb_dir, safe_name)
        torch.save({
            "prompt": prompt,
            "prompt_embeds": prompt_embeds,
        }, save_path)
        logger.info("  → %s  shape=%s", save_path, prompt_embeds.shape)

    # Save an index mapping prompt text → filename
    index = {all_prompts[i]: f"prompt_{i}.pt" for i in range(len(all_prompts))}
    index_path = os.path.join(emb_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    logger.info("Saved embedding index to %s", index_path)


def save_vae_bn_stats(pipe, output_dir: str):
    """Save the VAE's batch-norm running statistics.

    The FLUX.2 pipeline un-normalises latents using
    ``vae.bn.running_mean`` / ``vae.bn.running_var`` *before* passing
    them to the VAE decoder.  Because the VAE .pte doesn't include this
    pre-processing, we save the stats separately so the inference script
    can apply the same transformation.
    """
    vae = pipe.vae
    if not hasattr(vae, "bn"):
        logger.warning("VAE has no .bn attribute — skipping BN stats save.")
        return

    stats = {
        "running_mean": vae.bn.running_mean.detach().cpu().float(),
        "running_var":  vae.bn.running_var.detach().cpu().float(),
    }
    save_path = os.path.join(output_dir, "vae_bn_stats.pt")
    torch.save(stats, save_path)
    logger.info("VAE batch-norm stats saved to %s", save_path)


# ============================================================================
# 5.  Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Export FLUX.2-klein-4B to ExecuTorch XNNPACK (.pte)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-4B",
                    help="HuggingFace model ID or local path")
    p.add_argument("--output_dir", default="./exported_flux2_klein",
                    help="Directory for exported artefacts")
    p.add_argument("--height", type=int, default=512,
                    help="Target image height (determines transformer input shapes)")
    p.add_argument("--width", type=int, default=512,
                    help="Target image width")
    p.add_argument("--max_text_len", type=int, default=128,
                    help="Max text-token sequence length for the transformer")
    p.add_argument("--quantize", action="store_true",
                    help="Apply int8 symmetric quantization (XNNPACK)")
    p.add_argument("--component",
                    choices=["all", "transformer", "vae", "vae_encoder",
                             "embeddings"],
                    default="all",
                    help="Which component(s) to export")
    p.add_argument("--prompts", nargs="+",
                    default=["a cat sitting on a windowsill at sunset"],
                    help="Prompts to pre-compute text embeddings for.  "
                         "An empty-string (negative) prompt is always included.")
    p.add_argument("--num_img2img_images", type=int, default=0,
                    help="Number of reference images for img2img.  When > 0 "
                         "the transformer is exported with a combined sequence "
                         "length (noise + image tokens) and the VAE encoder "
                         "is also exported.")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32  # XNNPACK requires fp32

    # ---- load pipeline -------------------------------------------------
    pipe = load_pipeline(args.model_id, dtype=dtype)

    is_distilled = getattr(pipe.config, "is_distilled", True)

    # ---- save tokenizer ------------------------------------------------
    copy_tokenizer(pipe, str(out))

    # ---- save VAE batch-norm stats -------------------------------------
    save_vae_bn_stats(pipe, str(out))

    # ---- determine text encoder hidden_states_layers -------------------
    # The pipeline extracts hidden states from specific layers. For Klein,
    # these are typically evenly spaced through the model (e.g. 9, 18, 27
    # for a 28-layer model, or 12, 24, 36 for a 36-layer model).
    te_cfg = pipe.text_encoder.config
    num_te_layers = getattr(te_cfg, "num_hidden_layers", 28)
    # Default: extract from layers at 1/3, 2/3, and final
    hidden_states_layers = [
        num_te_layers // 3,
        2 * num_te_layers // 3,
        num_te_layers - 1,
    ]
    logger.info("Text encoder: %d layers, extracting from %s",
                num_te_layers, hidden_states_layers)

    # ---- save export metadata ------------------------------------------
    t_cfg = pipe.transformer.config
    vae_cfg = pipe.vae.config
    vae_sf = _get_vae_scale_factor(pipe)
    patch_h, patch_w = _compute_latent_dims(args.height, args.width, vae_sf)
    meta = {
        "model_id": args.model_id,
        "height": args.height,
        "width": args.width,
        "max_text_len": args.max_text_len,
        "quantized": args.quantize,
        "is_distilled": is_distilled,
        "num_inference_steps": 4 if is_distilled else 50,
        "guidance_scale": 1.0 if is_distilled else 4.0,
        "vae_scale_factor": vae_sf,
        "num_img2img_images": args.num_img2img_images,
        "patch_dims": [patch_h, patch_w],
        "text_encoder": {
            "hidden_states_layers": hidden_states_layers,
            "max_sequence_length": args.max_text_len,
        },
        "transformer": {
            "in_channels": t_cfg.in_channels,
            "out_channels": t_cfg.out_channels or t_cfg.in_channels,
            "num_layers": t_cfg.num_layers,
            "num_single_layers": t_cfg.num_single_layers,
            "joint_attention_dim": t_cfg.joint_attention_dim,
            "axes_dims_rope": list(t_cfg.axes_dims_rope),
            "guidance_embeds": getattr(t_cfg, "guidance_embeds", True),
        },
        "vae": {
            "latent_channels": getattr(vae_cfg, "latent_channels", None),
            "scaling_factor": getattr(vae_cfg, "scaling_factor", None),
            "shift_factor": getattr(vae_cfg, "shift_factor", None),
            "batch_norm_eps": getattr(vae_cfg, "batch_norm_eps", 1e-5),
        },
    }
    meta_path = out / "export_config.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %s", meta_path)

    # ---- pre-compute text embeddings ------------------------------------
    if args.component in ("all", "embeddings"):
        logger.info("=" * 60)
        logger.info("PRE-COMPUTING TEXT EMBEDDINGS (Qwen3 in PyTorch)")
        logger.info("=" * 60)
        precompute_embeddings(pipe, args.prompts, args.max_text_len, str(out),
                              is_distilled=is_distilled)

    # Free text encoder before heavy exports to reduce peak RAM
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        del pipe.text_encoder
        pipe.text_encoder = None
        _free_memory()
        logger.info("Freed text encoder to reduce memory for export.")

    # ---- export transformer --------------------------------------------
    if args.component in ("all", "transformer"):
        logger.info("=" * 60)
        logger.info("EXPORTING TRANSFORMER")
        logger.info("=" * 60)

        wrapper = Flux2TransformerWrapper(pipe.transformer).eval()
        inputs  = build_transformer_inputs(
            pipe, args.height, args.width, args.max_text_len, dtype,
            num_img2img_images=args.num_img2img_images,
        )

        logger.info("Sanity-checking forward pass …")
        with torch.no_grad():
            test_out = wrapper(*inputs)
        logger.info("  output shape: %s  ✓", test_out.shape)
        del test_out
        _free_memory()

        export_component_to_xnnpack(
            wrapper, inputs, str(out / "transformer.pte"),
            quantize=args.quantize,
        )
        del wrapper, inputs
        _free_memory()

    # ---- export VAE decoder --------------------------------------------
    if args.component in ("all", "vae"):
        logger.info("=" * 60)
        logger.info("EXPORTING VAE DECODER")
        logger.info("=" * 60)

        wrapper = VAEDecoderWrapper(pipe.vae).eval()
        inputs  = build_vae_inputs(pipe, args.height, args.width, dtype)

        logger.info("Sanity-checking forward pass …")
        with torch.no_grad():
            test_out = wrapper(*inputs)
        logger.info("  output shape: %s  ✓", test_out.shape)
        del test_out
        _free_memory()

        export_component_to_xnnpack(
            wrapper, inputs, str(out / "vae_decoder.pte"),
            quantize=args.quantize,
        )
        del wrapper, inputs
        _free_memory()

    # ---- export VAE encoder (for img2img) --------------------------------
    export_vae_enc = (
        args.component == "vae_encoder"
        or (args.component == "all" and args.num_img2img_images > 0)
    )
    if export_vae_enc:
        logger.info("=" * 60)
        logger.info("EXPORTING VAE ENCODER (for image-to-image)")
        logger.info("=" * 60)

        wrapper = VAEEncoderWrapper(pipe.vae).eval()
        inputs  = build_vae_encoder_inputs(args.height, args.width, dtype)

        logger.info("Sanity-checking forward pass …")
        with torch.no_grad():
            test_out = wrapper(*inputs)
        logger.info("  output shape: %s  ✓", test_out.shape)
        del test_out
        _free_memory()

        export_component_to_xnnpack(
            wrapper, inputs, str(out / "vae_encoder.pte"),
            quantize=args.quantize,
        )
        del wrapper, inputs
        _free_memory()

    # ---- summary -------------------------------------------------------
    del pipe
    _free_memory()

    banner = "\n" + "=" * 60 + "\n  EXPORT COMPLETE\n" + "=" * 60
    print(banner)
    for f in sorted(out.glob("*.pte")):
        print(f"  {f.name:30s}  {f.stat().st_size / 1024**2:>8.1f} MB")
    print(f"  {'export_config.json':30s}  (metadata)")
    bn_path = out / "vae_bn_stats.pt"
    if bn_path.exists():
        print(f"  {'vae_bn_stats.pt':30s}  (VAE batch-norm stats)")
    emb_dir = out / "embeddings"
    if emb_dir.is_dir():
        n_emb = len(list(emb_dir.glob("prompt_*.pt")))
        print(f"  {'embeddings/':30s}  ({n_emb} pre-computed prompt(s))")
    tok_dir = out / "tokenizer"
    if tok_dir.is_dir():
        print(f"  {'tokenizer/':30s}  (Qwen2TokenizerFast)")
    print("=" * 60)


if __name__ == "__main__":
    main()
