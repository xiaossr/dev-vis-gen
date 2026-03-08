#!/usr/bin/env python3
"""
Export FLUX.2-klein-4B to ExecuTorch XNNPACK (.pte) for fully on-device CPU inference.

Pipeline components (from model_index.json):
  - Text encoder : Qwen3ForCausalLM         → text_encoder.pte
  - Transformer  : Flux2Transformer2DModel   → transformer.pte
  - VAE decoder  : AutoencoderKLFlux2        → vae_decoder.pte
  - VAE encoder  : AutoencoderKLFlux2        → vae_encoder.pte  (for img2img)
  - Scheduler    : FlowMatchEulerDiscreteScheduler (pure Python at inference)

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
    python export_flux2_klein_xnnpack.py --component text_encoder
    python export_flux2_klein_xnnpack.py --component vae_encoder

    # Export with image-to-image support (1 reference image at same res):
    python export_flux2_klein_xnnpack.py --num_img2img_images 1

Notes
-----
- XNNPACK operates in float32.  bfloat16 weights are cast automatically.
- The transformer alone is ~4 B params (~16 GB fp32, ~4 GB int8).
- torch.export may not support every op in the model; unsupported ops fall
  back to ExecuTorch's portable CPU kernels (slower but functional).

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


def apply_8da4w_quantization(model: nn.Module, group_size: int = 128):
    """Apply 8da4w quantization: int8 dynamic activations + int4 weights.

    Uses the same TorchAO quantization path as ExecuTorch's LLM export
    pipeline (``extension.llm.export``).  All ``nn.Linear`` layers whose
    input dimension is divisible by *group_size* are quantized in-place.

    After quantization, ``unwrap_tensor_subclass`` is called so that the
    resulting model can be traced by ``torch.export``.
    """
    from torchao.quantization import (
        Int8DynamicActivationIntxWeightConfig,
        quantize_,
    )
    from torchao.quantization.granularity import PerGroup
    from torchao.utils import unwrap_tensor_subclass

    def filter_fn(m, fqn):
        return isinstance(m, nn.Linear) and m.weight.shape[1] % group_size == 0

    logger.info(
        "Applying 8da4w quantization (int4 weights, group_size=%d) …",
        group_size,
    )
    quantize_(
        model,
        Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(group_size),
        ),
        filter_fn=filter_fn,
    )
    unwrap_tensor_subclass(model)
    logger.info("8da4w quantization complete.")


def apply_w8a8_quantization(model: nn.Module):
    """Apply W8A8 quantization: int8 dynamic activations + int8 per-channel weights.

    Quantizes all ``nn.Linear`` layers in-place using TorchAO's
    ``Int8DynamicActivationIntxWeightConfig`` with int8 weights and
    per-channel granularity.  This is the standard W8A8 dynamic
    quantization used for transformer and VAE components.

    Conv layers are left in fp32 — XNNPACK will still accelerate them
    with its optimised fp32 conv kernels.
    """
    from torchao.quantization import (
        Int8DynamicActivationIntxWeightConfig,
        quantize_,
    )
    from torchao.quantization.granularity import PerAxis
    from torchao.utils import unwrap_tensor_subclass

    def filter_fn(m, fqn):
        return isinstance(m, nn.Linear)

    logger.info("Applying W8A8 quantization (int8 weights per-channel, "
                "int8 dynamic activations) …")
    quantize_(
        model,
        Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int8,
            weight_granularity=PerAxis(0),
        ),
        filter_fn=filter_fn,
    )
    unwrap_tensor_subclass(model)
    logger.info("W8A8 quantization complete.")


class _QuantizedEmbedding(nn.Module):
    """int8 per-channel quantized embedding using ExecuTorch's native op.

    Uses ``quantized_decomposed::embedding_byte.dtype`` which has proper
    out-variant support in ExecuTorch (unlike TorchAO's
    ``dequantize_affine``).
    """

    def __init__(self, weight_int8: torch.Tensor, scales: torch.Tensor,
                 output_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.output_dtype = output_dtype
        self.register_buffer("weight", weight_int8)  # (V, D) int8
        self.register_buffer("scales", scales)        # (V,) float16

    @torch.no_grad()
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized_decomposed.embedding_byte.dtype(
            self.weight, self.scales, None, -128, 127, indices,
            dtype=self.output_dtype,
        )


def _quantize_embedding_per_channel(weight: torch.Tensor):
    """Symmetric per-channel int8 quantization for an embedding table.

    Returns (weight_int8, scales) where
    ``weight ≈ weight_int8.float() * scales.unsqueeze(1)``.
    """
    weight_float = weight.detach().float()
    scales = weight_float.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    weight_int8 = torch.clamp(
        torch.round(weight_float / scales.unsqueeze(1)), -128, 127
    ).to(torch.int8)
    return weight_int8, scales.to(torch.float16)


def _replace_embeddings(module: nn.Module, output_dtype: torch.dtype = torch.float32):
    """Recursively replace every ``nn.Embedding`` with ``_QuantizedEmbedding``."""
    for name, child in module.named_children():
        if isinstance(child, nn.Embedding):
            w_int8, scales = _quantize_embedding_per_channel(child.weight)
            setattr(module, name, _QuantizedEmbedding(w_int8, scales, output_dtype))
        else:
            _replace_embeddings(child, output_dtype)


def apply_embedding_quantization(model: nn.Module):
    """Quantize all ``nn.Embedding`` layers to int8 per-channel.

    Mirrors the ``embedding_quantize: 8,0`` option in ExecuTorch's LLM
    export YAML config.  For Qwen3-4B the embedding table is
    151936 × 2560 = ~1.5 GB in fp32; quantizing to int8 shrinks it to ~0.4 GB.

    Uses ``quantized_decomposed::embedding_byte.dtype`` (with registered
    out-variant) instead of TorchAO's ``dequantize_affine`` to avoid
    the "Missing out variants" error at serialisation time.
    """
    # Ensure the quantized_decomposed custom ops are registered.
    import executorch.exir.passes._quant_patterns_and_replacements  # noqa: F401

    logger.info("Applying embedding quantization (int8, per-channel) …")
    _replace_embeddings(model, output_dtype=torch.float32)
    logger.info("Embedding quantization complete.")


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
    use_dynamic_quant_partitioner: bool = False,
):
    """torch.export → XNNPACK partitioning → ExecuTorch serialisation.

    If *quantize* is True, PT2E int8-symmetric quantization is applied
    before lowering (requires ``torchao``).

    If *use_dynamic_quant_partitioner* is True, the XNNPACK partitioner
    is configured to handle dynamically-quantized linear ops (needed for
    models that were source-transformed with 8da4w quantization).  Two
    partitioners run in sequence: first one for DQ-linear nodes, then a
    greedy one for everything else.
    """
    from torch.export import export
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackDynamicallyQuantizedPartitioner,
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
                    cal_inputs = []
                    for j, inp in enumerate(sample_inputs):
                        if inp.is_floating_point():
                            if inp.ndim == 1 and inp.shape[0] == 1:
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
    if use_dynamic_quant_partitioner:
        partitioners = [
            XnnpackDynamicallyQuantizedPartitioner(),
            XnnpackPartitioner(),
        ]
    else:
        partitioners = [XnnpackPartitioner()]

    edge_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=partitioners,
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
    p.add_argument("--max_text_len", type=int, default=512,
                    help="Max text-token sequence length for the transformer")
    p.add_argument("--quantize", action="store_true",
                    help="Apply int8 symmetric quantization (XNNPACK)")
    p.add_argument("--text_encoder_8da4w", action="store_true",
                    help="Apply 8da4w quantization (int8 dynamic activation + "
                         "int4 weight) to the Qwen3 text encoder instead of "
                         "the default PT2E int8.  This matches the quantization "
                         "used by ExecuTorch's LLM export for Qwen3 and "
                         "reduces text-encoder size by ~4× vs fp32.")
    p.add_argument("--group_size", type=int, default=128,
                    help="Group size for 8da4w weight quantization (default: 128)")
    p.add_argument("--w8a8", action="store_true",
                    help="Apply W8A8 quantization (int8 dynamic activation + "
                         "int8 per-channel weight) to the transformer and VAE "
                         "via TorchAO source transforms.  Linear layers are "
                         "quantized; conv layers remain fp32.")
    p.add_argument("--embedding_quantize", type=int, default=0, metavar="BITS",
                    help="Quantize nn.Embedding layers to BITS-bit (e.g. 8 for "
                         "int8).  Matches ExecuTorch's 'embedding_quantize: 8,0'.  "
                         "Set to 0 (default) to keep embeddings in fp32.")
    p.add_argument("--component",
                    choices=["all", "transformer", "vae", "vae_encoder",
                             "text_encoder"],
                    default="all",
                    help="Which component(s) to export")
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
    # Must match the pipeline's encode_prompt default for Klein (Qwen3): (9, 18, 27).
    te_cfg = pipe.text_encoder.config
    num_te_layers = getattr(te_cfg, "num_hidden_layers", 28)
    hidden_states_layers = [9, 18, 27]
    logger.info("Text encoder: %d layers, extracting from %s",
                num_te_layers, hidden_states_layers)

    # ---- save export metadata ------------------------------------------
    t_cfg = pipe.transformer.config
    vae_cfg = pipe.vae.config
    vae_sf = _get_vae_scale_factor(pipe)
    patch_h, patch_w = _compute_latent_dims(args.height, args.width, vae_sf)
    te_quant_mode = "8da4w" if args.text_encoder_8da4w else ("int8" if args.quantize else "none")
    tf_quant_mode = "w8a8" if args.w8a8 else ("int8_pt2e" if args.quantize else "none")
    vae_quant_mode = "w8a8" if args.w8a8 else ("int8_pt2e" if args.quantize else "none")
    any_quantized = args.quantize or args.text_encoder_8da4w or args.w8a8
    meta = {
        "model_id": args.model_id,
        "height": args.height,
        "width": args.width,
        "max_text_len": args.max_text_len,
        "quantized": any_quantized,
        "text_encoder_quantization": te_quant_mode,
        "text_encoder_group_size": args.group_size if args.text_encoder_8da4w else None,
        "text_encoder_embedding_quantize": args.embedding_quantize if args.embedding_quantize > 0 else None,
        "transformer_quantization": tf_quant_mode,
        "vae_quantization": vae_quant_mode,
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

    # ---- export text encoder ---------------------------------------------
    if args.component in ("all", "text_encoder"):
        logger.info("=" * 60)
        logger.info("EXPORTING TEXT ENCODER (Qwen3)")
        logger.info("=" * 60)

        te_use_8da4w = args.text_encoder_8da4w
        if te_use_8da4w:
            logger.info("8da4w requested — quantizing Qwen3 text encoder "
                        "(group_size=%d) before wrapping …", args.group_size)
            pipe.text_encoder = pipe.text_encoder.float()
            apply_8da4w_quantization(pipe.text_encoder, group_size=args.group_size)

        if args.embedding_quantize > 0:
            logger.info("Quantizing text encoder embeddings to int%d …",
                        args.embedding_quantize)
            apply_embedding_quantization(pipe.text_encoder)

        wrapper = Qwen3TextEncoderWrapper(
            pipe.text_encoder, hidden_states_layers=hidden_states_layers,
        ).eval()
        inputs = build_text_encoder_inputs(args.max_text_len)

        logger.info("Sanity-checking forward pass …")
        with torch.no_grad():
            test_out = wrapper(*inputs)
        logger.info("  output shape: %s  ✓", test_out.shape)
        del test_out
        _free_memory()

        export_component_to_xnnpack(
            wrapper, inputs, str(out / "text_encoder.pte"),
            quantize=args.quantize if not te_use_8da4w else False,
            use_dynamic_quant_partitioner=te_use_8da4w,
        )
        del wrapper, inputs
        _free_memory()

    # Free text encoder before heavy exports to reduce peak RAM
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        del pipe.text_encoder
        pipe.text_encoder = None
        _free_memory()
        logger.info("Freed text encoder to reduce memory for export.")

    # ---- export transformer --------------------------------------------
    if args.component in ("all", "transformer"):
        tf_use_w8a8 = args.w8a8
        if tf_use_w8a8:
            logger.info("W8A8 requested — quantizing transformer before wrapping …")
            pipe.transformer = pipe.transformer.float()
            apply_w8a8_quantization(pipe.transformer)

        wrapper = Flux2TransformerWrapper(pipe.transformer).eval()

        # Always export the text-to-image transformer (no image tokens)
        logger.info("=" * 60)
        logger.info("EXPORTING TRANSFORMER (text-to-image)")
        logger.info("=" * 60)

        t2i_inputs = build_transformer_inputs(
            pipe, args.height, args.width, args.max_text_len, dtype,
            num_img2img_images=0,
        )
        logger.info("Sanity-checking forward pass …")
        with torch.no_grad():
            test_out = wrapper(*t2i_inputs)
        logger.info("  output shape: %s  ✓", test_out.shape)
        del test_out
        _free_memory()

        export_component_to_xnnpack(
            wrapper, t2i_inputs, str(out / "transformer.pte"),
            quantize=args.quantize if not tf_use_w8a8 else False,
            use_dynamic_quant_partitioner=tf_use_w8a8,
        )
        del t2i_inputs
        _free_memory()

        # If img2img is requested, also export a second transformer
        # with the larger sequence length (noise + image tokens)
        if args.num_img2img_images > 0:
            logger.info("=" * 60)
            logger.info("EXPORTING TRANSFORMER (image-to-image, %d ref image(s))",
                        args.num_img2img_images)
            logger.info("=" * 60)

            img2img_inputs = build_transformer_inputs(
                pipe, args.height, args.width, args.max_text_len, dtype,
                num_img2img_images=args.num_img2img_images,
            )
            logger.info("Sanity-checking forward pass …")
            with torch.no_grad():
                test_out = wrapper(*img2img_inputs)
            logger.info("  output shape: %s  ✓", test_out.shape)
            del test_out
            _free_memory()

            export_component_to_xnnpack(
                wrapper, img2img_inputs,
                str(out / "transformer_img2img.pte"),
                quantize=args.quantize if not tf_use_w8a8 else False,
                use_dynamic_quant_partitioner=tf_use_w8a8,
            )
            del img2img_inputs
            _free_memory()

        del wrapper
        _free_memory()

    # ---- quantize VAE once (shared by decoder + encoder) -----------------
    vae_quantized_w8a8 = False
    if args.w8a8 and args.component in ("all", "vae", "vae_encoder"):
        logger.info("W8A8 requested — quantizing VAE linear layers …")
        pipe.vae = pipe.vae.float()
        apply_w8a8_quantization(pipe.vae)
        vae_quantized_w8a8 = True

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
            quantize=args.quantize if not vae_quantized_w8a8 else False,
            use_dynamic_quant_partitioner=vae_quantized_w8a8,
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
            quantize=args.quantize if not vae_quantized_w8a8 else False,
            use_dynamic_quant_partitioner=vae_quantized_w8a8,
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
    tok_dir = out / "tokenizer"
    if tok_dir.is_dir():
        print(f"  {'tokenizer/':30s}  (Qwen2TokenizerFast)")
    print("=" * 60)


if __name__ == "__main__":
    main()
