#!/usr/bin/env python3
"""
Export FLUX.2-klein-4B components to QAIRT W8A8 quantized DLC for Qualcomm HTP/DSP.

Pipeline per component:
  1. Load PyTorch model, wrap for export
  2. Export to ONNX (torch.onnx.export)
  3. Generate calibration data (random inputs saved as raw files)
  4. qairt.convert() with CalibrationConfig → quantized DLC (W8A8)

Components:
  - Text encoder : Qwen3ForCausalLM         → text_encoder.dlc
  - Transformer  : Flux2Transformer2DModel   → transformer.dlc
  - VAE decoder  : AutoencoderKLFlux2        → vae_decoder.dlc

Usage:
  python export_flux2_klein_qairt.py --component vae --num_calibration_samples 20
  python export_flux2_klein_qairt.py --component transformer --num_calibration_samples 10
  python export_flux2_klein_qairt.py --component text_encoder --num_calibration_samples 10
  python export_flux2_klein_qairt.py --component all
"""

import argparse
import gc
import json
import logging
import os
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("flux2_qairt_export")
DISABLE_ROTARY_FOR_QAIRT = True


# ============================================================================
# Wrapper modules (positional args, plain tensor returns)
# ============================================================================

class Qwen3TextEncoderWrapper(nn.Module):
    """Wraps Qwen3ForCausalLM to extract multi-layer hidden states."""

    def __init__(self, text_encoder, hidden_states_layers=(9, 18, 27)):
        super().__init__()
        self.text_encoder = text_encoder
        self.hidden_states_layers = list(hidden_states_layers)

    def _build_decoder_mask(self, attention_mask, sliding_window=None):
        batch, seq_len = attention_mask.shape
        device = attention_mask.device
        dtype = self.text_encoder.model.embed_tokens.weight.dtype
        neg_inf = torch.finfo(dtype).min

        q_positions = torch.arange(seq_len, device=device)
        k_positions = q_positions
        blocked = k_positions.unsqueeze(0) > q_positions.unsqueeze(1)
        if sliding_window is not None:
            blocked = blocked | (
                k_positions.unsqueeze(0) < (q_positions.unsqueeze(1) - sliding_window + 1)
            )

        mask = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
        mask = mask.masked_fill(blocked, neg_inf)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, seq_len, seq_len).clone()

        key_padding = attention_mask[:, None, None, :].eq(0)
        query_padding = attention_mask[:, None, :, None].eq(0)
        mask = mask.masked_fill(key_padding, neg_inf)
        return mask.masked_fill(query_padding, neg_inf)

    def forward(self, input_ids, attention_mask):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        position_ids = position_ids.expand(batch, -1)

        attn_mapping = {
            "full_attention": self._build_decoder_mask(attention_mask),
        }
        sliding_window = getattr(self.text_encoder.config, "sliding_window", None)
        if getattr(self.text_encoder.model, "has_sliding_layers", False) and sliding_window:
            attn_mapping["sliding_attention"] = self._build_decoder_mask(
                attention_mask, sliding_window=sliding_window
            )

        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attn_mapping,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        out = torch.stack(
            [output.hidden_states[k] for k in self.hidden_states_layers], dim=1
        )
        batch_size, num_channels, seq_len, hidden_dim = out.shape
        return out.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_channels * hidden_dim
        )


class ExportFriendlyQwen3RotaryEmbedding(nn.Module):
    """Equivalent rotary embedding with export-friendly broadcast math."""

    def __init__(self, rotary_emb):
        super().__init__()
        self.register_buffer("inv_freq", rotary_emb.inv_freq.detach().clone(), persistent=False)
        self.attention_scaling = rotary_emb.attention_scaling

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def patch_diffusers_timestep_embedding():
    """Replace diffusers.get_timestep_embedding with a slice-free implementation."""
    import diffusers.models.embeddings as emb_mod

    def _export_friendly_get_timestep_embedding(
        timesteps: torch.Tensor,
        embedding_dim: int,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 1,
        scale: float = 1,
        max_period: int = 10000,
    ) -> torch.Tensor:
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - downscale_freq_shift)
        freqs = torch.exp(exponent)
        timesteps_f = timesteps.to(torch.float32).reshape(-1, 1)
        emb = timesteps_f * freqs.view(1, -1)
        emb = scale * emb
        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        if flip_sin_to_cos:
            emb = torch.cat([cos_emb, sin_emb], dim=-1)
        else:
            emb = torch.cat([sin_emb, cos_emb], dim=-1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    emb_mod.get_timestep_embedding = _export_friendly_get_timestep_embedding


def patch_diffusers_adaln_chunk():
    """Replace AdaLayerNorm chunk ops with slice-based splits to avoid SplitToSequence."""
    import diffusers.models.normalization as norm_mod

    def _adaln_zero_forward(self, x, timestep=None, class_labels=None, hidden_dtype=None, emb=None):
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        emb = emb.view(emb.shape[0], 6, -1)
        shift_msa = emb[:, 0, :]
        scale_msa = emb[:, 1, :]
        gate_msa = emb[:, 2, :]
        shift_mlp = emb[:, 3, :]
        scale_mlp = emb[:, 4, :]
        gate_mlp = emb[:, 5, :]
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _adaln_zero_single_forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        emb = emb.view(emb.shape[0], 3, -1)
        shift_msa = emb[:, 0, :]
        scale_msa = emb[:, 1, :]
        gate_msa = emb[:, 2, :]
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa

    def _adaln_continuous_forward(self, x, conditioning_embedding):
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        emb = emb.view(emb.shape[0], 2, -1)
        scale = emb[:, 0, :]
        shift = emb[:, 1, :]
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x

    norm_mod.AdaLayerNormZero.forward = _adaln_zero_forward
    norm_mod.AdaLayerNormZeroSingle.forward = _adaln_zero_single_forward
    norm_mod.AdaLayerNormContinuous.forward = _adaln_continuous_forward


def patch_diffusers_rotary_pos_embed():
    """Replace get_1d_rotary_pos_embed to avoid repeat_interleave output_size quirks."""
    import diffusers.models.embeddings as emb_mod
    import numpy as np

    def _export_friendly_get_1d_rotary_pos_embed(
        dim: int,
        pos,
        theta: float = 10000.0,
        use_real=False,
        linear_factor=1.0,
        ntk_factor=1.0,
        repeat_interleave_real=True,
        freqs_dtype=torch.float32,
    ):
        assert dim % 2 == 0

        if isinstance(pos, int):
            pos_t = torch.arange(pos)
        elif isinstance(pos, np.ndarray):
            pos_t = torch.from_numpy(pos)
        else:
            pos_t = pos

        theta = theta * ntk_factor
        freqs = (
            1.0
            / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos_t.device) / dim))
            / linear_factor
        )
        freqs = torch.outer(pos_t, freqs)
        if use_real and repeat_interleave_real:
            freqs_cos = freqs.cos()
            freqs_sin = freqs.sin()
            freqs_cos = torch.stack([freqs_cos, freqs_cos], dim=-1).reshape(freqs_cos.shape[0], -1).float()
            freqs_sin = torch.stack([freqs_sin, freqs_sin], dim=-1).reshape(freqs_sin.shape[0], -1).float()
            return freqs_cos, freqs_sin
        elif use_real:
            freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()
            freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()
            return freqs_cos, freqs_sin
        else:
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            return freqs_cis

    emb_mod.get_1d_rotary_pos_embed = _export_friendly_get_1d_rotary_pos_embed


def patch_diffusers_flux2_chunks():
    """Replace chunk-based splits in Flux2 with slice-based versions."""
    from diffusers.models.transformers import transformer_flux2 as flux2_mod

    def _export_get_fused_projections(attn, hidden_states, encoder_hidden_states=None):
        qkv = attn.to_qkv(hidden_states)
        query, key, value = _chunk_last_dim(qkv, 3)

        encoder_query = encoder_key = encoder_value = None
        if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
            added_qkv = attn.to_added_qkv(encoder_hidden_states)
            encoder_query, encoder_key, encoder_value = _chunk_last_dim(added_qkv, 3)

        return query, key, value, encoder_query, encoder_key, encoder_value

    def _export_get_qkv_projections(attn, hidden_states, encoder_hidden_states=None):
        if attn.fused_projections:
            return _export_get_fused_projections(attn, hidden_states, encoder_hidden_states)
        return flux2_mod._get_projections(attn, hidden_states, encoder_hidden_states)

    def _swiglu_forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = _chunk_last_dim(x, 2)
        return self.gate_fn(x1) * x2

    flux2_mod._get_fused_projections = _export_get_fused_projections
    flux2_mod._get_qkv_projections = _export_get_qkv_projections
    flux2_mod.Flux2SwiGLU.forward = _swiglu_forward

def patch_diffusers_flux2_modulation_split():
    """Replace Flux2Modulation.split to avoid SplitToSequence."""
    from diffusers.models.transformers import transformer_flux2 as flux2_mod

    def _split(mod: torch.Tensor, mod_param_sets: int):
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        chunk = mod.shape[-1] // (3 * mod_param_sets)
        params = []
        for i in range(3 * mod_param_sets):
            start = i * chunk
            end = (i + 1) * chunk
            params.append(mod[..., start:end])
        return tuple(tuple(params[3 * i : 3 * (i + 1)]) for i in range(mod_param_sets))

    flux2_mod.Flux2Modulation.split = staticmethod(_split)


class Flux2TransformerWrapper(nn.Module):
    """Thin wrapper: positional args only, returns plain tensor, guidance=None."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids):
        result = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            return_dict=False,
        )
        return result[0]


def _manual_attention(query, key, value, attn_mask=None):
    # query/key/value: (B, S, H, D)
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores + attn_mask
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out.permute(0, 2, 1, 3)


def _split_last_dim(x: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    parts = []
    start = 0
    for size in sizes:
        parts.append(x[..., start : start + size])
        start += size
    return parts


def _chunk_last_dim(x: torch.Tensor, chunks: int) -> list[torch.Tensor]:
    last = x.shape[-1]
    chunk = last // chunks
    return _split_last_dim(x, [chunk] * chunks)


def _export_apply_rotary_emb(x: torch.Tensor, freqs_cis, sequence_dim: int = 1) -> torch.Tensor:
    cos, sin = freqs_cis
    if sequence_dim == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif sequence_dim == 1:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    else:
        raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

    cos = cos.to(x.device)
    sin = sin.to(x.device)

    half = x.shape[-1] // 2
    x_reshaped = x.reshape(*x.shape[:-1], half, 2)
    x_real = x_reshaped[..., 0]
    x_imag = x_reshaped[..., 1]
    x_rotated = torch.stack((-x_imag, x_real), dim=-1).reshape(*x.shape)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


class ExportFlux2AttnProcessor:
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        from diffusers.models.transformers.transformer_flux2 import (
            _get_qkv_projections,
        )
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None and not DISABLE_ROTARY_FOR_QAIRT:
            query = _export_apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = _export_apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = _manual_attention(query, key, value, attention_mask)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            enc_len = encoder_hidden_states.shape[1]
            encoder_hidden_states = hidden_states[:, :enc_len]
            hidden_states = hidden_states[:, enc_len:]
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class ExportFlux2ParallelSelfAttnProcessor:
    def __call__(
        self,
        attn,
        hidden_states,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = _split_last_dim(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor]
        )

        query, key, value = _chunk_last_dim(qkv, 3)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None and not DISABLE_ROTARY_FOR_QAIRT:
            query = _export_apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = _export_apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        attn_out = _manual_attention(query, key, value, attention_mask)
        attn_out = attn_out.flatten(2, 3).to(query.dtype)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        hidden_states = torch.cat([attn_out, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)
        return hidden_states

class VAEDecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents, return_dict=False)[0]


# ============================================================================
# Pipeline loading
# ============================================================================

def load_pipeline(model_id: str, dtype=torch.float32):
    from diffusers import Flux2KleinPipeline
    logger.info("Loading pipeline from %s ...", model_id)
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to("cpu")
    return pipe


def _get_vae_scale_factor(pipe) -> int:
    vae = pipe.vae
    if hasattr(vae, "config") and hasattr(vae.config, "scaling_factor"):
        return int(2 ** (len(vae.config.block_out_channels) - 1))
    return 8


def _compute_latent_dims(height, width, vae_sf):
    latent_h = 2 * (height // (vae_sf * 2))
    latent_w = 2 * (width // (vae_sf * 2))
    return latent_h // 2, latent_w // 2


# ============================================================================
# Input shape builders
# ============================================================================

def build_text_encoder_inputs(max_text_len: int, batch: int = 1):
    return {
        "input_ids": torch.ones(batch, max_text_len, dtype=torch.long),
        "attention_mask": torch.ones(batch, max_text_len, dtype=torch.long),
    }


def _prepare_latent_ids_klein(patch_h, patch_w, batch=1):
    h_ids = torch.arange(patch_h).view(-1, 1).expand(patch_h, patch_w).reshape(-1)
    w_ids = torch.arange(patch_w).view(1, -1).expand(patch_h, patch_w).reshape(-1)
    t_ids = torch.zeros(patch_h * patch_w, dtype=torch.long)
    l_ids = torch.zeros(patch_h * patch_w, dtype=torch.long)
    coords = torch.stack([t_ids, h_ids, w_ids, l_ids], dim=-1)
    return coords.unsqueeze(0).expand(batch, -1, -1)


def _prepare_text_ids_klein(seq_len, batch=1):
    t = torch.arange(1)
    h = torch.arange(1)
    w = torch.arange(1)
    seq = torch.arange(seq_len)
    coords = torch.cartesian_prod(t, h, w, seq)
    return coords.unsqueeze(0).expand(batch, -1, -1)


def build_transformer_inputs(pipe, height, width, max_text_len, dtype=torch.float32):
    t_cfg = pipe.transformer.config
    in_channels = t_cfg.in_channels
    joint_dim = t_cfg.joint_attention_dim
    vae_sf = _get_vae_scale_factor(pipe)
    patch_h, patch_w = _compute_latent_dims(height, width, vae_sf)
    num_tokens = patch_h * patch_w
    batch = 1

    return {
        "hidden_states": torch.randn(batch, num_tokens, in_channels, dtype=dtype),
        "encoder_hidden_states": torch.randn(batch, max_text_len, joint_dim, dtype=dtype),
        "timestep": torch.full((batch,), 0.5, dtype=dtype),
        "img_ids": _prepare_latent_ids_klein(patch_h, patch_w, batch).to(dtype),
        "txt_ids": _prepare_text_ids_klein(max_text_len, batch).to(dtype),
    }


def build_vae_inputs(pipe, height, width, dtype=torch.float32):
    vae_cfg = pipe.vae.config
    latent_ch = getattr(vae_cfg, "latent_channels", 32)
    vae_sf = _get_vae_scale_factor(pipe)
    patch_h, patch_w = _compute_latent_dims(height, width, vae_sf)
    latent_h, latent_w = patch_h * 2, patch_w * 2
    return {
        "latents": torch.randn(1, latent_ch, latent_h, latent_w, dtype=dtype),
    }


# ============================================================================
# Model export (TorchScript or ONNX)
# ============================================================================

def trace_to_torchscript(model, sample_inputs):
    """Trace a PyTorch model to an in-memory TorchScript module."""
    args = tuple(sample_inputs.values())
    with torch.no_grad():
        return torch.jit.trace(model, args)


def export_to_torchscript(model, sample_inputs, pt_path):
    """Export a PyTorch model to TorchScript via tracing."""
    logger.info("Tracing model to TorchScript: %s", pt_path)
    os.makedirs(os.path.dirname(pt_path) or ".", exist_ok=True)

    traced = trace_to_torchscript(model, sample_inputs)
    torch.jit.save(traced, pt_path)

    size_mb = os.path.getsize(pt_path) / (1024 * 1024)
    logger.info("TorchScript saved: %s (%.1f MB)", pt_path, size_mb)
    return pt_path


def _resolve_reshape_shapes(model):
    """Pre-resolve Reshape ops: replace 0 and -1 in shape constants with concrete values."""
    import onnx

    graph = model.graph
    shape_map = {}
    for vi in list(graph.value_info) + list(graph.output) + list(graph.input):
        if vi.type.tensor_type.HasField("shape"):
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                if d.dim_value > 0:
                    dims.append(d.dim_value)
                else:
                    dims = None
                    break
            if dims is not None:
                shape_map[vi.name] = dims

    resolved_count = 0
    for idx, node in enumerate(graph.node):
        if node.op_type == "Reshape" and len(node.input) >= 2:
            output_name = node.output[0]
            if output_name in shape_map:
                concrete_shape = shape_map[output_name]
                shape_array = np.array(concrete_shape, dtype=np.int64)
                shape_name = f"{output_name}__static_shape_{idx}"
                node.input[1] = shape_name
                graph.initializer.append(
                    onnx.numpy_helper.from_array(shape_array, name=shape_name)
                )
                resolved_count += 1

    logger.info("Pre-resolved %d Reshape shape constants", resolved_count)
    return model


def _fold_constant_chains(model):
    """Fold Constant->Cast->Reshape chains so Slice/Gather params become static initializers.

    The dynamo ONNX exporter produces patterns like:
      Constant(val=0) -> Cast(to=int64) -> Reshape([1]) -> Slice(starts=...)
    QAIRT needs Slice params to be constant initializers, not computed values.
    This pass evaluates these chains and replaces them with direct initializers.
    """
    import onnx

    graph = model.graph

    # Build output->node map
    producer = {}
    for node in graph.node:
        for out in node.output:
            producer[out] = node

    # Build initializer value map
    init_vals = {}
    for init in graph.initializer:
        if len(init.raw_data) <= 1024:
            try:
                init_vals[init.name] = onnx.numpy_helper.to_array(init)
            except Exception:
                pass

    # Collect Constant node values
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t is not None:
                    try:
                        init_vals[node.output[0]] = onnx.numpy_helper.to_array(attr.t)
                    except Exception:
                        pass

    def _eval(name, depth=0):
        """Try to evaluate a value to a numpy array by tracing constant chains."""
        if depth > 10:
            return None
        if name in init_vals:
            return init_vals[name]
        if name not in producer:
            return None

        node = producer[name]

        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t is not None:
                    try:
                        val = onnx.numpy_helper.to_array(attr.t)
                        init_vals[name] = val
                        return val
                    except Exception:
                        return None
            return None

        if node.op_type == "Cast":
            src = _eval(node.input[0], depth + 1)
            if src is None:
                return None
            to_type = None
            for attr in node.attribute:
                if attr.name == "to":
                    to_type = attr.i
            if to_type is None:
                return None
            dtype_map = {1: np.float32, 6: np.int32, 7: np.int64, 11: np.float64}
            if to_type in dtype_map:
                val = src.astype(dtype_map[to_type])
                init_vals[name] = val
                return val
            return None

        if node.op_type == "Reshape":
            src = _eval(node.input[0], depth + 1)
            shape = _eval(node.input[1], depth + 1)
            if src is not None and shape is not None:
                try:
                    val = src.reshape(shape.astype(np.int64).tolist())
                    init_vals[name] = val
                    return val
                except Exception:
                    return None
            return None

        if node.op_type == "Unsqueeze":
            src = _eval(node.input[0], depth + 1)
            if src is None:
                return None
            axes = None
            if len(node.input) > 1:
                axes = _eval(node.input[1], depth + 1)
            else:
                for attr in node.attribute:
                    if attr.name == "axes":
                        axes = np.array(list(attr.ints), dtype=np.int64)
            if axes is not None:
                val = src
                for ax in sorted(axes.flatten().tolist()):
                    val = np.expand_dims(val, axis=ax)
                init_vals[name] = val
                return val
            return None

        if node.op_type == "Squeeze":
            src = _eval(node.input[0], depth + 1)
            if src is None:
                return None
            if len(node.input) > 1:
                axes = _eval(node.input[1], depth + 1)
            else:
                axes = None
                for attr in node.attribute:
                    if attr.name == "axes":
                        axes = np.array(list(attr.ints), dtype=np.int64)
            if axes is not None:
                val = np.squeeze(src, axis=tuple(axes.flatten().tolist()))
            else:
                val = np.squeeze(src)
            init_vals[name] = val
            return val

        return None

    # Fold Slice parameters
    folded = 0
    for node in graph.node:
        if node.op_type == "Slice":
            for i in range(1, len(node.input)):
                inp_name = node.input[i]
                if not inp_name or inp_name in {init.name for init in graph.initializer}:
                    continue
                val = _eval(inp_name)
                if val is not None:
                    new_name = f"{inp_name}__folded"
                    graph.initializer.append(
                        onnx.numpy_helper.from_array(val, name=new_name)
                    )
                    node.input[i] = new_name
                    folded += 1

    logger.info("Folded %d Slice parameters into constants", folded)
    return model


def _rename_conflicting_tensors(model):
    """Rename tensor names that conflict with QAIRT internal naming.

    The dynamo exporter produces names like t_0, t_1, ... t_N which collide
    with QAIRT's internal buffer naming scheme, causing 'Duplicate buffer name'
    errors during IR optimization.
    """
    import re

    graph = model.graph
    t_pattern = re.compile(r"^t_\d+$")

    # Collect all names that need renaming
    rename_map = {}
    for node in graph.node:
        for out in node.output:
            if t_pattern.match(out):
                rename_map[out] = f"flux_{out}"

    if not rename_map:
        return model

    # Apply renames to all node inputs and outputs
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp in rename_map:
                node.input[i] = rename_map[inp]
        for i, out in enumerate(node.output):
            if out in rename_map:
                node.output[i] = rename_map[out]

    # Also rename in graph inputs/outputs/value_info
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name in rename_map:
            vi.name = rename_map[vi.name]

    for init in graph.initializer:
        if init.name in rename_map:
            init.name = rename_map[init.name]

    logger.info("Renamed %d tensors to avoid QAIRT naming conflicts", len(rename_map))
    return model


def _save_onnx_model(model, target_path):
    """Save ONNX model, falling back to external data if protobuf is too large."""
    import onnx

    try:
        onnx.save_model(model, target_path)
    except Exception as e:
        logger.warning("onnx.save_model failed (%s). Retrying with external data.", e)
        from onnx import external_data_helper
        external_data_helper.convert_model_to_external_data(
            model,
            all_tensors_to_one_file=True,
            location=os.path.basename(target_path) + ".data",
            size_threshold=1024,
        )
        onnx.save_model(model, target_path)


def simplify_onnx(onnx_path, has_external_data=False):
    """Simplify ONNX model and pre-resolve reshape shapes to avoid QAIRT bugs."""
    import onnx
    from onnxsim import simplify as onnx_simplify

    sim_path = onnx_path.replace(".onnx", "_sim.onnx")
    logger.info("Simplifying ONNX: %s -> %s", onnx_path, sim_path)

    if has_external_data:
        model = onnx.load(onnx_path)
    else:
        model = onnx.load(onnx_path, load_external_data=False)

    model_sim, check = onnx_simplify(model)
    if not check:
        logger.warning("onnxsim validation failed, using simplified model anyway")

    model_sim = onnx.shape_inference.infer_shapes(model_sim)
    model_sim = _resolve_reshape_shapes(model_sim)

    _save_onnx_model(model_sim, sim_path)
    size_mb = os.path.getsize(sim_path) / (1024 * 1024)
    logger.info("Simplified ONNX saved: %s (%.1f MB)", sim_path, size_mb)
    return sim_path


def resolve_onnx_reshapes(onnx_path, out_path=None):
    """Resolve Reshape shapes in an ONNX model without full onnxsim (for large models).

    Uses path-based shape inference to avoid protobuf 2GB serialization limit,
    then loads the model with external data to resolve Reshape shapes in-place.
    """
    import onnx

    if out_path is None:
        out_path = onnx_path.replace(".onnx", "_resolved.onnx")
    logger.info("Resolving Reshape shapes: %s -> %s", onnx_path, out_path)

    # Step 1: Run shape inference via file path (avoids in-memory 2GB protobuf limit).
    # Write output in-place to preserve external data file references.
    onnx.shape_inference.infer_shapes_path(onnx_path, out_path)
    logger.info("Shape inference done (path-based): %s", out_path)

    # Step 2: Load graph structure only (no weights) to fix up the graph.
    model = onnx.load(out_path, load_external_data=False)
    model = _resolve_reshape_shapes(model)
    model = _fold_constant_chains(model)
    model = _rename_conflicting_tensors(model)

    # Step 3: Save back. External data references in initializers still point
    # to the original .data file. If the output is in a different location,
    # we need the data file alongside it.
    src_data = onnx_path + ".data"
    out_data = out_path + ".data"
    # ONNX rejects symlinks/hard links for external data (security check).
    # If in/out are in the same dir, just reference the same .data file.
    # If different dirs, must copy (expensive for 15GB).
    if os.path.exists(src_data):
        if os.path.dirname(os.path.abspath(src_data)) == os.path.dirname(os.path.abspath(out_path)):
            # Same directory — just point to the existing data file
            target_data_name = os.path.basename(src_data)
            logger.info("Reusing external data file: %s", target_data_name)
        elif not os.path.exists(out_data):
            shutil.copy2(src_data, out_data)
            target_data_name = os.path.basename(out_data)
            logger.info("Copied external data: %s (%.1f GB)",
                         out_data, os.path.getsize(out_data) / (1024**3))
        else:
            target_data_name = os.path.basename(out_data)

        for tensor in model.graph.initializer:
            for entry in tensor.external_data:
                if entry.key == "location":
                    entry.value = target_data_name

    onnx.save_model(model, out_path)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info("Resolved ONNX saved: %s (%.1f MB)", out_path, size_mb)
    return out_path


def export_to_onnx(model, sample_inputs, onnx_path, input_names, output_names,
                   opset_version=11, dynamo=False, external_data=False,
                   fallback=False, report=False, artifacts_dir="."):
    """Export a PyTorch model to ONNX format."""
    logger.info("Exporting to ONNX: %s", onnx_path)
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    args = tuple(sample_inputs.values())

    with torch.no_grad():
        torch.onnx.export(
            model,
            args,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=dynamo,
            external_data=external_data,
            fallback=fallback,
            report=report,
            artifacts_dir=artifacts_dir,
        )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    logger.info("ONNX saved: %s (%.1f MB)", onnx_path, size_mb)
    return onnx_path


def export_traced_to_onnx_bundle(traced, sample_inputs, bundle_dir, input_names,
                                 output_names, opset_version=17,
                                 onnx_shape_inference=False):
    """Export a traced TorchScript module to an ONNX directory with external weights."""
    from torch.onnx import _constants, _exporter_states
    from torch.onnx import utils as onnx_utils

    logger.info("Exporting traced model to ONNX bundle: %s", bundle_dir)
    if os.path.isdir(bundle_dir):
        shutil.rmtree(bundle_dir)
    os.makedirs(bundle_dir, exist_ok=True)

    args = tuple(sample_inputs.values())
    onnx_utils._export(
        traced,
        args,
        bundle_dir,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        export_type=_exporter_states.ExportTypes.DIRECTORY,
        onnx_shape_inference=onnx_shape_inference,
    )
    model_proto_path = os.path.join(bundle_dir, _constants.ONNX_ARCHIVE_MODEL_PROTO_NAME)
    logger.info("ONNX bundle saved: %s", bundle_dir)
    return model_proto_path


def infer_onnx_shapes(onnx_path: str, out_path: str | None = None) -> str:
    import onnx
    if out_path is None:
        out_path = onnx_path
    logger.info("Running ONNX shape inference: %s", onnx_path)
    # Use path-based inference to avoid large protobuf serialization in memory.
    onnx.shape_inference.infer_shapes_path(onnx_path, out_path)
    return out_path


def rewrite_onnx_castlike(onnx_path: str, out_path: str | None = None) -> str:
    """Replace CastLike with Cast using the dtype of the second input."""
    import onnx

    if out_path is None:
        out_path = onnx_path

    model = onnx.load(onnx_path)
    vi_map = {v.name: v for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)}
    init_map = {i.name: i for i in model.graph.initializer}

    def _get_elem_type(name: str) -> int | None:
        if name in vi_map:
            t = vi_map[name].type.tensor_type
            return t.elem_type if t.HasField("elem_type") else None
        if name in init_map:
            return init_map[name].data_type
        return None

    replaced = 0
    new_nodes = []
    for node in model.graph.node:
        if node.op_type == "CastLike" and len(node.input) >= 2:
            to_type = _get_elem_type(node.input[1])
            if to_type is None:
                raise RuntimeError(f"CastLike target dtype not found for input {node.input[1]}")
            new_node = onnx.helper.make_node(
                "Cast",
                inputs=[node.input[0]],
                outputs=list(node.output),
                name=node.name or "",
                to=to_type,
            )
            new_nodes.append(new_node)
            replaced += 1
        else:
            new_nodes.append(node)

    if replaced:
        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)
        _save_onnx_model(model, out_path)
        logger.info("Rewrote %d CastLike nodes -> Cast: %s", replaced, out_path)
    else:
        if out_path != onnx_path:
            _save_onnx_model(model, out_path)
    return out_path


# ============================================================================
# Calibration data generation
# ============================================================================

def generate_calibration_data(sample_inputs, num_samples, cal_dir):
    """
    Generate calibration data as raw files + input_list.txt for QAIRT.

    Each calibration sample is a set of raw files (one per input tensor).
    input_list.txt has one line per sample, with space-separated paths to raw files.
    For named inputs: "input_name:=path input_name:=path"
    """
    os.makedirs(cal_dir, exist_ok=True)
    input_list_lines = []

    for i in range(num_samples):
        alpha = (i + 1) / (num_samples + 1)
        sample_parts = []
        for name, tensor in sample_inputs.items():
            if not tensor.is_floating_point():
                # Integer inputs (e.g. input_ids) — keep as-is
                data = tensor.clone().numpy()
            elif tensor.ndim == 1 and tensor.shape[0] == 1:
                # timestep: sweep from near-0 to near-1
                data = np.full(tensor.shape, alpha, dtype=np.float32)
            else:
                # Random data with similar scale
                scale = tensor.abs().mean().item() or 1.0
                data = (np.random.randn(*tensor.shape) * scale).astype(np.float32)

            raw_path = os.path.join(cal_dir, f"sample_{i:04d}_{name}.raw")
            data.tofile(raw_path)
            sample_parts.append(f"{name}:={raw_path}")

        input_list_lines.append(" ".join(sample_parts))

    input_list_path = os.path.join(cal_dir, "input_list.txt")
    with open(input_list_path, "w") as f:
        for line in input_list_lines:
            f.write(line + "\n")

    logger.info("Calibration data: %d samples in %s", num_samples, cal_dir)
    return input_list_path


# ============================================================================
# QAIRT convert + quantize
# ============================================================================

def qairt_convert_quantize(model_path, input_list_path, input_tensor_configs,
                           output_dir, component_name,
                           act_precision=8, weights_precision=8):
    """
    Convert model (ONNX or TorchScript) to quantized DLC using QAIRT Python API.
    """
    import qairt
    from qairt.api.converter.converter_config import CalibrationConfig
    import patch_qairt_reshape  # noqa: F401

    logger.info("QAIRT converting + quantizing %s (W%dA%d) ...",
                component_name, weights_precision, act_precision)

    cal_config = CalibrationConfig(
        dataset=input_list_path,
        act_precision=act_precision,
        weights_precision=weights_precision,
        bias_precision=8,
        per_channel_quantization=True,
    )

    # CRITICAL: Disable Python GC during QAIRT conversion.
    # The QAIRT SDK C++ code holds raw pointers to numpy array data buffers.
    # When Python GCs the numpy arrays (local variables in Op.__init__ methods),
    # the C++ side reads freed memory, producing garbage shape values.
    import gc as _gc
    _gc.disable()
    try:
        converted_model = qairt.convert(
            model_path,
            calibration_config=cal_config,
            input_tensor_config=input_tensor_configs,
        )
    finally:
        _gc.enable()

    dlc_path = os.path.join(output_dir, f"{component_name}.dlc")
    saved_path = converted_model.save(dlc_path)
    size_mb = os.path.getsize(saved_path) / (1024 * 1024)
    logger.info("Saved quantized DLC: %s (%.1f MB)", saved_path, size_mb)
    return saved_path


def qairt_cli_convert_quantize(model_path, input_list_path, input_tensor_configs,
                               output_dir, component_name,
                               act_precision=8, weights_precision=8):
    """
    Fallback: convert ONNX → DLC using CLI tools (qairt-converter + qairt-quantizer).

    Use this when the Python API (qairt.convert) crashes with shape/canonicalization
    errors. The CLI tools sometimes handle edge cases differently.
    """
    import subprocess

    qairt_sdk = os.environ.get("QAIRT_SDK_ROOT", "")
    converter_bin = os.path.join(qairt_sdk, "bin", "x86_64-linux-clang", "qairt-converter") if qairt_sdk else "qairt-converter"
    quantizer_bin = os.path.join(qairt_sdk, "bin", "x86_64-linux-clang", "qairt-quantizer") if qairt_sdk else "qairt-quantizer"

    # Step 1: Convert ONNX → unquantized DLC
    fp_dlc = os.path.join(output_dir, f"{component_name}_fp.dlc")
    conv_cmd = [converter_bin, "--input_network", model_path, "--output_path", fp_dlc]
    for cfg in input_tensor_configs:
        shape_str = ",".join(str(d) for d in cfg["shape"])
        conv_cmd.extend(["--input_dim", cfg["name"], shape_str])

    logger.info("CLI convert: %s", " ".join(conv_cmd))
    result = subprocess.run(conv_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("qairt-converter failed:\nstdout: %s\nstderr: %s", result.stdout, result.stderr)
        raise RuntimeError(f"qairt-converter failed (exit {result.returncode})")
    logger.info("Unquantized DLC: %s", fp_dlc)

    # Step 2: Quantize DLC with calibration data
    dlc_path = os.path.join(output_dir, f"{component_name}.dlc")
    quant_cmd = [
        quantizer_bin,
        "--input_dlc", fp_dlc,
        "--output_dlc", dlc_path,
        "--input_list", input_list_path,
        "--act_bitwidth", str(act_precision),
        "--weights_bitwidth", str(weights_precision),
        "--bias_bitwidth", "8",
        "--use_per_channel_quantization",
    ]

    logger.info("CLI quantize: %s", " ".join(quant_cmd))
    result = subprocess.run(quant_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("qairt-quantizer failed:\nstdout: %s\nstderr: %s", result.stdout, result.stderr)
        raise RuntimeError(f"qairt-quantizer failed (exit {result.returncode})")

    size_mb = os.path.getsize(dlc_path) / (1024 * 1024)
    logger.info("Saved quantized DLC (CLI): %s (%.1f MB)", dlc_path, size_mb)
    return dlc_path


# ============================================================================
# Tokenizer + VAE BN stats (same as XNNPACK version)
# ============================================================================

def copy_tokenizer(pipe, output_dir: str):
    tok_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    pipe.tokenizer.save_pretrained(tok_dir)
    logger.info("Tokenizer saved to %s/", tok_dir)


def save_vae_bn_stats(pipe, output_dir: str):
    vae = pipe.vae
    if not hasattr(vae, "bn"):
        logger.warning("VAE has no .bn attribute — skipping BN stats save.")
        return
    stats = {
        "running_mean": vae.bn.running_mean.detach().cpu().float(),
        "running_var": vae.bn.running_var.detach().cpu().float(),
    }
    torch.save(stats, os.path.join(output_dir, "vae_bn_stats.pt"))
    logger.info("VAE batch-norm stats saved.")


# ============================================================================
# Component export routines
# ============================================================================

def export_text_encoder(pipe, args, output_dir):
    logger.info("=" * 60)
    logger.info("Exporting TEXT ENCODER ...")

    hidden_states_layers = [9, 18, 27]
    pipe.text_encoder.config._attn_implementation = "eager"
    pipe.text_encoder.model.config._attn_implementation = "eager"
    pipe.text_encoder.model.rotary_emb = ExportFriendlyQwen3RotaryEmbedding(
        pipe.text_encoder.model.rotary_emb
    )
    model = Qwen3TextEncoderWrapper(pipe.text_encoder, hidden_states_layers).eval()

    sample_inputs = build_text_encoder_inputs(args.max_text_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sample_inputs_dev = {k: v.to(device) for k, v in sample_inputs.items()}

    traced = trace_to_torchscript(model, sample_inputs_dev).to("cpu")

    onnx_bundle_dir = os.path.join(output_dir, "onnx", "text_encoder")
    onnx_path = export_traced_to_onnx_bundle(
        traced,
        sample_inputs,
        onnx_bundle_dir,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        opset_version=17,
        onnx_shape_inference=False,
    )

    model = model.cpu()
    del traced
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cal_dir = os.path.join(output_dir, "calibration", "text_encoder")
    input_list_path = generate_calibration_data(
        sample_inputs, args.num_calibration_samples, cal_dir
    )

    input_tensor_configs = [
        {"name": "input_ids", "shape": (1, args.max_text_len), "datatype": "int64"},
        {"name": "attention_mask", "shape": (1, args.max_text_len), "datatype": "int64"},
    ]

    convert_fn = qairt_cli_convert_quantize if args.use_cli else qairt_convert_quantize
    dlc_path = convert_fn(
        onnx_path, input_list_path, input_tensor_configs,
        output_dir, "text_encoder",
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dlc_path


def export_transformer(pipe, args, output_dir):
    logger.info("=" * 60)
    logger.info("Exporting TRANSFORMER ...")

    # Replace timestep embedding with export-friendly version (avoid slice ops)
    patch_diffusers_timestep_embedding()
    patch_diffusers_adaln_chunk()
    patch_diffusers_flux2_modulation_split()
    patch_diffusers_rotary_pos_embed()
    patch_diffusers_flux2_chunks()

    # Force export-friendly attention processors (avoid SDPA)
    attn_procs = {}
    for name in pipe.transformer.attn_processors.keys():
        if "single_transformer_blocks" in name:
            attn_procs[name] = ExportFlux2ParallelSelfAttnProcessor()
        else:
            attn_procs[name] = ExportFlux2AttnProcessor()
    pipe.transformer.set_attn_processor(attn_procs)

    model = Flux2TransformerWrapper(pipe.transformer).eval()
    sample_inputs = build_transformer_inputs(
        pipe, args.height, args.width, args.max_text_len, dtype=torch.float32
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sample_inputs_dev = {k: v.to(device) for k, v in sample_inputs.items()}

    onnx_path = export_to_onnx(
        model,
        sample_inputs_dev,
        os.path.join(output_dir, "onnx", "transformer.onnx"),
        input_names=list(sample_inputs.keys()),
        output_names=["output"],
        opset_version=17,
        dynamo=True,
        external_data=True,
        fallback=False,
        report=True,
        artifacts_dir=os.path.join(output_dir, "onnx_artifacts", "transformer"),
    )
    onnx_path = infer_onnx_shapes(
        onnx_path, os.path.join(output_dir, "onnx", "transformer_inferred.onnx")
    )
    onnx_path = rewrite_onnx_castlike(
        onnx_path, os.path.join(output_dir, "onnx", "transformer_fixed.onnx")
    )

    # Resolve Reshape shapes — critical for QAIRT converter which chokes on
    # -1/0 shape placeholders in Reshape ops. This was already done for VAE
    # (via simplify_onnx) but was missing for the transformer path.
    onnx_path = resolve_onnx_reshapes(
        onnx_path, os.path.join(output_dir, "onnx", "transformer_resolved.onnx")
    )

    model = model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cal_dir = os.path.join(output_dir, "calibration", "transformer")
    input_list_path = generate_calibration_data(
        sample_inputs, args.num_calibration_samples, cal_dir
    )

    input_tensor_configs = []
    for name, tensor in sample_inputs.items():
        dt = "float32"
        if tensor.dtype == torch.long or tensor.dtype == torch.int64:
            dt = "int64"
        input_tensor_configs.append({
            "name": name,
            "shape": tuple(tensor.shape),
            "datatype": dt,
        })

    convert_fn = qairt_cli_convert_quantize if args.use_cli else qairt_convert_quantize
    dlc_path = convert_fn(
        onnx_path, input_list_path, input_tensor_configs,
        output_dir, "transformer",
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dlc_path


def export_vae_decoder(pipe, args, output_dir):
    logger.info("=" * 60)
    logger.info("Exporting VAE DECODER ...")

    model = VAEDecoderWrapper(pipe.vae).eval()
    sample_inputs = build_vae_inputs(pipe, args.height, args.width, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sample_inputs_dev = {k: v.to(device) for k, v in sample_inputs.items()}

    # Export to ONNX (better control than TorchScript for QAIRT)
    onnx_dir = os.path.join(output_dir, "onnx")
    onnx_path = export_to_onnx(
        model, sample_inputs_dev,
        os.path.join(onnx_dir, "vae_decoder.onnx"),
        input_names=["latents"],
        output_names=["output"],
        opset_version=14,
    )

    # Simplify ONNX to fold constants and resolve static reshapes
    onnx_sim_path = simplify_onnx(onnx_path)

    # Move model back to CPU and free GPU memory before calibration
    model = model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cal_dir = os.path.join(output_dir, "calibration", "vae_decoder")
    input_list_path = generate_calibration_data(
        sample_inputs, args.num_calibration_samples, cal_dir
    )

    input_tensor_configs = [
        {"name": "latents", "shape": tuple(sample_inputs["latents"].shape)},
    ]

    convert_fn = qairt_cli_convert_quantize if args.use_cli else qairt_convert_quantize
    dlc_path = convert_fn(
        onnx_sim_path, input_list_path, input_tensor_configs,
        output_dir, "vae_decoder",
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dlc_path


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Export FLUX.2-klein-4B to QAIRT W8A8 quantized DLC for Qualcomm HTP/DSP",
    )
    p.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-4B")
    p.add_argument("--output_dir", default="./exported_flux2_klein_qairt")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--max_text_len", type=int, default=512)
    p.add_argument("--num_calibration_samples", type=int, default=20,
                   help="Number of calibration samples for W8A8 quantization")
    p.add_argument("--component",
                   choices=["all", "transformer", "vae", "text_encoder"],
                   default="all")
    p.add_argument("--use_cli", action="store_true",
                   help="Use CLI tools (qairt-converter + qairt-quantizer) instead of Python API")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load pipeline
    pipe = load_pipeline(args.model_id, dtype=torch.float32)

    copy_tokenizer(pipe, str(out))
    save_vae_bn_stats(pipe, str(out))

    # Save metadata
    vae_sf = _get_vae_scale_factor(pipe)
    patch_h, patch_w = _compute_latent_dims(args.height, args.width, vae_sf)
    t_cfg = pipe.transformer.config
    vae_cfg = pipe.vae.config
    meta = {
        "model_id": args.model_id,
        "backend": "qairt_htp",
        "quantization": "w8a8",
        "height": args.height,
        "width": args.width,
        "max_text_len": args.max_text_len,
        "num_calibration_samples": args.num_calibration_samples,
        "is_distilled": getattr(pipe.config, "is_distilled", True),
        "num_inference_steps": 4,
        "vae_scale_factor": vae_sf,
        "patch_dims": [patch_h, patch_w],
        "text_encoder": {
            "hidden_states_layers": [9, 18, 27],
            "max_sequence_length": args.max_text_len,
        },
        "transformer": {
            "in_channels": t_cfg.in_channels,
            "joint_attention_dim": t_cfg.joint_attention_dim,
        },
        "vae": {
            "latent_channels": getattr(vae_cfg, "latent_channels", 32),
        },
    }
    (out / "export_config.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote export_config.json")

    # Export components
    if args.component in ("all", "vae"):
        export_vae_decoder(pipe, args, str(out))

    if args.component in ("all", "text_encoder"):
        export_text_encoder(pipe, args, str(out))

    if args.component in ("all", "transformer"):
        export_transformer(pipe, args, str(out))

    logger.info("=" * 60)
    logger.info("Export complete. Output: %s", out)


if __name__ == "__main__":
    main()
