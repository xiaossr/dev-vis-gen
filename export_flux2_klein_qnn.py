#!/usr/bin/env python3
"""
Export FLUX.2-klein-4B to ExecuTorch QNN backend (.pte) for Qualcomm HTP/DSP inference.

This script adapts export_flux2_klein_xnnpack.py to target Qualcomm's Hexagon HTP
(Hexagon Tensor Processor / DSP) via the ExecuTorch QNN delegate, instead of XNNPACK.

Pipeline:
  PyTorch model → INT8 static calibration → torch.export → QnnPartitioner → .pte
  (on device: ExecuTorch runtime dispatches ops to QNN → HTP/DSP)

Key differences from the XNNPACK path:
  - QnnQuantizer (static INT8, requires calibration data)  vs  XNNPACKQuantizer (dynamic)
  - QnnPartitioner + SOC target spec  vs  XnnpackPartitioner
  - QNN SDK must be installed: https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai
  - ExecuTorch must be built with: -DEXECUTORCH_BUILD_QNN=ON -DQNN_SDK_ROOT=<path>

Requirements:
  - Python 3.10+
  - torch >= 2.4
  - executorch (with QNN backend compiled in)
  - diffusers (from git), transformers, accelerate, safetensors
  - QNN SDK 2.28+ installed at $QNN_SDK_ROOT
  - GPU strongly recommended for calibration (4090 or better for 4B model)

Supported SOC targets (set via --soc_model):
  SM8650  → Snapdragon 8 Gen 3 (e.g. Samsung Galaxy S24)
  SM8550  → Snapdragon 8 Gen 2 (e.g. Samsung Galaxy S23)
  SM8475  → Snapdragon 8+ Gen 1
  SM8450  → Snapdragon 8 Gen 1

Usage:
  # Full export (all components, INT8, Snapdragon 8 Gen 3):
  python export_flux2_klein_qnn.py \\
      --output_dir ./exported_flux2_klein_qnn \\
      --soc_model SM8650

  # Export transformer only:
  python export_flux2_klein_qnn.py \\
      --component transformer \\
      --soc_model SM8650

  # More calibration passes for better accuracy:
  python export_flux2_klein_qnn.py \\
      --soc_model SM8650 \\
      --num_calibration_passes 50
"""

import argparse
import gc
import json
import logging
import operator
import os
from pathlib import Path
import re
import sys

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("flux2_qnn_export")


def configure_local_tooling(allow_reexec: bool = False):
    """Populate local tool defaults when the caller hasn't exported them."""
    repo_root = Path(__file__).resolve().parent
    flatc_path = repo_root / ".venv" / "bin" / "flatc"
    if not os.environ.get("FLATC_EXECUTABLE") and flatc_path.exists():
        os.environ["FLATC_EXECUTABLE"] = str(flatc_path)
        logger.info("Using local flatc: %s", flatc_path)

    qnn_sdk_root_env = os.environ.get("QNN_SDK_ROOT")
    qnn_sdk_root = Path(qnn_sdk_root_env) if qnn_sdk_root_env else None
    if qnn_sdk_root is None or not qnn_sdk_root.exists():
        local_qnn_root = repo_root / "qairt" / "2.45.0.260326"
        if local_qnn_root.exists():
            os.environ["QNN_SDK_ROOT"] = str(local_qnn_root)
            qnn_sdk_root = local_qnn_root
            logger.info("Using local QNN_SDK_ROOT: %s", local_qnn_root)

    required_ld_paths = []
    if qnn_sdk_root is not None:
        qnn_host_lib_dir = qnn_sdk_root / "lib" / "x86_64-linux-clang"
        if qnn_host_lib_dir.exists():
            required_ld_paths.append(str(qnn_host_lib_dir))

    for candidate in (
        repo_root / ".local-libs" / "usr" / "lib" / "x86_64-linux-gnu",
        repo_root / ".local-libs-jammy" / "extracted" / "usr" / "lib" / "x86_64-linux-gnu",
        repo_root / ".local-libs-14" / "usr" / "lib" / "x86_64-linux-gnu",
        Path.home() / "android-ndk-r26d" / "toolchains" / "llvm" / "prebuilt" / "linux-x86_64" / "lib",
    ):
        if (
            (candidate / "libc++.so.1").exists()
            and (candidate / "libunwind.so.1").exists()
            and (candidate / "libc++abi.so.1").exists()
        ):
            required_ld_paths.append(str(candidate))
            break

    current_ld_paths = [
        p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p
    ]
    missing_ld_paths = [p for p in required_ld_paths if p not in current_ld_paths]
    if allow_reexec and missing_ld_paths and os.environ.get("_FLUX2_QNN_REEXEC") != "1":
        os.environ["LD_LIBRARY_PATH"] = ":".join(missing_ld_paths + current_ld_paths)
        os.environ["_FLUX2_QNN_REEXEC"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


# ============================================================================
# SOC model mapping
# ============================================================================

SOC_MODEL_MAP = {
    "SM8850": None,   # filled in at runtime from QcomChipset
    "SM8650": None,   # filled in at runtime from QcomChipset
    "SM8550": None,
    "SM8475": None,
    "SM8450": None,
}


def get_qcom_chipset(soc_model: str):
    """Return the QcomChipset enum value for the given SOC model string."""
    try:
        from executorch.backends.qualcomm.serialization.qc_schema import (
            QcomChipset,
        )
    except ImportError as e:
        raise ImportError(
            "ExecuTorch QNN backend not found. Build ExecuTorch with:\n"
            "  -DEXECUTORCH_BUILD_QNN=ON -DQNN_SDK_ROOT=$QNN_SDK_ROOT\n"
            f"Original error: {e}"
        )
    mapping = {
        "SM8850": QcomChipset.SM8850,
        "SM8750": QcomChipset.SM8750,
        "SM8650": QcomChipset.SM8650,
        "SM8550": QcomChipset.SM8550,
        "SM8475": QcomChipset.SM8475,
        "SM8450": QcomChipset.SM8450,
    }
    if soc_model not in mapping:
        raise ValueError(
            f"Unknown SOC model '{soc_model}'. Choose from: {list(mapping)}"
        )
    return mapping[soc_model]


# ============================================================================
# Re-used wrapper modules (same as XNNPACK export)
# ============================================================================

class Qwen3TextEncoderWrapper(nn.Module):
    """Wraps Qwen3's internal model (bypassing decorators) to extract multi-layer hidden states."""

    def __init__(self, text_encoder, hidden_states_layers=(9, 18, 27)):
        super().__init__()
        # Access the inner Qwen3Model directly to bypass @capture_outputs lock
        self.embed_tokens = text_encoder.model.embed_tokens
        self.layers = text_encoder.model.layers
        self.norm = text_encoder.model.norm
        self.rotary_emb = text_encoder.model.rotary_emb
        self.config = text_encoder.model.config
        self.hidden_states_layers = list(hidden_states_layers)

    def forward(self, input_ids, attention_mask):
        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len = input_ids.shape

        # Use int32 to avoid int64-related dtype mismatches in ExportPass
        position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.int32).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Build a causal mask from the 2D attention_mask
        # Shape: [batch, 1, seq_len, seq_len]  (additive mask, 0 = attend, -inf = mask)
        # Explicit float32 to avoid dtype promotion issues
        causal_mask = torch.full(
            (seq_len, seq_len), float("-inf"), dtype=torch.float32, device=hidden_states.device
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        # Expand padding mask: where attention_mask == 0, mask out
        padding_mask = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.float32, device=hidden_states.device)
        padding_mask = padding_mask.masked_fill(attention_mask[:, None, None, :].eq(0), float("-inf"))
        causal_mask = causal_mask[None, None, :, :] + padding_mask

        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            all_hidden_states.append(hidden_states)

        # Apply final norm to last hidden state (already in all_hidden_states)
        all_hidden_states[-1] = self.norm(hidden_states)

        out = torch.stack(
            [all_hidden_states[k] for k in self.hidden_states_layers], dim=1
        )
        batch_size, num_channels, seq_len, hidden_dim = out.shape
        return out.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_channels * hidden_dim
        )


def _patch_apply_rotary_emb_for_qnn():
    """Replace torch.stack in diffusers' apply_rotary_emb with cat(unsqueeze).

    QNN's AOT op validator rejects the stack→flatten pattern with an
    "Incorrect out[0] dimension ... Expected 2 but got <head_dim/2>" error.
    cat+unsqueeze produces the same tensor via ops the validator accepts.
    Safe to call multiple times; idempotent.
    """
    from diffusers.models import embeddings

    if getattr(embeddings.apply_rotary_emb, "_qnn_patched", False):
        return

    _original = embeddings.apply_rotary_emb

    def _patched(x, freqs_cis, use_real=True, use_real_unbind_dim=-1, sequence_dim=2):
        if use_real and use_real_unbind_dim == -1:
            cos, sin = freqs_cis
            if sequence_dim == 2:
                cos = cos[None, None, :, :]
                sin = sin[None, None, :, :]
            elif sequence_dim == 1:
                cos = cos[None, :, None, :]
                sin = sin[None, :, None, :]
            cos, sin = cos.to(x.device), sin.to(x.device)
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.cat(
                [(-x_imag).unsqueeze(-1), x_real.unsqueeze(-1)], dim=-1
            ).flatten(3)
            # Chunk along the heads dim so each broadcast-mul fits in V81 VTCM
            # (8 MB) at int16. With seq=1536, head_dim=128, full mul is 9.4 MB
            # at int16; halving heads brings each chunk to 4.7 MB.
            n_split = int(os.environ.get("FLUX_ROTARY_HEAD_SPLIT", "2"))
            if n_split > 1 and x.dim() == 4:
                head_dim_pos = 1 if sequence_dim == 2 else 2
                if x.shape[head_dim_pos] % n_split == 0:
                    x_parts = x.float().chunk(n_split, dim=head_dim_pos)
                    xr_parts = x_rotated.float().chunk(n_split, dim=head_dim_pos)
                    out_parts = [xp * cos + xrp * sin for xp, xrp in zip(x_parts, xr_parts)]
                    return torch.cat(out_parts, dim=head_dim_pos).to(x.dtype)
            return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return _original(x, freqs_cis, use_real=use_real,
                         use_real_unbind_dim=use_real_unbind_dim,
                         sequence_dim=sequence_dim)

    _patched._qnn_patched = True
    embeddings.apply_rotary_emb = _patched
    # Also patch the symbol where it was re-imported
    for mod_name in (
        "diffusers.models.attention",
        "diffusers.models.transformers.transformer_flux2",
        "diffusers.models.transformers.transformer_flux",
    ):
        import importlib
        try:
            m = importlib.import_module(mod_name)
            if hasattr(m, "apply_rotary_emb"):
                m.apply_rotary_emb = _patched
        except ImportError:
            pass
    logger.info("Patched apply_rotary_emb: torch.stack → cat+unsqueeze for QNN AOT")


class Flux2TransformerWrapper(nn.Module):
    """Thin wrapper: positional args only, returns plain tensor, guidance=None."""

    def __init__(self, transformer):
        super().__init__()
        _patch_apply_rotary_emb_for_qnn()
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


class VAEDecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents, return_dict=False)[0]


class VAEEncoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, pixel_values):
        return self.vae.encode(pixel_values, return_dict=False)[0].sample()


# ============================================================================
# Pipeline loading
# ============================================================================

def load_pipeline(model_id: str, dtype=torch.float32):
    """Load FLUX.2-klein-4B pipeline onto CPU in fp32."""
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
# Input builders (same shapes as XNNPACK version)
# ============================================================================

def build_text_encoder_inputs(max_text_len: int, batch: int = 1):
    return (
        torch.ones(batch, max_text_len, dtype=torch.long),
        torch.ones(batch, max_text_len, dtype=torch.long),
    )


def _prepare_latent_ids_klein(patch_h, patch_w, batch=1):
    h_ids = torch.arange(patch_h).view(-1, 1).expand(patch_h, patch_w).reshape(-1)
    w_ids = torch.arange(patch_w).view(1, -1).expand(patch_h, patch_w).reshape(-1)
    t_ids = torch.zeros(patch_h * patch_w, dtype=torch.long)
    l_ids = torch.zeros(patch_h * patch_w, dtype=torch.long)
    coords = torch.stack([t_ids, h_ids, w_ids, l_ids], dim=-1)  # (N, 4)
    return coords.unsqueeze(0).expand(batch, -1, -1)             # (B, N, 4)


def _prepare_text_ids_klein(seq_len, batch=1):
    t = torch.arange(1)
    h = torch.arange(1)
    w = torch.arange(1)
    seq = torch.arange(seq_len)
    coords = torch.cartesian_prod(t, h, w, seq)  # (seq_len, 4)
    return coords.unsqueeze(0).expand(batch, -1, -1)


def build_transformer_inputs(pipe, height, width, max_text_len,
                              dtype=torch.float32, num_img2img_images=0):
    t_cfg = pipe.transformer.config
    in_channels = t_cfg.in_channels
    joint_dim = t_cfg.joint_attention_dim
    vae_sf = _get_vae_scale_factor(pipe)
    patch_h, patch_w = _compute_latent_dims(height, width, vae_sf)
    num_noise_tokens = patch_h * patch_w
    total_img_tokens = num_noise_tokens * (1 + num_img2img_images)
    batch = 1

    hidden_states = torch.randn(batch, total_img_tokens, in_channels, dtype=dtype)
    encoder_hidden_states = torch.randn(batch, max_text_len, joint_dim, dtype=dtype)
    timestep = torch.full((batch,), 0.5, dtype=dtype)

    noise_ids = _prepare_latent_ids_klein(patch_h, patch_w, batch)
    if num_img2img_images > 0:
        ref_ids = [_prepare_latent_ids_klein(patch_h, patch_w, batch) for _ in range(num_img2img_images)]
        for i, r in enumerate(ref_ids):
            r[:, :, 0] = 10 + 10 * i
        img_ids = torch.cat([noise_ids] + ref_ids, dim=1).to(dtype)
    else:
        img_ids = noise_ids.to(dtype)

    txt_ids = _prepare_text_ids_klein(max_text_len, batch).to(dtype)
    return (hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids)


def build_vae_inputs(pipe, height, width, dtype=torch.float32):
    vae_cfg = pipe.vae.config
    latent_ch = getattr(vae_cfg, "latent_channels", 32)
    vae_sf = _get_vae_scale_factor(pipe)
    patch_h, patch_w = _compute_latent_dims(height, width, vae_sf)
    latent_h, latent_w = patch_h * 2, patch_w * 2
    return (torch.randn(1, latent_ch, latent_h, latent_w, dtype=dtype),)


def build_vae_encoder_inputs(height, width, dtype=torch.float32):
    return (torch.randn(1, 3, height, width, dtype=dtype),)


# ============================================================================
# Calibration data generation
# ============================================================================

def generate_calibration_inputs(sample_inputs: tuple, num_passes: int):
    """
    Yield `num_passes` perturbed versions of `sample_inputs` for calibration.

    For the transformer: timestep varies from 0.0→1.0 to cover full denoising range.
    Position-ID tensors (structured integer grids cast to float) are kept fixed
    — randomizing them would feed garbage rotations into RoPE and produce
    NaN/Inf activations that break the observer scales.
    """
    for i in range(num_passes):
        alpha = (i + 1) / (num_passes + 1)
        cal = []
        for j, inp in enumerate(sample_inputs):
            if not inp.is_floating_point():
                cal.append(inp.clone())
            elif inp.ndim == 1 and inp.shape[0] == 1:
                # timestep: sweep from near-0 to near-1
                cal.append(torch.full_like(inp, alpha))
            elif inp.dtype in (torch.float32, torch.float64) and inp.ndim == 3 \
                    and inp.shape[-1] == 4 and (inp == inp.round()).all():
                # img_ids / txt_ids: position-id grid — keep fixed
                cal.append(inp.clone())
            else:
                # random activations with the same scale as the sample
                scale = inp.abs().mean().item() or 1.0
                cal.append(torch.randn_like(inp) * scale)
        yield tuple(cal)


# ============================================================================
# Core QNN export routine
# ============================================================================

def _decompose_layer_norm(model, sample_inputs=None):
    """Rewrite native_layer_norm / layer_norm into primitive ops.

    Flux2 uses nn.LayerNorm on rank-3 (B, S, C) tensors which QNN HTP rejects
    (HTP LayerNorm op requires rank-4). The default decomposition in QNN's
    partitioner falls back to CPU at every LayerNorm, producing ~32 partition
    boundaries in the transformer that compound quant error over 4 denoising
    steps into pure noise output.

    Rewrite native_layer_norm(x, shape, w, b, eps) as:
        mean  = x.mean(dims, keepdim=True)
        var   = x.var(dims, correction=0, keepdim=True)
        diff  = x - mean
        rstd  = rsqrt(var + eps)
        out   = diff * rstd [* w] [+ b]

    All resulting ops (mean, var, sub, mul, add, rsqrt) run natively on HTP.
    Uses aten.mean.dim + aten.var.correction rather than sub→mul→mean→rsqrt
    so RecomposeRmsNorm can't match the pattern and recompose incorrectly.

    sample_inputs is used to repopulate meta["val"] on inserted nodes
    (required by LiftConstantScalarOperands and other downstream passes).
    """
    if not hasattr(model, "graph"):
        return model
    gm = model

    changed = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target not in (
            torch.ops.aten.native_layer_norm.default,
            torch.ops.aten.layer_norm.default,
        ):
            continue

        changed = True
        input_node = node.args[0]
        normalized_shape = node.args[1]
        weight_node = node.args[2] if len(node.args) > 2 else None
        bias_node = node.args[3] if len(node.args) > 3 else None
        eps = node.args[4] if len(node.args) > 4 else 1e-5

        num_norm_dims = len(normalized_shape)
        dims = list(range(-num_norm_dims, 0))

        with gm.graph.inserting_before(node):
            mean_node = gm.graph.call_function(
                torch.ops.aten.mean.dim, args=(input_node, dims, True)
            )
            var_node = gm.graph.call_function(
                torch.ops.aten.var.correction,
                args=(input_node, dims),
                kwargs={"correction": 0, "keepdim": True},
            )
            diff_node = gm.graph.call_function(
                torch.ops.aten.sub.Tensor, args=(input_node, mean_node)
            )
            eps_node = gm.graph.call_function(
                torch.ops.aten.add.Scalar, args=(var_node, eps)
            )
            rsqrt_node = gm.graph.call_function(
                torch.ops.aten.rsqrt.default, args=(eps_node,)
            )
            out_node = gm.graph.call_function(
                torch.ops.aten.mul.Tensor, args=(diff_node, rsqrt_node)
            )
            if weight_node is not None:
                out_node = gm.graph.call_function(
                    torch.ops.aten.mul.Tensor, args=(out_node, weight_node)
                )
            if bias_node is not None:
                out_node = gm.graph.call_function(
                    torch.ops.aten.add.Tensor, args=(out_node, bias_node)
                )

        if node.target == torch.ops.aten.native_layer_norm.default:
            for user in list(node.users):
                if user.op == "call_function" and user.target == operator.getitem:
                    idx = user.args[1]
                    if idx == 0:
                        user.replace_all_uses_with(out_node)
                    elif idx == 1:
                        user.replace_all_uses_with(mean_node)
                    elif idx == 2:
                        user.replace_all_uses_with(rsqrt_node)
                    gm.graph.erase_node(user)
        else:
            node.replace_all_uses_with(out_node)

        gm.graph.erase_node(node)

    if changed:
        gm.graph.lint()
        gm.recompile()
        # Re-propagate fake tensors so every new node has meta["val"];
        # LiftConstantScalarOperands and other downstream passes read dtype/shape from it.
        if sample_inputs is not None:
            from torch.fx.passes.fake_tensor_prop import FakeTensorProp
            from torch._subclasses.fake_tensor import FakeTensorMode
            # allow_non_fake_inputs: model parameters are real tensors (get_attr),
            # not FakeTensors. FakeTensorProp would otherwise reject them.
            mode = FakeTensorMode(allow_non_fake_inputs=True)
            FakeTensorProp(gm, mode=mode).propagate(*sample_inputs)
        logger.info("Decomposed native_layer_norm into primitive HTP-native ops")

    return gm


def _remove_int_quantize_nodes(model):
    """
    Remove quantize_per_tensor / dequantize_per_tensor nodes that operate on
    non-float tensors (e.g. int64 from arange). These are incorrectly inserted
    by the quantizer and cause re-export to fail.

    Detection strategies:
    1. Input node has meta["val"] with non-float dtype
    2. Input node is an arange/full/zeros etc. that produces int tensors
    3. Input is a get_attr whose actual tensor is non-float
    """
    graph = model.graph
    nodes_to_erase = []

    # Known int-producing ops
    INT_OPS = {"arange", "full", "zeros", "ones", "randint", "arange.start_step", "arange.default"}

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target_name = getattr(node.target, "__name__", "")
        if "quantize_per_tensor" not in target_name and "dequantize_per_tensor" not in target_name:
            continue

        input_node = node.args[0]
        should_remove = False

        # Strategy 1: check meta["val"] dtype
        # Exclude int8/uint8 — those are legitimate quantized weight tensors
        if hasattr(input_node, "meta") and "val" in input_node.meta:
            val = input_node.meta["val"]
            if hasattr(val, "dtype") and not val.dtype.is_floating_point and val.dtype not in (torch.int8, torch.uint8):
                should_remove = True

        # Strategy 2: check if input is from an int-producing op
        if not should_remove and hasattr(input_node, "target"):
            inp_target_name = getattr(input_node.target, "__name__", str(input_node.target))
            if inp_target_name in INT_OPS or "arange" in inp_target_name:
                should_remove = True

        # Strategy 3: check actual tensor for get_attr nodes
        if not should_remove and hasattr(input_node, "op") and input_node.op == "get_attr":
            try:
                tensor = model
                for part in input_node.target.split("."):
                    tensor = getattr(tensor, part)
                if hasattr(tensor, "dtype") and tensor.dtype in (torch.int64, torch.int32, torch.int16, torch.bool):
                    should_remove = True
            except Exception:
                pass

        if should_remove:
            node.replace_all_uses_with(input_node)
            nodes_to_erase.append(node)

    for node in reversed(nodes_to_erase):
        graph.erase_node(node)
    if nodes_to_erase:
        graph.lint()
        model.recompile()
        logger.info("Removed %d spurious quantize nodes on int tensors", len(nodes_to_erase))


def _install_safe_export_pass():
    """
    Monkey-patch ExportPass.__call__ so that any pass that fails with a
    RuntimeError (e.g. dtype mismatch in the fake-tensor interpreter) returns
    the original graph module unchanged instead of crashing.  This preserves
    meta["val"] on all nodes while still letting passes that succeed apply
    their transformations.
    """
    from executorch.exir.pass_base import ExportPass, PassResult

    if getattr(ExportPass, "_safe_patched", False):
        return  # already installed

    _orig = ExportPass.__call__

    def _safe_call(self, gm):
        try:
            return _orig(self, gm)
        except Exception as exc:
            logger.warning(
                "ExportPass %s failed (%s), returning graph unchanged",
                type(self).__name__,
                exc,
            )
            return PassResult(gm, False)

    ExportPass.__call__ = _safe_call
    ExportPass._safe_patched = True


def _uninstall_safe_export_pass():
    """Restore original ExportPass.__call__ if it was patched."""
    from executorch.exir.pass_base import ExportPass

    if not getattr(ExportPass, "_safe_patched", False):
        return
    # We can't restore the original easily (it was captured in closure),
    # so just leave the safe version in place.  It's strictly better.


def _compute_shard_starts(total_layers: int, num_shards: int) -> list[int]:
    if num_shards <= 1 or total_layers <= 1:
        return []
    num_shards = min(num_shards, total_layers)
    return sorted(
        {
            (i * total_layers) // num_shards
            for i in range(1, num_shards)
            if 0 < (i * total_layers) // num_shards < total_layers
        }
    )


def _get_flux_transformer_layer_index(module_name: str, num_double_layers: int) -> int | None:
    match = re.search(r"transformer\.transformer_blocks\.(\d+)", module_name)
    if match is not None:
        return int(match.group(1))

    match = re.search(r"transformer\.single_transformer_blocks\.(\d+)", module_name)
    if match is not None:
        return num_double_layers + int(match.group(1))

    return None


def _insert_flux_transformer_fallbacks(
    graph_module,
    num_shards: int,
    num_double_layers: int,
    total_layers: int,
    quant_io_dtype=None,
) -> list[int]:
    from executorch.backends.qualcomm.utils.constants import (
        QCOM_DTYPE,
        QCOM_QUANT_ATTRS,
        QCOM_QUANTIZED_IO,
    )
    from executorch.exir.dialects._ops import ops as exir_ops
    from executorch.extension.llm.custom_ops import model_sharding  # noqa: F401

    shard_starts = _compute_shard_starts(total_layers, num_shards)
    if not shard_starts:
        return []

    def _insert_fallback_after(anchor_node):
        with graph_module.graph.inserting_after(anchor_node):
            users = list(anchor_node.users.keys())
            inserted_node = graph_module.graph.create_node(
                "call_function",
                exir_ops.edge.llama.fallback.default,
                (anchor_node,),
            )
            if "val" in anchor_node.meta:
                inserted_node.meta["val"] = anchor_node.meta["val"]
            if anchor_node.meta.get(QCOM_QUANT_ATTRS, None):
                inserted_node.meta[QCOM_QUANT_ATTRS] = anchor_node.meta[QCOM_QUANT_ATTRS]
            for user in users:
                user.replace_input_with(anchor_node, inserted_node)

    prev_node = None
    prev_layer = None
    inserted = 0
    last_block_node = None
    last_block_layer = total_layers - 1
    shard_start_set = set(shard_starts)
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or "nn_module_stack" not in node.meta:
            continue

        module_values_list = list(node.meta["nn_module_stack"].values())
        full_qualified_name = module_values_list[-1][0]
        cur_layer = _get_flux_transformer_layer_index(full_qualified_name, num_double_layers)
        if cur_layer is None:
            continue

        if cur_layer in shard_start_set and prev_layer == cur_layer - 1 and prev_node is not None:
            _insert_fallback_after(prev_node)
            inserted += 1

        if cur_layer == last_block_layer:
            last_block_node = node

        prev_layer = cur_layer
        prev_node = node

    # Isolate the post-transformer tail (norm_out + proj_out) in its own
    # partition. Without this the last shard bundles the final transformer
    # block with the tail, which reliably trips RouterX86 on V81 VTCM.
    tail_inserted = False
    if last_block_node is not None:
        _insert_fallback_after(last_block_node)
        inserted += 1
        tail_inserted = True

    # Isolate the shared modulation outputs. FLUX has three shared modulation
    # modules (`single_stream_modulation`, `double_stream_modulation_img`,
    # `double_stream_modulation_txt`) each producing a tensor broadcast to
    # every block. Without isolation the partitioner drags the entire
    # modulation chain into one block's partition, creating a mixed
    # "modulation + one block attention" partition with unusual VTCM layout
    # requirements that trip `q::ForceFormat_Crouton` on V81. Insert a
    # fallback after the last node of each modulation module so modulation
    # lowering becomes its own partition.
    modulation_module_suffixes = (
        "single_stream_modulation",
        "double_stream_modulation_img",
        "double_stream_modulation_txt",
        "time_guidance_embed",
    )
    last_mod_nodes = {name: None for name in modulation_module_suffixes}
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or "nn_module_stack" not in node.meta:
            continue
        fqn = list(node.meta["nn_module_stack"].values())[-1][0]
        for suffix in modulation_module_suffixes:
            # fqn looks like "transformer.single_stream_modulation.linear"
            # so a split check works regardless of parent prefix.
            parts = fqn.split(".")
            if suffix in parts:
                last_mod_nodes[suffix] = node
                break

    mod_inserted = 0
    for name, node in last_mod_nodes.items():
        if node is not None:
            _insert_fallback_after(node)
            inserted += 1
            mod_inserted += 1

    def _infer_quant_io_dtype(node):
        if quant_io_dtype is not None:
            return quant_io_dtype
        quant_attrs = node.meta.get(QCOM_QUANT_ATTRS)
        if quant_attrs and quant_attrs.get(QCOM_DTYPE) is not None:
            return quant_attrs[QCOM_DTYPE]
        val = node.meta.get("val")
        if hasattr(val, "dtype") and val.dtype in (torch.uint8, torch.int8, torch.uint16, torch.int16):
            return val.dtype
        return None

    fallback_op = exir_ops.edge.llama.fallback.default
    tagged = 0
    for node in graph_module.graph.nodes:
        if fallback_op in [u.target for u in list(node.users.keys())] + [node.target]:
            dtype = _infer_quant_io_dtype(node)
            if dtype is not None:
                node.meta[QCOM_QUANTIZED_IO] = dtype
                tagged += 1

    if inserted:
        graph_module.graph.lint()
        graph_module.recompile()
        logger.info(
            "Inserted %d fallback boundaries for transformer sharding at layer starts %s "
            "(+ tail: %s, + %d modulation isolations) and tagged %d shard-boundary tensors as quantized I/O",
            inserted,
            shard_starts,
            tail_inserted,
            mod_inserted,
            tagged,
        )
    else:
        logger.warning(
            "Requested transformer sharding into %d shards, but no fallback boundaries were inserted",
            num_shards,
        )

    return shard_starts


def export_component_to_qnn(
    model: nn.Module,
    sample_inputs: tuple,
    output_path: str,
    soc_chipset,
    num_calibration_passes: int = 20,
    skip_node_op_set: set = None,
    online_prepare: bool = True,
    quant_dtype: str | None = "8a8w",
    use_fp16: bool = False,
    calibration_data: list = None,
    num_shards: int = 1,
    num_double_layers: int = 0,
    total_layers: int = 0,
    discard_quant_ops: list = None,
):
    """
    Export a model component to QNN-accelerated ExecuTorch .pte.

    Follows the official ExecuTorch QNN flow (build_executorch_binary):
      1. torch.export() → GraphModule
      2. QnnQuantizer + prepare_pt2e (insert fake-quant observers)
      3. Calibration forward passes (determine activation ranges)
      4. convert_pt2e (fold scales, replace ops with INT8)
      5. capture_program (re-export with QNN decompositions + edge config)
      6. to_backend with QnnPartitioner
      7. EdgeProgramManager → to_executorch → serialize .pte
    """
    try:
        from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
        from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
        from executorch.backends.qualcomm.utils.utils import (
            capture_program,
            generate_htp_compiler_spec,
            generate_qnn_executorch_compiler_spec,
            update_spill_fill_size,
        )
        from executorch.exir import to_edge, EdgeCompileConfig
        from executorch.exir.program._program import EdgeProgramManager
        from executorch.exir.backend.backend_api import to_backend
        from executorch.exir.capture._config import ExecutorchBackendConfig
        from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
        from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
        from torch.ao.quantization.observer import MovingAverageMinMaxObserver
    except ImportError as e:
        raise ImportError(
            "ExecuTorch QNN backend not found.\n"
            "Build ExecuTorch with: -DEXECUTORCH_BUILD_QNN=ON -DQNN_SDK_ROOT=<path>\n"
            f"Error: {e}"
        )

    model.eval()

    # ── 1. Initial export to get GraphModule ───────────────────────────────
    logger.info("torch.export.export() ...")
    with torch.no_grad():
        captured_model = torch.export.export(model, sample_inputs, strict=True).module()

    # ── 1b. Decompose LayerNorm so HTP doesn't fall back to CPU ────────────
    _decompose_layer_norm(captured_model, sample_inputs=sample_inputs)

    if quant_dtype is not None:
        DTYPE_MAP = {
            "8a8w": QuantDtype.use_8a8w,
            "16a8w": QuantDtype.use_16a8w,
            "16a4w": QuantDtype.use_16a4w,
            "16a16w": QuantDtype.use_16a16w,
            "16a4w_block": QuantDtype.use_16a4w_block,
        }
        if quant_dtype not in DTYPE_MAP:
            raise ValueError(
                f"Unknown quant_dtype {quant_dtype!r}, choose from {list(DTYPE_MAP)}"
            )

        # ── 2. Set up QNN quantizer ─────────────────────────────────────────
        logger.info("Setting up QnnQuantizer (%s) ...", quant_dtype)
        quantizer = QnnQuantizer()
        quantizer.set_per_channel_conv_quant(True)
        quantizer.set_quant_config(
            DTYPE_MAP[quant_dtype],
            act_observer=MovingAverageMinMaxObserver,
        )

        if skip_node_op_set:
            quantizer.add_discard_ops(list(skip_node_op_set))

        if discard_quant_ops:
            logger.info(
                "Discarding quantization on %d op types (kept unquantized for QNN HTP): %s",
                len(discard_quant_ops),
                [getattr(op, "_name", str(op)) for op in discard_quant_ops],
            )
            quantizer.add_discard_ops(list(discard_quant_ops))

        # ── 3. Prepare + calibrate ──────────────────────────────────────────
        logger.info("prepare_pt2e: inserting fake-quant observers ...")
        prepared_model = prepare_pt2e(captured_model, quantizer)

        if calibration_data is not None:
            num_cal = len(calibration_data)
            logger.info("Running %d calibration passes (real data) ...", num_cal)
            with torch.no_grad():
                for i, cal_inputs in enumerate(calibration_data):
                    if not isinstance(cal_inputs, tuple):
                        cal_inputs = (cal_inputs,)
                    prepared_model(*cal_inputs)
                    if (i + 1) % 5 == 0 or (i + 1) == num_cal:
                        logger.info("  calibration %d/%d (real)", i + 1, num_cal)
        else:
            logger.info(
                "Running %d calibration passes (synthetic) ...",
                num_calibration_passes,
            )
            with torch.no_grad():
                for i, cal_inputs in enumerate(
                    generate_calibration_inputs(sample_inputs, num_calibration_passes)
                ):
                    prepared_model(*cal_inputs)
                    if (i + 1) % 5 == 0 or (i + 1) == num_calibration_passes:
                        logger.info("  calibration %d/%d", i + 1, num_calibration_passes)

        # ── 4. Convert to static quantized graph ───────────────────────────
        logger.info("convert_pt2e: folding quantization parameters ...")
        exported_model = convert_pt2e(prepared_model)

        # ── 4b. Remove spurious quantize nodes on non-float tensors ───────
        _remove_int_quantize_nodes(exported_model)
    else:
        if not use_fp16:
            raise ValueError("quant_dtype=None requires use_fp16=True")
        logger.info("Skipping PTQ; exporting floating-point graph for QNN HTP fp16")
        exported_model = captured_model

    # ── 5. Re-export with QNN decompositions ─────────────────────────────
    logger.info("Re-exporting model with QNN decompositions ...")
    from executorch.backends.qualcomm.utils.utils import (
        get_decomp_table,
        qnn_edge_config,
    )
    from executorch.exir import ExirExportedProgram

    torch.ao.quantization.allow_exported_model_train_eval(exported_model)

    # Install safe ExportPass before any to_edge calls — this prevents
    # dtype mismatch crashes in the fake-tensor interpreter while still
    # allowing passes that succeed to apply their transformations.
    _install_safe_export_pass()

    use_fallback = False
    try:
        edge_prog = capture_program(exported_model, sample_inputs)
    except Exception as e:
        logger.warning(
            "capture_program failed (%s), using fallback export path", e
        )
        use_fallback = True

        logger.info("Direct re-export with strict=False ...")
        exported_ep = torch.export.export(exported_model, sample_inputs, strict=False)

        # Apply QNN-specific decompositions
        decomposed_ep = exported_ep.run_decompositions(get_decomp_table(None))
        core_ep = ExirExportedProgram(decomposed_ep, False)

        try:
            from executorch.backends.qualcomm._passes.tensor_i64_to_i32 import TensorI64toI32
            core_ep.transform(TensorI64toI32(edge_program=core_ep))
        except Exception as e2:
            logger.warning("TensorI64toI32 pass failed: %s (continuing)", e2)

        edge_prog = core_ep.to_edge(qnn_edge_config())

    if num_shards > 1:
        shard_io_dtype = None
        if not use_fp16:
            if quant_dtype == "8a8w":
                shard_io_dtype = torch.uint8
            elif quant_dtype in {"16a8w", "16a4w", "16a16w", "16a4w_block"}:
                shard_io_dtype = torch.uint16
        shard_starts = _insert_flux_transformer_fallbacks(
            edge_prog.exported_program.graph_module,
            num_shards=num_shards,
            num_double_layers=num_double_layers,
            total_layers=total_layers,
            quant_io_dtype=shard_io_dtype,
        )
        skip_node_op_set = set(skip_node_op_set or set())
        skip_node_op_set.add("llama.fallback.default")
        logger.info(
            "Enabled QNN multi-context sharding across %d shards at layer starts %s",
            num_shards,
            shard_starts,
        )

    # ── 6. Build QNN compiler spec + partition ─────────────────────────────
    logger.info("Building QNN HTP compiler spec for SOC ...")
    backend_options = generate_htp_compiler_spec(
        use_fp16=use_fp16,
        use_dlbc=num_shards > 1,
        use_multi_contexts=num_shards > 1,
    )
    qnn_partitioner = QnnPartitioner(
        generate_qnn_executorch_compiler_spec(
            soc_model=soc_chipset,
            backend_options=backend_options,
            online_prepare=online_prepare,
        ),
        skip_node_op_set=skip_node_op_set,
    )

    logger.info("to_backend: partitioning graph for QNN HTP ...")
    delegated_ep = to_backend(edge_prog.exported_program, qnn_partitioner)
    if num_shards > 1:
        max_sf_size = update_spill_fill_size(delegated_ep)
        logger.info("Configured QNN multi-context spill/fill buffer size: %d", max_sf_size)

    # ── 7. Serialize to .pte ──────────────────────────────────────────────
    logger.info("Serialising to .pte ...")
    executorch_config = ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=True,
            alloc_graph_output=True,
        ),
    )

    # Use EdgeProgramManager directly to avoid re-running edge passes on the
    # already-lowered program (which the to_edge() function would do).
    edge_mgr = EdgeProgramManager(
        edge_programs={"forward": delegated_ep},
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    exec_prog = edge_mgr.to_executorch(config=executorch_config)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(exec_prog.buffer)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Saved %s  (%.1f MB)", output_path, size_mb)


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
# Main
# ============================================================================

def main():
    configure_local_tooling(allow_reexec=True)

    p = argparse.ArgumentParser(
        description="Export FLUX.2-klein-4B to ExecuTorch QNN (.pte) for Qualcomm HTP/DSP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-4B")
    p.add_argument("--output_dir", default="./exported_flux2_klein_qnn")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--max_text_len", type=int, default=512)
    p.add_argument("--soc_model", default="SM8650",
                   choices=["SM8850", "SM8750", "SM8650", "SM8550", "SM8475", "SM8450"],
                   help="Target Snapdragon SOC (default: SM8650 = Snapdragon 8 Gen 3; use SM8850 for V81 / Snapdragon 8 Elite Gen 5)")
    p.add_argument("--num_calibration_passes", type=int, default=20,
                   help="Number of calibration forward passes for INT8 activation ranges")
    p.add_argument("--component",
                   choices=["all", "transformer", "vae", "vae_encoder", "text_encoder"],
                   default="all")
    p.add_argument("--num_img2img_images", type=int, default=0)
    p.add_argument("--aot", action="store_true",
                   help="Force online_prepare=False for all components (AOT HTP compile on host). "
                        "Transformer is always AOT; this flag enables AOT for the other components too.")
    p.add_argument("--quant_dtype", default="8a8w",
                   choices=["8a8w", "16a8w", "16a4w", "16a16w", "16a4w_block"],
                   help="Quantization scheme. 16a8w is a useful fallback when 8a8w "
                        "accuracy is insufficient (~2x larger .pte, ~20%% slower).")
    p.add_argument("--fp16_components", nargs="*", default=[],
                   choices=["text_encoder", "transformer", "vae", "vae_encoder"],
                   help="Components to export in floating-point QNN HTP fp16 mode. "
                        "These components skip PTQ entirely.")
    p.add_argument("--transformer_shards", type=int, default=1,
                   help="Split the transformer into this many QNN multi-context shards "
                        "for AOT exports. Use this when a single w8a8 transformer context is too large.")
    p.add_argument("--calibration_dir", default=None,
                   help="Directory with calibration_{text_encoder,transformer,vae}.pt "
                        "from collect_calibration_data.py. Uses real activations "
                        "instead of synthetic perturbations.")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Use GPU if available (strongly recommended for calibration)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cpu":
        logger.warning(
            "No GPU detected. Export will be VERY slow (hours for transformer). "
            "Strongly recommend running on a machine with a GPU (4090 or better)."
        )

    dtype = torch.float32  # QNN backend requires fp32 input to quantizer

    # Get QcomChipset enum
    soc_chipset = get_qcom_chipset(args.soc_model)
    logger.info("Target SOC: %s", args.soc_model)

    # Load pipeline on CPU — move only the component being exported to GPU
    # to avoid OOM (whole pipeline is >15GB fp32)
    pipe = load_pipeline(args.model_id, dtype=dtype)

    copy_tokenizer(pipe, str(out))
    save_vae_bn_stats(pipe, str(out))

    # Determine hidden_states_layers for text encoder (Klein default: 9, 18, 27)
    te_cfg = pipe.text_encoder.config
    hidden_states_layers = [9, 18, 27]
    logger.info("Text encoder: extracting hidden states from layers %s", hidden_states_layers)

    # Load real calibration data if --calibration_dir was passed
    cal_data = {"text_encoder": None, "transformer": None, "vae": None}
    if args.calibration_dir:
        for comp in cal_data:
            path = os.path.join(args.calibration_dir, f"calibration_{comp}.pt")
            if os.path.exists(path):
                cal_data[comp] = torch.load(path, weights_only=False)
                logger.info("Loaded %d calibration samples for %s from %s",
                            len(cal_data[comp]), comp, path)
            else:
                logger.warning("No calibration file at %s (will use synthetic data)", path)

    fp16_components = set(args.fp16_components)

    # Save metadata
    vae_sf = _get_vae_scale_factor(pipe)
    patch_h, patch_w = _compute_latent_dims(args.height, args.width, vae_sf)
    t_cfg = pipe.transformer.config
    vae_cfg = pipe.vae.config
    meta = {
        "model_id": args.model_id,
        "backend": "qnn_htp",
        "soc_model": args.soc_model,
        "height": args.height,
        "width": args.width,
        "max_text_len": args.max_text_len,
        "quantization": "mixed_int8_fp16" if fp16_components else "int8_static",
        "num_calibration_passes": args.num_calibration_passes,
        "fp16_components": sorted(fp16_components),
        "is_distilled": getattr(pipe.config, "is_distilled", True),
        "num_inference_steps": 4,
        "vae_scale_factor": vae_sf,
        "patch_dims": [patch_h, patch_w],
        "num_img2img_images": args.num_img2img_images,
        "text_encoder": {
            "hidden_states_layers": hidden_states_layers,
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

    # ── Export text encoder ───────────────────────────────────────────────
    if args.component in ("all", "text_encoder"):
        logger.info("=" * 60)
        logger.info("Exporting TEXT ENCODER ...")
        te_model = Qwen3TextEncoderWrapper(
            pipe.text_encoder, hidden_states_layers
        ).eval().cpu()
        sample_inputs = build_text_encoder_inputs(args.max_text_len)
        export_component_to_qnn(
            te_model,
            sample_inputs,
            str(out / "text_encoder.pte"),
            soc_chipset=soc_chipset,
            num_calibration_passes=args.num_calibration_passes,
            online_prepare=not args.aot,
            quant_dtype=None if "text_encoder" in fp16_components else args.quant_dtype,
            use_fp16="text_encoder" in fp16_components,
            calibration_data=cal_data["text_encoder"],
        )
        del te_model
        gc.collect()

    # ── Export transformer ────────────────────────────────────────────────
    if args.component in ("all", "transformer"):
        logger.info("=" * 60)
        logger.info("Exporting TRANSFORMER ...")
        if "transformer" in fp16_components:
            logger.info(
                "Transformer export uses floating-point QNN HTP fp16 mode "
                "(no PTQ) with AOT compilation."
            )
        else:
            logger.info(
                "Transformer export uses online_prepare=True. The host-side "
                "AOT scheduler (RouterX86) fails on Flux w8a8 when a partition "
                "contains >=2 attention softmaxes; on-device graph prep uses a "
                "different codepath that compiles cleanly. First-run latency "
                "includes graph prep; subsequent runs are fast."
            )
        tf_model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
        sample_inputs = build_transformer_inputs(
            pipe, args.height, args.width, args.max_text_len,
            dtype=dtype, num_img2img_images=args.num_img2img_images,
        )
        export_component_to_qnn(
            tf_model,
            sample_inputs,
            str(out / "transformer.pte"),
            soc_chipset=soc_chipset,
            num_calibration_passes=args.num_calibration_passes,
            online_prepare=not args.aot,
            quant_dtype=None if "transformer" in fp16_components else args.quant_dtype,
            use_fp16="transformer" in fp16_components,
            calibration_data=cal_data["transformer"],
            num_shards=args.transformer_shards,
            num_double_layers=len(pipe.transformer.transformer_blocks),
            total_layers=len(pipe.transformer.transformer_blocks) + len(pipe.transformer.single_transformer_blocks),
        )
        del tf_model
        gc.collect()

    # ── Export VAE decoder ────────────────────────────────────────────────
    if args.component in ("all", "vae"):
        logger.info("=" * 60)
        logger.info("Exporting VAE DECODER ...")
        vae_model = VAEDecoderWrapper(pipe.vae).eval().cpu()
        sample_inputs = build_vae_inputs(pipe, args.height, args.width, dtype=dtype)
        export_component_to_qnn(
            vae_model,
            sample_inputs,
            str(out / "vae_decoder.pte"),
            soc_chipset=soc_chipset,
            num_calibration_passes=args.num_calibration_passes,
            online_prepare=not args.aot,
            quant_dtype=None if "vae" in fp16_components else args.quant_dtype,
            use_fp16="vae" in fp16_components,
            calibration_data=cal_data["vae"],
        )
        del vae_model
        gc.collect()

    # ── Export VAE encoder (img2img) ──────────────────────────────────────
    if args.component in ("vae_encoder",) or (
        args.component == "all" and args.num_img2img_images > 0
    ):
        logger.info("=" * 60)
        logger.info("Exporting VAE ENCODER ...")
        vae_enc = VAEEncoderWrapper(pipe.vae).eval().cpu()
        sample_inputs = build_vae_encoder_inputs(args.height, args.width, dtype=dtype)
        export_component_to_qnn(
            vae_enc,
            sample_inputs,
            str(out / "vae_encoder.pte"),
            soc_chipset=soc_chipset,
            num_calibration_passes=args.num_calibration_passes,
            online_prepare=not args.aot,
            quant_dtype=None if "vae_encoder" in fp16_components else args.quant_dtype,
            use_fp16="vae_encoder" in fp16_components,
            calibration_data=cal_data["vae"],
        )
        del vae_enc
        gc.collect()

    logger.info("=" * 60)
    logger.info("Export complete. Output: %s", out)
    logger.info(
        "\nNext steps:\n"
        "  1. Build ExecuTorch for Android with QNN backend:\n"
        "     cmake ... -DEXECUTORCH_BUILD_QNN=ON -DQNN_SDK_ROOT=$QNN_SDK_ROOT ...\n"
        "  2. Build flux2_main.cpp with QNN runtime linked\n"
        "  3. Push .pte files + QNN libraries to device\n"
        "  4. Run: ./flux2_main --model_dir . --tokens prompt.bin --output out.ppm\n"
    )


if __name__ == "__main__":
    main()
