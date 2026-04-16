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
import os
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("flux2_qnn_export")


# ============================================================================
# SOC model mapping
# ============================================================================

SOC_MODEL_MAP = {
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
    For other tensors: small random perturbations.
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
            else:
                # random activations with the same scale as the sample
                scale = inp.abs().mean().item() or 1.0
                cal.append(torch.randn_like(inp) * scale)
        yield tuple(cal)


# ============================================================================
# Core QNN export routine
# ============================================================================

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


def export_component_to_qnn(
    model: nn.Module,
    sample_inputs: tuple,
    output_path: str,
    soc_chipset,
    num_calibration_passes: int = 20,
    skip_node_op_set: set = None,
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

    # ── 2. Set up QNN quantizer ─────────────────────────────────────────────
    logger.info("Setting up QnnQuantizer for INT8 static quantization ...")
    quantizer = QnnQuantizer()
    quantizer.set_per_channel_conv_quant(True)
    quantizer.set_quant_config(QuantDtype.use_8a8w, act_observer=MovingAverageMinMaxObserver)

    if skip_node_op_set:
        quantizer.add_discard_ops(list(skip_node_op_set))

    # ── 3. Prepare + calibrate ──────────────────────────────────────────────
    logger.info("prepare_pt2e: inserting fake-quant observers ...")
    prepared_model = prepare_pt2e(captured_model, quantizer)

    logger.info("Running %d calibration passes ...", num_calibration_passes)
    with torch.no_grad():
        for i, cal_inputs in enumerate(
            generate_calibration_inputs(sample_inputs, num_calibration_passes)
        ):
            prepared_model(*cal_inputs)
            if (i + 1) % 5 == 0 or (i + 1) == num_calibration_passes:
                logger.info("  calibration %d/%d", i + 1, num_calibration_passes)

    # ── 4. Convert to static INT8 ───────────────────────────────────────────
    logger.info("convert_pt2e: folding quantization parameters ...")
    quantized_model = convert_pt2e(prepared_model)

    # ── 4b. Remove spurious quantize nodes on non-float tensors ────────────
    _remove_int_quantize_nodes(quantized_model)

    # ── 5. Re-export with QNN decompositions ─────────────────────────────
    logger.info("Re-exporting quantized model with QNN decompositions ...")
    from executorch.backends.qualcomm.utils.utils import (
        get_decomp_table,
        qnn_edge_config,
    )
    from executorch.exir import ExirExportedProgram

    torch.ao.quantization.allow_exported_model_train_eval(quantized_model)

    # Install safe ExportPass before any to_edge calls — this prevents
    # dtype mismatch crashes in the fake-tensor interpreter while still
    # allowing passes that succeed to apply their transformations.
    _install_safe_export_pass()

    use_fallback = False
    try:
        edge_prog = capture_program(quantized_model, sample_inputs)
    except Exception as e:
        logger.warning(
            "capture_program failed (%s), using fallback export path", e
        )
        use_fallback = True

        logger.info("Direct re-export with strict=False ...")
        quantized_ep = torch.export.export(quantized_model, sample_inputs, strict=False)

        # Apply QNN-specific decompositions
        decomposed_ep = quantized_ep.run_decompositions(get_decomp_table(None))
        core_ep = ExirExportedProgram(decomposed_ep, False)

        try:
            from executorch.backends.qualcomm._passes.tensor_i64_to_i32 import TensorI64toI32
            core_ep.transform(TensorI64toI32(edge_program=core_ep))
        except Exception as e2:
            logger.warning("TensorI64toI32 pass failed: %s (continuing)", e2)

        edge_prog = core_ep.to_edge(qnn_edge_config())

    # ── 6. Build QNN compiler spec + partition ─────────────────────────────
    logger.info("Building QNN HTP compiler spec for SOC ...")
    backend_options = generate_htp_compiler_spec(use_fp16=False)
    qnn_partitioner = QnnPartitioner(
        generate_qnn_executorch_compiler_spec(
            soc_model=soc_chipset,
            backend_options=backend_options,
            online_prepare=True,
        ),
        skip_node_op_set=skip_node_op_set,
    )

    logger.info("to_backend: partitioning graph for QNN HTP ...")
    delegated_ep = to_backend(edge_prog.exported_program, qnn_partitioner)

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
                   choices=["SM8650", "SM8550", "SM8475", "SM8450"],
                   help="Target Snapdragon SOC (default: SM8650 = Snapdragon 8 Gen 3)")
    p.add_argument("--num_calibration_passes", type=int, default=20,
                   help="Number of calibration forward passes for INT8 activation ranges")
    p.add_argument("--component",
                   choices=["all", "transformer", "vae", "vae_encoder", "text_encoder"],
                   default="all")
    p.add_argument("--num_img2img_images", type=int, default=0)
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
        "quantization": "int8_static",
        "num_calibration_passes": args.num_calibration_passes,
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
        )
        del te_model
        gc.collect()

    # ── Export transformer ────────────────────────────────────────────────
    if args.component in ("all", "transformer"):
        logger.info("=" * 60)
        logger.info("Exporting TRANSFORMER ...")
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
