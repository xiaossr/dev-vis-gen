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
        from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
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
    """Wraps Qwen3ForCausalLM to extract multi-layer hidden states."""

    def __init__(self, text_encoder, hidden_states_layers=(9, 18, 27)):
        super().__init__()
        self.text_encoder = text_encoder
        self.hidden_states_layers = list(hidden_states_layers)

    def forward(self, input_ids, attention_mask):
        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
    pipe.eval()
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

    Steps:
      1. torch.export()            (traces the graph)
      2. QnnQuantizer + prepare_pt2e   (insert fake-quant observers)
      3. Calibration forward passes    (determine activation ranges)
      4. convert_pt2e                  (fold scales, replace ops with INT8)
      5. QnnPartitioner                (annotate graph for HTP dispatch)
      6. to_edge_transform_and_lower() (compile to ExecuTorch IR)
      7. to_executorch() + serialize   (write .pte)
    """
    from torch.export import export
    from executorch.exir import to_edge_transform_and_lower

    try:
        from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer
        from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
        from executorch.backends.qualcomm.utils.utils import (
            canonicalize_program,
            generate_htp_compiler_spec,
            generate_qnn_executorch_compiler_spec,
        )
        from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    except ImportError as e:
        raise ImportError(
            "ExecuTorch QNN backend or torchao not found.\n"
            "Build ExecuTorch with: -DEXECUTORCH_BUILD_QNN=ON -DQNN_SDK_ROOT=<path>\n"
            f"Error: {e}"
        )

    model.eval()

    # ── 1. Initial export to get the graph ─────────────────────────────────
    logger.info("torch.export.export() ...")
    with torch.no_grad():
        exported_program = export(model, sample_inputs)

    exported_module = exported_program.module()

    # ── 2. Set up QNN quantizer ─────────────────────────────────────────────
    logger.info("Setting up QnnQuantizer for INT8 static quantization ...")
    quantizer = QnnQuantizer()
    # Use INT8 for both weights and activations (HTP optimal)
    quantizer.set_bit8_op_str_override("ON")

    if skip_node_op_set:
        # Allow skipping specific ops that cause issues (e.g. softmax in attn)
        quantizer.set_skip_ops(skip_node_op_set)

    prepared_model = prepare_pt2e(exported_module, quantizer)

    # ── 3. Calibration ──────────────────────────────────────────────────────
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
    quantized_module = convert_pt2e(prepared_model)

    # ── 5. Build QNN compiler spec ─────────────────────────────────────────
    logger.info("Building QNN HTP compiler spec for SOC ...")
    htp_options = generate_htp_compiler_spec(
        use_fp16=False,          # Force INT8 path (not FP16)
    )
    compiler_spec = generate_qnn_executorch_compiler_spec(
        soc_model=soc_chipset,
        backend_options=htp_options,
        is_from_context_binary=False,
        debug=False,
    )

    # ── 6. Re-export quantized model and partition ─────────────────────────
    logger.info("Re-exporting quantized model ...")
    quantized_exported = export(quantized_module, sample_inputs)

    logger.info("Applying QnnPartitioner ...")
    edge_program = to_edge_transform_and_lower(
        quantized_exported,
        partitioner=[QnnPartitioner(compiler_specs=compiler_spec)],
    )

    # ── 7. Canonicalize and serialize ──────────────────────────────────────
    logger.info("Serialising to .pte ...")
    et_program = edge_program.to_executorch()
    canonicalize_program(et_program)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

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

    # Load pipeline
    pipe = load_pipeline(args.model_id, dtype=dtype)
    if device.type == "cuda":
        # Load to GPU for faster calibration, then move back for export
        pipe = pipe.to(device)

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
        ).eval()
        if device.type == "cuda":
            te_model = te_model.to(device)
        sample_inputs = build_text_encoder_inputs(args.max_text_len)
        if device.type == "cuda":
            sample_inputs = tuple(x.to(device) for x in sample_inputs)
        export_component_to_qnn(
            te_model,
            sample_inputs,
            str(out / "text_encoder.pte"),
            soc_chipset=soc_chipset,
            num_calibration_passes=args.num_calibration_passes,
        )
        del te_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Export transformer ────────────────────────────────────────────────
    if args.component in ("all", "transformer"):
        logger.info("=" * 60)
        logger.info("Exporting TRANSFORMER ...")
        tf_model = Flux2TransformerWrapper(pipe.transformer).eval()
        if device.type == "cuda":
            tf_model = tf_model.to(device)
        sample_inputs = build_transformer_inputs(
            pipe, args.height, args.width, args.max_text_len,
            dtype=dtype, num_img2img_images=args.num_img2img_images,
        )
        if device.type == "cuda":
            sample_inputs = tuple(x.to(device) for x in sample_inputs)
        export_component_to_qnn(
            tf_model,
            sample_inputs,
            str(out / "transformer.pte"),
            soc_chipset=soc_chipset,
            num_calibration_passes=args.num_calibration_passes,
        )
        del tf_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Export VAE decoder ────────────────────────────────────────────────
    if args.component in ("all", "vae"):
        logger.info("=" * 60)
        logger.info("Exporting VAE DECODER ...")
        vae_model = VAEDecoderWrapper(pipe.vae).eval()
        if device.type == "cuda":
            vae_model = vae_model.to(device)
        sample_inputs = build_vae_inputs(pipe, args.height, args.width, dtype=dtype)
        if device.type == "cuda":
            sample_inputs = tuple(x.to(device) for x in sample_inputs)
        export_component_to_qnn(
            vae_model,
            sample_inputs,
            str(out / "vae_decoder.pte"),
            soc_chipset=soc_chipset,
            num_calibration_passes=args.num_calibration_passes,
        )
        del vae_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Export VAE encoder (img2img) ──────────────────────────────────────
    if args.component in ("vae_encoder",) or (
        args.component == "all" and args.num_img2img_images > 0
    ):
        logger.info("=" * 60)
        logger.info("Exporting VAE ENCODER ...")
        vae_enc = VAEEncoderWrapper(pipe.vae).eval()
        if device.type == "cuda":
            vae_enc = vae_enc.to(device)
        sample_inputs = build_vae_encoder_inputs(args.height, args.width, dtype=dtype)
        if device.type == "cuda":
            sample_inputs = tuple(x.to(device) for x in sample_inputs)
        export_component_to_qnn(
            vae_enc,
            sample_inputs,
            str(out / "vae_encoder.pte"),
            soc_chipset=soc_chipset,
            num_calibration_passes=args.num_calibration_passes,
        )
        del vae_enc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
