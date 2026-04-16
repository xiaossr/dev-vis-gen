#!/usr/bin/env python3
"""
Export FLUX.2-klein-4B to ExecuTorch QNN/HTP (.pte) for on-device Hexagon NPU inference.

This script mirrors ``export_flux2_klein_xnnpack.py`` but targets the Qualcomm
QNN backend (Hexagon Tensor Processor) instead of XNNPACK CPU.  The model
wrappers, pipeline loading, and input builders are reused from the XNNPACK
export script.

Pipeline components:
  - Text encoder : Qwen3ForCausalLM         -> text_encoder.pte
  - Transformer  : Flux2Transformer2DModel   -> transformer.pte
  - VAE decoder  : AutoencoderKLFlux2        -> vae_decoder.pte
  - VAE encoder  : AutoencoderKLFlux2        -> vae_encoder.pte  (for img2img)

QNN quantization
----------------
Unlike XNNPACK (which uses TorchAO source transforms), QNN uses its own
PT2E quantization flow via ``QnnQuantizer``.  The typical flow is:

  1. ``torch.export`` the model
  2. ``prepare_pt2e(model, QnnQuantizer(...))``
  3. Run calibration forward passes
  4. ``convert_pt2e(model)``
  5. Lower to QNN via ``to_edge_transform_and_lower_to_qnn()``

For non-quantized models, QNN HTP can run in fp16 mode.

Requirements
------------
    pip install -r requirements_export.txt
    export QNN_SDK_ROOT=/path/to/qnn-sdk

Usage
-----
    # Export all components in fp16 (no quantization):
    python export_flux2_klein_qnn.py --output_dir ./exported_flux2_klein_qnn --soc SM8650

    # Export with int8 quantization (PTQ):
    python export_flux2_klein_qnn.py --output_dir ./exported_flux2_klein_qnn --soc SM8650 --quantize

    # Export with 16a4w quantization:
    python export_flux2_klein_qnn.py --output_dir ./exported_flux2_klein_qnn --soc SM8650 \\
        --quantize --quant_dtype 16a4w

    # Export only one component:
    python export_flux2_klein_qnn.py --component transformer --soc SM8650 --quantize
"""

import argparse
import gc
import json
import logging
import os
from pathlib import Path

import operator

import torch
import torch.nn as nn

# Reuse wrappers and helpers from XNNPACK export script
from export_flux2_klein_xnnpack import (
    Qwen3TextEncoderWrapper,
    Flux2TransformerWrapper,
    VAEDecoderWrapper,
    VAEEncoderWrapper,
    load_pipeline,
    copy_tokenizer,
    save_vae_bn_stats,
    build_text_encoder_inputs,
    build_transformer_inputs,
    build_vae_inputs,
    build_vae_encoder_inputs,
    _compute_latent_dims,
    _get_vae_scale_factor,
    _free_memory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("flux2_export_qnn")


def _decompose_layer_norm(model):
    """Decompose native_layer_norm into primitive ops in the FX graph.

    Replaces native_layer_norm(input, shape, weight, bias, eps) with:
        mean = input.mean(-1, keepdim=True)
        diff = input - mean
        var = (diff * diff).mean(-1, keepdim=True)
        rstd = 1 / sqrt(var + eps)
        normalized = diff * rstd
        output = normalized * weight + bias  (if weight/bias exist)

    All resulting ops (mean, sub, mul, add, rsqrt) are supported by QNN HTP.
    """
    if isinstance(model, torch.fx.GraphModule):
        gm = model
    elif hasattr(model, 'graph'):
        gm = model
    else:
        return model

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

        # Number of dims to normalize over
        num_norm_dims = len(normalized_shape)
        dims = list(range(-num_norm_dims, 0))

        with gm.graph.inserting_before(node):
            # Decompose layer_norm using aten.var/mean directly.
            # Avoids the sub→mul→mean→rsqrt pattern that RecomposeRmsNorm
            # would incorrectly match and recompose with wrong shapes.

            # mean = input.mean(dims, keepdim=True)
            mean_node = gm.graph.call_function(
                torch.ops.aten.mean.dim, args=(input_node, dims, True)
            )
            # var = input.var(dims, correction=0, keepdim=True)
            var_node = gm.graph.call_function(
                torch.ops.aten.var.correction,
                args=(input_node, dims),
                kwargs={"correction": 0, "keepdim": True},
            )
            # diff = input - mean
            diff_node = gm.graph.call_function(
                torch.ops.aten.sub.Tensor, args=(input_node, mean_node)
            )
            # inv_std = rsqrt(var + eps)
            eps_node = gm.graph.call_function(
                torch.ops.aten.add.Scalar, args=(var_node, eps)
            )
            rsqrt_node = gm.graph.call_function(
                torch.ops.aten.rsqrt.default, args=(eps_node,)
            )
            # normalized = diff * inv_std
            norm_node = gm.graph.call_function(
                torch.ops.aten.mul.Tensor, args=(diff_node, rsqrt_node)
            )
            # Apply weight and bias if present
            out_node = norm_node
            if weight_node is not None and not (
                isinstance(weight_node, type(None))
            ):
                out_node = gm.graph.call_function(
                    torch.ops.aten.mul.Tensor, args=(out_node, weight_node)
                )
            if bias_node is not None and not (
                isinstance(bias_node, type(None))
            ):
                out_node = gm.graph.call_function(
                    torch.ops.aten.add.Tensor, args=(out_node, bias_node)
                )

        # native_layer_norm returns (output, mean, rstd) — replace all uses
        if node.target == torch.ops.aten.native_layer_norm.default:
            # Find getitem users for the tuple outputs
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
            # layer_norm returns just the output
            node.replace_all_uses_with(out_node)

        gm.graph.erase_node(node)

    if changed:
        gm.graph.lint()
        gm.recompile()
        logger.info("Decomposed native_layer_norm into primitive ops")

    return gm


# ============================================================================
# 1.  QNN export routine
# ============================================================================

def export_component_to_qnn(
    model: nn.Module,
    sample_inputs: tuple,
    output_path: str,
    soc_model: str = "SM8650",
    quantize: bool = False,
    quant_dtype: str = "8a8w",
    num_calibration: int = 10,
    calibration_data: list = None,
):
    """torch.export -> QNN/HTP partitioning -> ExecuTorch serialisation.

    Parameters
    ----------
    model : nn.Module
        The wrapped model to export.
    sample_inputs : tuple
        Example inputs for tracing.
    output_path : str
        Where to write the .pte file.
    soc_model : str
        Target Qualcomm SoC (e.g. "SM8650", "SM8750").
    quantize : bool
        If True, apply QNN PTQ quantization via QnnQuantizer.
    quant_dtype : str
        Quantization scheme: "8a8w", "16a8w", "16a4w", "16a16w".
    num_calibration : int
        Number of calibration forward passes for PTQ (used only when
        calibration_data is None).
    calibration_data : list, optional
        List of real calibration input tuples from collect_calibration_data.py.
        When provided, these are used instead of random calibration data.
    """
    from torch.export import export
    from executorch.backends.qualcomm.quantizer.quantizer import (
        QnnQuantizer,
        QuantDtype,
    )
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QnnExecuTorchBackendType,
    )
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        get_soc_to_chipset_map,
        to_edge_transform_and_lower_to_qnn,
    )
    from torchao.quantization.pt2e.quantize_pt2e import (
        convert_pt2e,
        prepare_pt2e,
    )

    model.eval()

    # Map string quant_dtype to QnnQuantizer QuantDtype
    dtype_map = {
        "8a8w": QuantDtype.use_8a8w,
        "16a8w": QuantDtype.use_16a8w,
        "16a4w": QuantDtype.use_16a4w,
        "16a16w": QuantDtype.use_16a16w,
        "8a4w": QuantDtype.use_8a4w,
    }

    chipset_map = get_soc_to_chipset_map()
    if soc_model not in chipset_map:
        available = list(chipset_map.keys())
        raise ValueError(
            f"Unknown SoC '{soc_model}'. Available: {available}"
        )
    chipset = chipset_map[soc_model]

    if quantize:
        # ---- QNN PTQ quantization -----------------------------------------
        logger.info("Applying QNN PTQ quantization (%s) ...", quant_dtype)

        qnn_quant_dtype = dtype_map.get(quant_dtype)
        if qnn_quant_dtype is None:
            raise ValueError(
                f"Unknown quant_dtype '{quant_dtype}'. "
                f"Available: {list(dtype_map.keys())}"
            )

        # Step 1: Export to get traced module
        logger.info("Pre-export for quantization ...")
        pre_ep = export(model, sample_inputs)
        m = pre_ep.module()

        # Step 2: Prepare with QnnQuantizer
        quantizer = QnnQuantizer(
            backend=QnnExecuTorchBackendType.kHtpBackend,
            soc_model=chipset,
        )
        quantizer.set_default_quant_config(qnn_quant_dtype)

        m = prepare_pt2e(m, quantizer)

        # Step 3: Calibration
        if calibration_data is not None:
            num_cal = len(calibration_data)
            logger.info("Running %d calibration passes (real data) ...", num_cal)
            with torch.no_grad():
                for cal_i, cal_inputs in enumerate(calibration_data):
                    if not isinstance(cal_inputs, tuple):
                        cal_inputs = (cal_inputs,)
                    m(*cal_inputs)
                    logger.info("  calibration %d/%d (real)", cal_i + 1, num_cal)
        else:
            logger.info("Running %d calibration passes (random data) ...", num_calibration)
            with torch.no_grad():
                for cal_i in range(num_calibration):
                    cal_inputs = []
                    for inp in sample_inputs:
                        if inp.is_floating_point():
                            if inp.ndim == 1 and inp.shape[0] == 1:
                                cal_inputs.append(
                                    torch.full_like(
                                        inp,
                                        (cal_i + 1) / (num_calibration + 1),
                                    )
                                )
                            else:
                                cal_inputs.append(torch.randn_like(inp))
                        else:
                            cal_inputs.append(inp)
                    m(*cal_inputs)
                    logger.info("  calibration %d/%d (random)", cal_i + 1, num_calibration)

        # Step 4: Convert
        m = convert_pt2e(m)
        logger.info("QNN quantization complete.")
    else:
        m = model

    # ---- Generate HTP compiler specs ----------------------------------
    use_fp16 = not quantize
    logger.info(
        "Generating HTP compiler spec (fp16=%s, soc=%s) ...",
        use_fp16, soc_model,
    )
    backend_options = generate_htp_compiler_spec(use_fp16=use_fp16)
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=chipset,
        backend_options=backend_options,
    )

    # ---- Lower to QNN -------------------------------------------------
    logger.info("Lowering to QNN backend ...")
    delegated_program = to_edge_transform_and_lower_to_qnn(
        m,
        sample_inputs,
        compiler_specs,
    )

    # ---- Serialise to .pte --------------------------------------------
    logger.info("Serialising to .pte ...")
    et_program = delegated_program.to_executorch()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Saved %s  (%.1f MB)", output_path, size_mb)


# ============================================================================
# 2.  Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Export FLUX.2-klein-4B to ExecuTorch QNN/HTP (.pte)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-4B",
                    help="HuggingFace model ID or local path")
    p.add_argument("--output_dir", default="./exported_flux2_klein_qnn",
                    help="Directory for exported artefacts")
    p.add_argument("--height", type=int, default=512,
                    help="Target image height")
    p.add_argument("--width", type=int, default=512,
                    help="Target image width")
    p.add_argument("--max_text_len", type=int, default=512,
                    help="Max text-token sequence length")
    p.add_argument("--soc", type=str, required=True,
                    help="Target Qualcomm SoC model (e.g. SM8650, SM8750)")
    p.add_argument("--quantize", action="store_true",
                    help="Apply QNN PTQ quantization via QnnQuantizer")
    p.add_argument("--quant_dtype", type=str, default="8a8w",
                    choices=["8a8w", "16a8w", "16a4w", "16a16w", "8a4w"],
                    help="Quantization dtype scheme (default: 8a8w)")
    p.add_argument("--num_calibration", type=int, default=10,
                    help="Number of calibration forward passes for PTQ")
    p.add_argument("--component",
                    choices=["all", "transformer", "vae", "vae_encoder",
                             "text_encoder"],
                    default="all",
                    help="Which component(s) to export")
    p.add_argument("--num_img2img_images", type=int, default=0,
                    help="Number of reference images for img2img")
    p.add_argument("--calibration_dir", type=str, default=None,
                    help="Directory with real calibration data from "
                         "collect_calibration_data.py. If provided, uses "
                         "real data instead of random for PTQ calibration.")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32

    # ---- load pipeline -------------------------------------------------
    pipe = load_pipeline(args.model_id, dtype=dtype)
    is_distilled = getattr(pipe.config, "is_distilled", True)

    # ---- save tokenizer & BN stats ------------------------------------
    copy_tokenizer(pipe, str(out))
    save_vae_bn_stats(pipe, str(out))

    # ---- determine text encoder hidden_states_layers -------------------
    te_cfg = pipe.text_encoder.config
    num_te_layers = getattr(te_cfg, "num_hidden_layers", 28)
    hidden_states_layers = [9, 18, 27]
    logger.info("Text encoder: %d layers, extracting from %s",
                num_te_layers, hidden_states_layers)

    # ---- load calibration data if available -----------------------------
    cal_data = {"text_encoder": None, "transformer": None, "vae": None}
    if args.calibration_dir and args.quantize:
        for comp in cal_data:
            path = os.path.join(args.calibration_dir, f"calibration_{comp}.pt")
            if os.path.exists(path):
                cal_data[comp] = torch.load(path, weights_only=False)
                logger.info("Loaded %d calibration samples for %s from %s",
                            len(cal_data[comp]), comp, path)
            else:
                logger.warning("No calibration file found at %s — will use random data", path)

    # ---- save export metadata ------------------------------------------
    t_cfg = pipe.transformer.config
    vae_cfg = pipe.vae.config
    vae_sf = _get_vae_scale_factor(pipe)
    patch_h, patch_w = _compute_latent_dims(args.height, args.width, vae_sf)

    quant_mode = args.quant_dtype if args.quantize else "none"
    meta = {
        "model_id": args.model_id,
        "backend": "qnn_htp",
        "soc": args.soc,
        "height": args.height,
        "width": args.width,
        "max_text_len": args.max_text_len,
        "quantized": args.quantize,
        "quant_dtype": quant_mode,
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

    # ---- export text encoder -------------------------------------------
    if args.component in ("all", "text_encoder"):
        logger.info("=" * 60)
        logger.info("EXPORTING TEXT ENCODER (Qwen3) -> QNN/HTP")
        logger.info("=" * 60)

        wrapper = Qwen3TextEncoderWrapper(
            pipe.text_encoder, hidden_states_layers=hidden_states_layers,
        ).eval()
        inputs = build_text_encoder_inputs(args.max_text_len)

        logger.info("Sanity-checking forward pass ...")
        with torch.no_grad():
            test_out = wrapper(*inputs)
        logger.info("  output shape: %s", test_out.shape)
        del test_out
        _free_memory()

        export_component_to_qnn(
            wrapper, inputs, str(out / "text_encoder.pte"),
            soc_model=args.soc,
            quantize=args.quantize,
            quant_dtype=args.quant_dtype,
            num_calibration=args.num_calibration,
            calibration_data=cal_data["text_encoder"],
        )
        del wrapper, inputs
        _free_memory()

    # Free text encoder before heavy exports
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        del pipe.text_encoder
        pipe.text_encoder = None
        _free_memory()
        logger.info("Freed text encoder to reduce memory for export.")

    # ---- export transformer --------------------------------------------
    if args.component in ("all", "transformer"):
        logger.info("=" * 60)
        logger.info("EXPORTING TRANSFORMER (text-to-image) -> QNN/HTP")
        logger.info("=" * 60)

        wrapper = Flux2TransformerWrapper(pipe.transformer).eval()
        t2i_inputs = build_transformer_inputs(
            pipe, args.height, args.width, args.max_text_len, dtype,
            num_img2img_images=0,
        )

        logger.info("Sanity-checking forward pass ...")
        with torch.no_grad():
            test_out = wrapper(*t2i_inputs)
        logger.info("  output shape: %s", test_out.shape)
        del test_out
        _free_memory()

        export_component_to_qnn(
            wrapper, t2i_inputs, str(out / "transformer.pte"),
            soc_model=args.soc,
            quantize=args.quantize,
            quant_dtype=args.quant_dtype,
            num_calibration=args.num_calibration,
            calibration_data=cal_data["transformer"],
        )
        del t2i_inputs
        _free_memory()

        # img2img variant
        if args.num_img2img_images > 0:
            logger.info("=" * 60)
            logger.info("EXPORTING TRANSFORMER (img2img, %d ref image(s)) -> QNN/HTP",
                        args.num_img2img_images)
            logger.info("=" * 60)

            img2img_inputs = build_transformer_inputs(
                pipe, args.height, args.width, args.max_text_len, dtype,
                num_img2img_images=args.num_img2img_images,
            )
            export_component_to_qnn(
                wrapper, img2img_inputs,
                str(out / "transformer_img2img.pte"),
                soc_model=args.soc,
                quantize=args.quantize,
                quant_dtype=args.quant_dtype,
                num_calibration=args.num_calibration,
                calibration_data=cal_data["transformer"],
            )
            del img2img_inputs
            _free_memory()

        del wrapper
        _free_memory()

    # ---- export VAE decoder --------------------------------------------
    if args.component in ("all", "vae"):
        logger.info("=" * 60)
        logger.info("EXPORTING VAE DECODER -> QNN/HTP")
        logger.info("=" * 60)

        wrapper = VAEDecoderWrapper(pipe.vae).eval()
        inputs = build_vae_inputs(pipe, args.height, args.width, dtype)

        logger.info("Sanity-checking forward pass ...")
        with torch.no_grad():
            test_out = wrapper(*inputs)
        logger.info("  output shape: %s", test_out.shape)
        del test_out
        _free_memory()

        export_component_to_qnn(
            wrapper, inputs, str(out / "vae_decoder.pte"),
            soc_model=args.soc,
            quantize=args.quantize,
            quant_dtype=args.quant_dtype,
            num_calibration=args.num_calibration,
            calibration_data=cal_data["vae"],
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
        logger.info("EXPORTING VAE ENCODER -> QNN/HTP")
        logger.info("=" * 60)

        wrapper = VAEEncoderWrapper(pipe.vae).eval()
        inputs = build_vae_encoder_inputs(args.height, args.width, dtype)

        logger.info("Sanity-checking forward pass ...")
        with torch.no_grad():
            test_out = wrapper(*inputs)
        logger.info("  output shape: %s", test_out.shape)
        del test_out
        _free_memory()

        export_component_to_qnn(
            wrapper, inputs, str(out / "vae_encoder.pte"),
            soc_model=args.soc,
            quantize=args.quantize,
            quant_dtype=args.quant_dtype,
            num_calibration=args.num_calibration,
            calibration_data=cal_data["vae"],
        )
        del wrapper, inputs
        _free_memory()

    # ---- summary -------------------------------------------------------
    del pipe
    _free_memory()

    banner = "\n" + "=" * 60 + "\n  EXPORT COMPLETE (QNN/HTP)\n" + "=" * 60
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
