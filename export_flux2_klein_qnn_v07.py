#!/usr/bin/env python3
"""
Export FLUX.2-klein-4B to ExecuTorch QNN/HTP (.pte) using the v0.7.0 API.

Faithful port of April's export_flux2_klein_qnn.py (commit 342a6cd),
retargeted for SM8850 / Hexagon V81 (Snapdragon 8 Elite Gen 5). Uses the
high-level to_edge_transform_and_lower_to_qnn helper introduced in
ExecuTorch 0.7.0 — the low-level capture_program path we used on v0.6.0 is
deprecated in 0.7.0 and is what produced the RouterX86 / libQnnHtp.so
crashes we kept hitting.

Run with:
    .venv-et07/bin/python export_flux2_klein_qnn_v07.py --soc SM8850 ...

Not to be confused with export_flux2_klein_qnn.py which is the v0.6.0 path.
"""

import argparse
import gc
import json
import logging
import operator
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("flux2_export_qnn_v07")


# Ensure `flatc` is on PATH (same issue the xnnpack path fixed).
def _ensure_flatc_on_path() -> None:
    if os.environ.get("FLATC_EXECUTABLE"):
        return
    # Prefer the native flatc binary from the sibling .venv (installed
    # executorch 0.6.0 ships the compiled x86 flatc in data/bin/flatc).
    repo_root = Path(__file__).resolve().parent
    native_flatc = (
        repo_root / ".venv" / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages" / "executorch" / "data" / "bin" / "flatc"
    )
    if native_flatc.is_file() and os.access(native_flatc, os.X_OK):
        os.environ["FLATC_EXECUTABLE"] = str(native_flatc)
        logger.info("FLATC_EXECUTABLE=%s", native_flatc)
        return
    # Fallback.
    import shutil
    found = shutil.which("flatc")
    if found:
        os.environ["FLATC_EXECUTABLE"] = found


_ensure_flatc_on_path()


def _ensure_qnn_libs_on_ldpath() -> None:
    """QNN host libs must be on LD_LIBRARY_PATH so libQnnSystem.so etc. load."""
    repo_root = Path(__file__).resolve().parent
    qnn_root_env = os.environ.get("QNN_SDK_ROOT")
    qnn_root = Path(qnn_root_env) if qnn_root_env else repo_root / "qairt" / "2.45.0.260326"
    if qnn_root.exists():
        os.environ["QNN_SDK_ROOT"] = str(qnn_root)

    need_paths = []
    x86_lib = qnn_root / "lib" / "x86_64-linux-clang"
    if x86_lib.exists():
        need_paths.append(str(x86_lib))

    # Local C++ runtime (same logic as the v0.6.0 script).
    for cand in (
        repo_root / ".local-libs" / "usr" / "lib" / "x86_64-linux-gnu",
        repo_root / ".local-libs-jammy" / "extracted" / "usr" / "lib" / "x86_64-linux-gnu",
        repo_root / ".local-libs-14" / "usr" / "lib" / "x86_64-linux-gnu",
    ):
        if (cand / "libc++.so.1").exists():
            need_paths.append(str(cand))
            break

    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    missing = [p for p in need_paths if p not in current]
    if missing and os.environ.get("_QNN_V07_REEXEC") != "1":
        os.environ["LD_LIBRARY_PATH"] = ":".join(missing + current)
        os.environ["_QNN_V07_REEXEC"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


_ensure_qnn_libs_on_ldpath()


# Reuse the model wrappers and helpers from our XNNPACK export script.
# These wrappers are pure PyTorch and version-agnostic.
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


# ---------------------------------------------------------------------------
# LayerNorm decomposition (same logic April used)
# ---------------------------------------------------------------------------


def _decompose_layer_norm(model):
    """Replace native_layer_norm with primitive ops.

    Stock QNN HTP rejects rank-3 native_layer_norm; this rewrite keeps the
    normalisation on-DSP by expressing it in terms of mean/var/rsqrt/mul/add.
    """
    if isinstance(model, torch.fx.GraphModule):
        gm = model
    elif hasattr(model, "graph"):
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
            # Use a 0-D float32 tensor for eps to avoid fp64 promotion
            # that tripped QNN's dtype map (which has no torch.float64).
            eps_tensor_node = gm.graph.call_function(
                torch.ops.aten.scalar_tensor.default,
                args=(float(eps),),
                kwargs={"dtype": torch.float32},
            )
            eps_node = gm.graph.call_function(
                torch.ops.aten.add.Tensor, args=(var_node, eps_tensor_node)
            )
            rsqrt_node = gm.graph.call_function(
                torch.ops.aten.rsqrt.default, args=(eps_node,)
            )
            norm_node = gm.graph.call_function(
                torch.ops.aten.mul.Tensor, args=(diff_node, rsqrt_node)
            )
            out_node = norm_node
            if weight_node is not None and not isinstance(weight_node, type(None)):
                out_node = gm.graph.call_function(
                    torch.ops.aten.mul.Tensor, args=(out_node, weight_node)
                )
            if bias_node is not None and not isinstance(bias_node, type(None)):
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
        logger.info("Decomposed native_layer_norm into primitive ops")
    return gm


# ---------------------------------------------------------------------------
# Strip Q/DQ nodes wrongly inserted on integer tensors (arange, ids).
# ---------------------------------------------------------------------------


def _remove_int_quantize_nodes(model):
    """PT2E sometimes annotates arange / int-id tensors for quantization;
    those Q/DQ nodes blow up at re-export time. Walk the graph after
    convert_pt2e and erase them."""
    graph = model.graph
    to_erase = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        name = getattr(node.target, "__name__", "")
        if "quantize_per_tensor" not in name and "dequantize_per_tensor" not in name:
            continue
        inp = node.args[0]
        should_remove = False
        if hasattr(inp, "meta") and "val" in inp.meta:
            val = inp.meta["val"]
            if hasattr(val, "dtype") and not val.dtype.is_floating_point \
                    and val.dtype not in (torch.int8, torch.uint8):
                should_remove = True
        if not should_remove and hasattr(inp, "target"):
            inp_name = getattr(inp.target, "__name__", str(inp.target))
            if "arange" in inp_name or inp_name in {"full", "zeros", "ones"}:
                should_remove = True
        if not should_remove and getattr(inp, "op", None) == "get_attr":
            try:
                t = model
                for part in inp.target.split("."):
                    t = getattr(t, part)
                if hasattr(t, "dtype") and t.dtype in (torch.int64, torch.int32, torch.int16, torch.bool):
                    should_remove = True
            except Exception:
                pass
        if should_remove:
            node.replace_all_uses_with(inp)
            to_erase.append(node)
    for n in reversed(to_erase):
        graph.erase_node(n)
    if to_erase:
        graph.lint()
        model.recompile()
        logger.info("Removed %d spurious int-tensor quant nodes", len(to_erase))


# ---------------------------------------------------------------------------
# QNN export routine (April's flow, v0.7.0 API)
# ---------------------------------------------------------------------------


def export_component_to_qnn(
    model: nn.Module,
    sample_inputs: tuple,
    output_path: str,
    soc_model: str = "SM8850",
    quantize: bool = False,
    quant_dtype: str = "8a8w",
    num_calibration: int = 10,
    calibration_data: list = None,
):
    """torch.export -> (PTQ) -> to_edge_transform_and_lower_to_qnn -> .pte."""
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

    dtype_map = {
        "8a8w": QuantDtype.use_8a8w,
        "16a8w": QuantDtype.use_16a8w,
        "16a4w": QuantDtype.use_16a4w,
        "16a16w": QuantDtype.use_16a16w,
    }

    chipset_map = get_soc_to_chipset_map()
    if soc_model not in chipset_map:
        raise ValueError(f"Unknown SoC '{soc_model}'. Available: {list(chipset_map)}")
    chipset = chipset_map[soc_model]

    if quantize:
        logger.info("Applying QNN PTQ quantization (%s) ...", quant_dtype)
        qnn_quant_dtype = dtype_map[quant_dtype]

        logger.info("Pre-export for quantization ...")
        pre_ep = export(model, sample_inputs)
        m = pre_ep.module()

        # Decompose LayerNorm before quantization to keep it on DSP.
        _decompose_layer_norm(m)

        # April's quantizer config — use_per_channel_weight_quant enabled by
        # default in ET 0.7.0 for linears, nothing else to tweak.
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(qnn_quant_dtype)

        m = prepare_pt2e(m, quantizer)

        if calibration_data is not None:
            num_cal = len(calibration_data)
            logger.info("Running %d calibration passes (real data) ...", num_cal)
            with torch.no_grad():
                for cal_i, cal_inputs in enumerate(calibration_data):
                    if not isinstance(cal_inputs, tuple):
                        cal_inputs = (cal_inputs,)
                    m(*cal_inputs)
                    if (cal_i + 1) % 5 == 0 or cal_i + 1 == num_cal:
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
                                    torch.full_like(inp, (cal_i + 1) / (num_calibration + 1))
                                )
                            else:
                                cal_inputs.append(torch.randn_like(inp))
                        else:
                            cal_inputs.append(inp)
                    m(*cal_inputs)
                    if (cal_i + 1) % 5 == 0 or cal_i + 1 == num_calibration:
                        logger.info("  calibration %d/%d", cal_i + 1, num_calibration)

        m = convert_pt2e(m)
        _remove_int_quantize_nodes(m)
        logger.info("QNN quantization complete.")
    else:
        m = model

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

    logger.info("Lowering to QNN backend via to_edge_transform_and_lower_to_qnn ...")
    delegated_program = to_edge_transform_and_lower_to_qnn(
        m,
        sample_inputs,
        compiler_specs,
    )

    logger.info("Serialising to .pte ...")
    et_program = delegated_program.to_executorch()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Saved %s  (%.1f MB)", output_path, size_mb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Export FLUX.2-klein-4B to ExecuTorch QNN/HTP v0.7.0 path",
    )
    p.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-4B")
    p.add_argument("--output_dir", default="./exported_flux2_klein_qnn_v07")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--max_text_len", type=int, default=512)
    p.add_argument("--soc", type=str, default="SM8850",
                   help="e.g. SM8850 (V81, Snapdragon 8 Elite Gen 5)")
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--quant_dtype", default="8a8w",
                   choices=["8a8w", "16a8w", "16a4w", "16a16w"])
    p.add_argument("--num_calibration", type=int, default=10)
    p.add_argument("--component",
                   choices=["all", "transformer", "vae", "vae_encoder", "text_encoder"],
                   default="all")
    p.add_argument("--num_img2img_images", type=int, default=0)
    p.add_argument("--calibration_dir", type=str, default=None)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32

    pipe = load_pipeline(args.model_id, dtype=dtype)
    is_distilled = getattr(pipe.config, "is_distilled", True)

    copy_tokenizer(pipe, str(out))
    save_vae_bn_stats(pipe, str(out))

    te_cfg = pipe.text_encoder.config
    hidden_states_layers = [9, 18, 27]
    logger.info("Text encoder: extracting hidden states from %s", hidden_states_layers)

    cal_data = {"text_encoder": None, "transformer": None, "vae": None}
    if args.calibration_dir and args.quantize:
        for comp in cal_data:
            path = os.path.join(args.calibration_dir, f"calibration_{comp}.pt")
            if os.path.exists(path):
                cal_data[comp] = torch.load(path, weights_only=False)
                logger.info("Loaded %d calibration samples for %s", len(cal_data[comp]), comp)

    t_cfg = pipe.transformer.config
    vae_cfg = pipe.vae.config
    vae_sf = _get_vae_scale_factor(pipe)
    patch_h, patch_w = _compute_latent_dims(args.height, args.width, vae_sf)

    meta = {
        "model_id": args.model_id,
        "backend": "qnn_htp",
        "soc": args.soc,
        "soc_model": args.soc,
        "et_version": "0.7.0",
        "height": args.height,
        "width": args.width,
        "max_text_len": args.max_text_len,
        "quantized": args.quantize,
        "quant_dtype": args.quant_dtype if args.quantize else "none",
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
            "joint_attention_dim": t_cfg.joint_attention_dim,
        },
        "vae": {
            "latent_channels": getattr(vae_cfg, "latent_channels", None),
            "batch_norm_eps": getattr(vae_cfg, "batch_norm_eps", 1e-5),
        },
    }
    (out / "export_config.json").write_text(json.dumps(meta, indent=2))

    # Text encoder
    if args.component in ("all", "text_encoder"):
        logger.info("=" * 60)
        logger.info("EXPORTING TEXT ENCODER (Qwen3) -> QNN/HTP")
        wrapper = Qwen3TextEncoderWrapper(
            pipe.text_encoder, hidden_states_layers=hidden_states_layers,
        ).eval()
        inputs = build_text_encoder_inputs(args.max_text_len)

        export_component_to_qnn(
            wrapper, inputs, str(out / "text_encoder.pte"),
            soc_model=args.soc, quantize=args.quantize, quant_dtype=args.quant_dtype,
            num_calibration=args.num_calibration,
            calibration_data=cal_data["text_encoder"],
        )
        del wrapper, inputs
        _free_memory()

    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        del pipe.text_encoder
        pipe.text_encoder = None
        _free_memory()

    # Transformer
    if args.component in ("all", "transformer"):
        logger.info("=" * 60)
        logger.info("EXPORTING TRANSFORMER -> QNN/HTP")
        wrapper = Flux2TransformerWrapper(pipe.transformer).eval()
        t2i_inputs = build_transformer_inputs(
            pipe, args.height, args.width, args.max_text_len, dtype,
            num_img2img_images=0,
        )

        export_component_to_qnn(
            wrapper, t2i_inputs, str(out / "transformer.pte"),
            soc_model=args.soc, quantize=args.quantize, quant_dtype=args.quant_dtype,
            num_calibration=args.num_calibration,
            calibration_data=cal_data["transformer"],
        )
        del wrapper, t2i_inputs
        _free_memory()

    # VAE decoder
    if args.component in ("all", "vae"):
        logger.info("=" * 60)
        logger.info("EXPORTING VAE DECODER -> QNN/HTP")
        wrapper = VAEDecoderWrapper(pipe.vae).eval()
        inputs = build_vae_inputs(pipe, args.height, args.width, dtype)
        export_component_to_qnn(
            wrapper, inputs, str(out / "vae_decoder.pte"),
            soc_model=args.soc, quantize=args.quantize, quant_dtype=args.quant_dtype,
            num_calibration=args.num_calibration,
            calibration_data=cal_data["vae"],
        )
        del wrapper, inputs
        _free_memory()

    del pipe
    _free_memory()

    print("\n" + "=" * 60 + "\n  EXPORT COMPLETE\n" + "=" * 60)
    for f in sorted(out.glob("*.pte")):
        print(f"  {f.name:30s}  {f.stat().st_size / 1024**2:>8.1f} MB")


if __name__ == "__main__":
    main()
