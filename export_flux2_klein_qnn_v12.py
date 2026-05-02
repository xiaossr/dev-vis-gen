"""
Minimal v1.2.0 QNN export for the FLUX.2-klein transformer.

No sharding, no fallback insertion, no observer override — let the QNN
partitioner decide where to cut. Real calibration data via
calibration_data/calibration_transformer.pt.

Usage:
    python export_flux2_klein_qnn_v12.py \
        --component transformer \
        --output_dir exported_flux2_klein_qnn_v12 \
        --calibration_dir calibration_data
"""
import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from export_flux2_klein_qnn import (  # noqa: E402
    Flux2TransformerWrapper,
    Qwen3TextEncoderWrapper,
    VAEDecoderWrapper,
    build_text_encoder_inputs,
    build_transformer_inputs,
    build_vae_inputs,
    configure_local_tooling,
    copy_tokenizer,
    get_qcom_chipset,
    load_pipeline,
    save_vae_bn_stats,
)

configure_local_tooling()

from executorch.backends.qualcomm.quantizer.quantizer import (  # noqa: E402
    QnnQuantizer,
    QuantDtype,
)
from executorch.backends.qualcomm.serialization.qc_schema import (  # noqa: E402
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.utils.utils import (  # noqa: E402
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.exir import ExecutorchBackendConfig  # noqa: E402
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass  # noqa: E402
from torchao.quantization.pt2e.quantize_pt2e import (  # noqa: E402
    convert_pt2e,
    prepare_pt2e,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("export_v12")


def _load_int16_node_names(profile_path: Path) -> set:
    if profile_path is None or not profile_path.exists():
        return set()
    import json
    data = json.loads(profile_path.read_text())
    return set(data.get("int16_producer_node_names", []))


def _vtcm_overflow_at_int16(node) -> bool:
    """A node operating on >= 2M-element tensor at int16 will not fit V81's 8 MB VTCM."""
    val = node.meta.get("val")
    if val is None or not hasattr(val, "shape"):
        return False
    n = 1
    for d in val.shape:
        n *= int(d)
    return n >= 2_000_000  # 2M elem * 2B = 4 MB; conservative


def export_one(
    name: str,
    model: torch.nn.Module,
    sample_inputs: tuple,
    out_pte: Path,
    soc_chipset,
    quant_dtype: QuantDtype | None,
    calibration_data: list | None,
    int16_profile: Path | None = None,
):
    logger.info("=" * 60)
    logger.info("EXPORTING %s", name.upper())
    logger.info("=" * 60)
    logger.info("Sample input shapes: %s", [tuple(x.shape) for x in sample_inputs])

    logger.info("Pre-export with torch.export ...")
    captured = torch.export.export(model, sample_inputs, strict=True).module()

    if quant_dtype is not None:
        logger.info("Setting up v1.2.0 QnnQuantizer (backend=HTP, soc=%s) ...", soc_chipset)
        quantizer = QnnQuantizer(
            backend=QnnExecuTorchBackendType.kHtpBackend,
            soc_model=soc_chipset,
        )
        # HistogramObserver from torchao (not torch.ao — that path triggers a
        # _PartialWrapper bug in torchao prepare). Cuts outliers via percentile.
        from torchao.quantization.pt2e.observer import HistogramObserver
        from executorch.backends.qualcomm.quantizer.quantizer import ModuleQConfig
        quantizer.set_default_quant_config(
            quant_dtype,
            is_conv_per_channel=True,
            is_linear_per_channel=True,
            act_observer=HistogramObserver,
        )

        # Mixed precision: nodes flagged by profiling get 16a8w
        int16_names = _load_int16_node_names(int16_profile)
        if int16_names:
            logger.info("Mixed precision: %d nodes promoted to 16a8w", len(int16_names))

            def _is_int16_node(node):
                return node.name in int16_names

            quantizer.set_submodule_qconfig_list([
                (_is_int16_node, ModuleQConfig(
                    quant_dtype=QuantDtype.use_16a8w,
                    is_conv_per_channel=True,
                    is_linear_per_channel=True,
                    act_observer=HistogramObserver,
                )),
            ])

        logger.info("prepare_pt2e ...")
        prepared = prepare_pt2e(captured, quantizer)

        if calibration_data:
            n = len(calibration_data)
            logger.info("Running %d calibration passes (real data) ...", n)
            with torch.no_grad():
                for i, cal in enumerate(calibration_data):
                    if not isinstance(cal, tuple):
                        cal = (cal,)
                    prepared(*cal)
                    if (i + 1) % 5 == 0 or (i + 1) == n:
                        logger.info("  calibration %d/%d", i + 1, n)
        else:
            logger.info("No calibration data; running 1 synthetic pass ...")
            with torch.no_grad():
                prepared(*sample_inputs)

        logger.info("convert_pt2e ...")
        converted = convert_pt2e(prepared)
    else:
        logger.info("Skipping PTQ (fp16 path)")
        converted = captured

    logger.info("Building HTP compiler spec ...")
    backend_options = generate_htp_compiler_spec(
        use_fp16=quant_dtype is None,
        use_dlbc=True,  # Bandwidth compression to reduce activation memory
    )
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=soc_chipset,
        backend_options=backend_options,
    )

    # Auto-skip is disabled by default — it was too aggressive (caught tilable
    # matmuls). Set FLUX_ENABLE_AUTO_SKIP=1 to re-enable.
    skip_id_set = None
    if quant_dtype in (QuantDtype.use_16a8w, QuantDtype.use_16a4w, QuantDtype.use_16a16w) \
            and os.environ.get("FLUX_ENABLE_AUTO_SKIP") == "1":
        big = []
        for node in converted.graph.nodes:
            if node.op == "call_function" and _vtcm_overflow_at_int16(node):
                big.append(node.name)
        if big:
            skip_id_set = set(big)
            logger.info("Skipping %d nodes from QNN (would overflow V81 VTCM at int16)",
                        len(skip_id_set))
            logger.info("  examples: %s", sorted(skip_id_set)[:8])

    logger.info("to_edge_transform_and_lower_to_qnn ...")
    edge_mgr = to_edge_transform_and_lower_to_qnn(
        converted, sample_inputs, compiler_specs,
        skip_node_id_set=skip_id_set,
    )

    logger.info("to_executorch ...")
    exec_prog = edge_mgr.to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=True,
                alloc_graph_output=True,
            ),
        ),
    )

    logger.info("Writing %s ...", out_pte)
    out_pte.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pte, "wb") as f:
        exec_prog.write_to_file(f)
    logger.info("Wrote %s (%.1f MB)", out_pte, out_pte.stat().st_size / 1e6)


def load_calibration(calibration_dir: Path | None) -> dict:
    if calibration_dir is None:
        return {"text_encoder": None, "transformer": None, "vae": None}
    out = {}
    for name in ("text_encoder", "transformer", "vae"):
        path = calibration_dir / f"calibration_{name}.pt"
        if path.exists():
            data = torch.load(path, weights_only=False)
            logger.info("Loaded %d real calibration samples for %s", len(data), name)
            out[name] = data
        else:
            out[name] = None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-4B")
    p.add_argument("--component", choices=["transformer", "vae", "text_encoder"], default="transformer")
    p.add_argument("--output_dir", default="exported_flux2_klein_qnn_v12")
    p.add_argument("--soc_model", default="SM8850")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--max_text_len", type=int, default=512)
    p.add_argument("--num_img2img_images", type=int, default=0)
    p.add_argument("--quant_dtype", default="16a8w", choices=["8a8w", "16a8w", "16a4w", "16a16w"])
    p.add_argument("--calibration_dir", default=None)
    p.add_argument("--use_fp16", action="store_true", help="Skip PTQ; export fp16")
    p.add_argument("--int16_profile", default=None,
                   help="Path to int16_profile.json for mixed precision; producer node names get 16a8w")
    args = p.parse_args()

    soc_chipset = get_qcom_chipset(args.soc_model)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cal_dir = Path(args.calibration_dir) if args.calibration_dir else None
    cal = load_calibration(cal_dir)

    DTYPE_MAP = {
        "8a8w": QuantDtype.use_8a8w,
        "16a8w": QuantDtype.use_16a8w,
        "16a4w": QuantDtype.use_16a4w,
        "16a16w": QuantDtype.use_16a16w,
    }
    quant_dtype = None if args.use_fp16 else DTYPE_MAP[args.quant_dtype]

    logger.info("Loading pipeline %s ...", args.model_id)
    pipe = load_pipeline(args.model_id, dtype=torch.float32)

    if args.component == "text_encoder":
        copy_tokenizer(pipe, str(out))
        model = Qwen3TextEncoderWrapper(pipe.text_encoder, hidden_states_layers=()).eval().cpu()
        sample_inputs = build_text_encoder_inputs(args.max_text_len)
        export_one("text_encoder", model, sample_inputs, out / "text_encoder.pte",
                   soc_chipset, quant_dtype, cal["text_encoder"])
    elif args.component == "transformer":
        model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
        sample_inputs = build_transformer_inputs(
            pipe, args.height, args.width, args.max_text_len,
            dtype=torch.float32, num_img2img_images=args.num_img2img_images,
        )
        int16_profile_path = Path(args.int16_profile) if args.int16_profile else None
        export_one("transformer", model, sample_inputs, out / "transformer.pte",
                   soc_chipset, quant_dtype, cal["transformer"],
                   int16_profile=int16_profile_path)
    elif args.component == "vae":
        save_vae_bn_stats(pipe, str(out))
        model = VAEDecoderWrapper(pipe.vae).eval().cpu()
        sample_inputs = build_vae_inputs(pipe, args.height, args.width, dtype=torch.float32)
        export_one("vae", model, sample_inputs, out / "vae_decoder.pte",
                   soc_chipset, quant_dtype, cal["vae"])

    del pipe
    gc.collect()


if __name__ == "__main__":
    main()
