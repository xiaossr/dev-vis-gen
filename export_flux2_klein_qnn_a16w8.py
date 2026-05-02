"""
Linear-only-discard a16w8 QNN HTP export for the FLUX.2-klein transformer.

Same recipe as export_flux2_klein_qnn_lin_only.py except activations are
16-bit (use_16a8w) instead of 8-bit (use_8a8w). Host PT2E gives +23.1 dB /
cos 0.998 (vs +3.6 dB / cos 0.82 at a8w8). Selective promotion of "noisy"
Linears didn't move the number; promoting all 109 Linears does — quant noise
is broadly distributed across blocks, not concentrated.

Risks at compile time: int16 activations flowing through broadcast muls
(modulation, attn bmm) may exceed VTCM tile budget — same failure pattern
rotary chunking already addresses. If a downstream broadcast mul fails,
apply rotary-style head/seq chunking.

Usage:
    FLUX_ROTARY_HEAD_SPLIT=2 \
    python export_flux2_klein_qnn_a16w8.py \
        --output_dir exported_flux2_klein_qnn_a16w8 \
        --calibration_dir calibration_data
"""
import argparse
import gc
import logging
import os
import sys
from pathlib import Path

import torch

_REPO = str(Path(__file__).resolve().parent)
sys.path.insert(0, _REPO)
from export_flux2_klein_qnn import (  # noqa: E402
    Flux2TransformerWrapper,
    build_transformer_inputs,
    configure_local_tooling,
    get_qcom_chipset,
    load_pipeline,
)
configure_local_tooling(allow_reexec=True)

# IMPORTANT: keep _REPO on sys.path. The local `executorch/` tree carries
# patches that the v1.2.0 site-packages version lacks — see V12_PATH.md,
# specifically the LayerNorm None-weight/bias fixes in
# backends/qualcomm/builders/op_layer_norm.py and
# backends/qualcomm/quantizer/annotators/htp_rules.py. Removing _REPO falls
# back to the unpatched venv executorch and fails at the partitioner with
# "AttributeError: 'NoneType' object has no attribute 'name'".
# Run with `.venv-et12/bin/python` (the v1.2.0 env), NOT `.venv/bin/python`
# (v0.6 — incompatible with the local v1.2.0 tree).

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
from torchao.quantization.pt2e.observer import HistogramObserver  # noqa: E402
from torchao.quantization.pt2e.quantize_pt2e import (  # noqa: E402
    convert_pt2e,
    prepare_pt2e,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("export_lin_only")


def build_aggressive_discard_list(quantizer: QnnQuantizer) -> list:
    """Discard everything in `quant_ops` except linear/conv variants.

    Mirrors the `aggressive` list in diag_linear_only_quant.py.
    """
    aten = torch.ops.aten
    keep = {aten.linear.default, aten.conv2d.default, aten.conv1d.default}
    discard = [op for op in quantizer.quant_ops if op not in keep]
    return discard


def export_transformer(
    model: torch.nn.Module,
    sample_inputs: tuple,
    out_pte: Path,
    soc_chipset,
    calibration_data: list,
):
    logger.info("=" * 60)
    logger.info("EXPORTING TRANSFORMER (a16w8, linear-only-discard)")
    logger.info("=" * 60)
    logger.info("Sample input shapes: %s", [tuple(x.shape) for x in sample_inputs])

    logger.info("Pre-export with torch.export ...")
    captured = torch.export.export(model, sample_inputs, strict=True).module()

    logger.info("Setting up QnnQuantizer (HTP, soc=%s, 16a8w) ...", soc_chipset)
    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=soc_chipset,
    )
    quantizer.set_default_quant_config(
        QuantDtype.use_16a8w,
        is_conv_per_channel=True,
        is_linear_per_channel=True,
        act_observer=HistogramObserver,
    )

    discard = build_aggressive_discard_list(quantizer)
    logger.info(
        "Aggressive discard: %d / %d ops (keeping only linear+conv)",
        len(discard), len(quantizer.quant_ops),
    )
    quantizer.add_discard_ops(discard)

    logger.info("prepare_pt2e ...")
    prepared = prepare_pt2e(captured, quantizer)

    n = len(calibration_data)
    logger.info("Running %d calibration passes (real data) ...", n)
    with torch.no_grad():
        for i, cal in enumerate(calibration_data):
            if not isinstance(cal, tuple):
                cal = (cal,)
            prepared(*cal)
            logger.info("  calibration %d/%d", i + 1, n)

    logger.info("convert_pt2e ...")
    converted = convert_pt2e(prepared)

    logger.info("Building HTP compiler spec ...")
    backend_options = generate_htp_compiler_spec(
        use_fp16=False,
        use_dlbc=True,
    )
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=soc_chipset,
        backend_options=backend_options,
    )

    logger.info("to_edge_transform_and_lower_to_qnn ...")
    edge_mgr = to_edge_transform_and_lower_to_qnn(
        converted, sample_inputs, compiler_specs,
        skip_node_id_set=None,
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-4B")
    p.add_argument("--output_dir", default="exported_flux2_klein_qnn_a16w8")
    p.add_argument("--soc_model", default="SM8850")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--max_text_len", type=int, default=512)
    p.add_argument("--num_img2img_images", type=int, default=0)
    p.add_argument("--calibration_dir", default="calibration_data")
    p.add_argument("--num_calibration_samples", type=int, default=5)
    args = p.parse_args()

    soc_chipset = get_qcom_chipset(args.soc_model)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cal_path = Path(args.calibration_dir) / "calibration_transformer.pt"
    if not cal_path.exists():
        raise FileNotFoundError(f"Calibration data not found: {cal_path}")
    logger.info("Loading calibration data from %s ...", cal_path)
    cal = torch.load(str(cal_path), weights_only=False)
    cal = cal[: args.num_calibration_samples]
    logger.info("Using %d calibration samples", len(cal))

    logger.info("Loading pipeline %s ...", args.model_id)
    pipe = load_pipeline(args.model_id, dtype=torch.float32)

    model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
    sample_inputs = build_transformer_inputs(
        pipe, args.height, args.width, args.max_text_len,
        dtype=torch.float32, num_img2img_images=args.num_img2img_images,
    )

    export_transformer(
        model=model,
        sample_inputs=sample_inputs,
        out_pte=out / "transformer.pte",
        soc_chipset=soc_chipset,
        calibration_data=cal,
    )

    del pipe
    gc.collect()


if __name__ == "__main__":
    main()
