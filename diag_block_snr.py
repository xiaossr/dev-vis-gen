"""Per-block SNR propagation.

Hook on each transformer block's output (and also on key intermediate
points within attention) for both fp32 and 8a8w-PT2E runs. Compare to see
where noise enters and how it compounds.

Output: a table of per-block SNR/cos showing the propagation profile.
"""
import json
import logging
import sys
from pathlib import Path

import torch
from torchao.quantization.pt2e.observer import HistogramObserver

sys.path.insert(0, str(Path(__file__).parent))
from export_flux2_klein_qnn import (
    Flux2TransformerWrapper,
    build_transformer_inputs,
    configure_local_tooling,
    load_pipeline,
)
configure_local_tooling()

from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType, QcomChipset,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("blksnr")


def snr_db(ref, q):
    return (20 * torch.log10(ref.norm() / ((ref - q).norm() + 1e-12))).item()


def cos_sim(ref, q):
    return torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0), q.flatten().unsqueeze(0)).item()


def collect_block_outputs(transformer, sample_inputs):
    """Run a forward and capture {block_name: tuple_of_outputs} for every block."""
    captures = {}

    def make_hook(name):
        def h(_m, _inp, out):
            # outputs may be tensor or tuple
            if isinstance(out, tuple):
                captures[name] = tuple(o.detach().clone() if torch.is_tensor(o) else o for o in out)
            elif torch.is_tensor(out):
                captures[name] = out.detach().clone()
        return h

    handles = []
    # double-stream blocks
    for i, blk in enumerate(transformer.transformer_blocks):
        handles.append(blk.register_forward_hook(make_hook(f"D{i:02d}")))
    # single-stream blocks
    for i, blk in enumerate(transformer.single_transformer_blocks):
        handles.append(blk.register_forward_hook(make_hook(f"S{i:02d}")))

    with torch.no_grad():
        # call as wrapper does
        _ = transformer(
            hidden_states=sample_inputs[0],
            encoder_hidden_states=sample_inputs[1],
            timestep=sample_inputs[2],
            img_ids=sample_inputs[3],
            txt_ids=sample_inputs[4],
            guidance=None,
            return_dict=False,
        )
    for h in handles:
        h.remove()
    return captures


def main():
    log.info("Loading ...")
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    cal = torch.load(
        str(Path(__file__).parent / "calibration_data" / "calibration_transformer.pt"),
        weights_only=False)[:5]
    probe = cal[0] if isinstance(cal[0], tuple) else (cal[0],)

    transformer = pipe.transformer.eval().cpu()

    log.info("=== fp32 forward with hooks ===")
    fp_caps = collect_block_outputs(transformer, probe)
    log.info("Captured %d blocks (fp32)", len(fp_caps))

    # Now PT2E 8a8w
    log.info("=== Build 8a8w PT2E model ===")
    wrapper = Flux2TransformerWrapper(transformer).eval().cpu()
    captured = torch.export.export(wrapper, probe, strict=True).module()
    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=QcomChipset.SM8850,
    )
    quantizer.set_default_quant_config(
        QuantDtype.use_8a8w,
        is_conv_per_channel=True, is_linear_per_channel=True,
        act_observer=HistogramObserver,
    )
    log.info("prepare ...")
    prepared = prepare_pt2e(captured, quantizer)
    log.info("calibrate ...")
    with torch.no_grad():
        for c in cal:
            if not isinstance(c, tuple): c = (c,)
            prepared(*c)
    log.info("convert ...")
    converted = convert_pt2e(prepared)

    # The PT2E-converted module is a flat FX module — block boundaries are
    # gone. We need a different trick: re-attach hooks to the ORIGINAL
    # transformer's blocks, then route the calibrated quantization through
    # the original module by replacing parameters.
    # Easier path: run the PT2E model fwd with hooks on a *parallel* original
    # transformer that mirrors block computations. Not feasible since PT2E
    # transforms the graph globally.
    #
    # Alt: capture each block's output via its FX-graph signature. Hard.
    #
    # Practical alt: write a simulated 8a8w forward. For each block, take
    # fp32 input, quantize via the captured observers' scales, run block
    # fp32, dequantize. Same intermediate quality as PT2E. We don't have
    # observer values per-block easily.
    #
    # SIMPLEST: run the fp32 model with FAKE-QUANT (at int8) on every
    # activation. We can do this by inserting a hook that round-trips x
    # through int8 at each block boundary. This isn't the exact PT2E noise,
    # but matches in spirit: per-tensor int8 with min/max from fp32.
    log.info("=== Simulated per-block round-trip int8 (per-tensor min-max) ===")

    # First pass: collect per-block min/max from calibration (fp32)
    log.info("Calibrate per-block min/max from %d samples ...", len(cal))
    block_stats = {}  # name -> (min, max)
    def make_stat_hook(name):
        def h(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(t): return
            mn, mx = t.detach().min().item(), t.detach().max().item()
            cur = block_stats.get(name)
            if cur is None: block_stats[name] = (mn, mx)
            else: block_stats[name] = (min(cur[0], mn), max(cur[1], mx))
        return h
    handles = []
    for i, blk in enumerate(transformer.transformer_blocks):
        handles.append(blk.register_forward_hook(make_stat_hook(f"D{i:02d}")))
    for i, blk in enumerate(transformer.single_transformer_blocks):
        handles.append(blk.register_forward_hook(make_stat_hook(f"S{i:02d}")))
    with torch.no_grad():
        for c in cal:
            if not isinstance(c, tuple): c = (c,)
            transformer(
                hidden_states=c[0], encoder_hidden_states=c[1], timestep=c[2],
                img_ids=c[3], txt_ids=c[4], guidance=None, return_dict=False,
            )
    for h in handles: h.remove()

    # Second pass: round-trip int8 quantize each block output
    log.info("Round-trip int8 with calibrated min/max ...")
    quantized_caps = {}
    def make_rt_hook(name):
        def h(_m, _inp, out):
            stats = block_stats.get(name)
            if stats is None: return out
            mn, mx = stats
            scale = max(abs(mn), abs(mx)) / 127.0
            scale = max(scale, 1e-8)
            def quantize_t(t):
                qt = torch.round(t / scale).clamp(-128, 127)
                return (qt * scale).to(t.dtype)
            if isinstance(out, tuple):
                new_out = tuple(quantize_t(o) if torch.is_tensor(o) else o for o in out)
            else:
                new_out = quantize_t(out)
            quantized_caps[name] = (new_out[0] if isinstance(new_out, tuple) else new_out).detach().clone()
            return new_out
        return h
    handles = []
    for i, blk in enumerate(transformer.transformer_blocks):
        handles.append(blk.register_forward_hook(make_rt_hook(f"D{i:02d}")))
    for i, blk in enumerate(transformer.single_transformer_blocks):
        handles.append(blk.register_forward_hook(make_rt_hook(f"S{i:02d}")))
    with torch.no_grad():
        out_q = transformer(
            hidden_states=probe[0], encoder_hidden_states=probe[1], timestep=probe[2],
            img_ids=probe[3], txt_ids=probe[4], guidance=None, return_dict=False,
        )
    for h in handles: h.remove()

    log.info("=== Per-block SNR (round-trip per-tensor int8 simulation) ===")
    log.info("%-6s %-25s %-10s %-10s %-10s", "block", "shape", "max|fp|", "SNR_dB", "cos")
    rows = []
    sorted_names = sorted([n for n in fp_caps if n in quantized_caps])
    for name in sorted_names:
        fp = fp_caps[name]
        if isinstance(fp, tuple): fp = fp[0]
        q = quantized_caps[name]
        if not torch.is_tensor(fp) or not torch.is_tensor(q): continue
        snr = snr_db(fp, q)
        cos = cos_sim(fp, q)
        sh = tuple(fp.shape)
        log.info("%-6s %-25s %-10.3f %-10.2f %-10.5f",
                 name, sh, fp.abs().max().item(), snr, cos)
        rows.append({"block": name, "shape": sh, "max_abs": fp.abs().max().item(),
                     "snr_db": snr, "cos": cos})

    out_path = Path(__file__).parent / "block_snr.json"
    out_path.write_text(json.dumps(rows, indent=2))
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
