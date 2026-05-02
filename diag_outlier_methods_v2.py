"""Outlier methods v2 — host-side preprocessing of encoder_hidden_states.

The v1 attempt wrapped context_embedder.forward inside the captured graph,
which meant PT2E's activation observer was placed on the RAW input to the
graph (before division). v2 simulates host-side preprocessing: we apply the
SmoothQuant/Hadamard transform to encoder_hidden_states OUTSIDE the wrapper,
so the observer sees the rescaled tensor (which has good range).

In deployment this corresponds to running a tiny preproc op on CPU between
the text encoder pte and the transformer pte.

Also tries per-token L2 dynamic — for this we modify the wrapper to consume
already-rescaled input and store s in a side-channel buffer that's used
post-output. NOTE: since per-token rescale is dynamic, simulating it host-side
requires also passing s, but our wrapper signature is fixed. We just apply
the per-token rescale to the inputs and skip the post-output multiply (the
SNR comparison will reflect the model's behavior on rescaled inputs only —
not a perfect simulation but informative).
"""
import argparse
import copy
import json
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchao.quantization.pt2e.observer import HistogramObserver

sys.path.insert(0, str(Path(__file__).parent))
from export_flux2_klein_qnn import (
    Flux2TransformerWrapper,
    build_transformer_inputs,
    configure_local_tooling,
    load_pipeline,
)
configure_local_tooling()

from executorch.backends.qualcomm.quantizer.quantizer import (
    QnnQuantizer, QuantDtype,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType, QcomChipset,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("v2")


def diff(name, ref, q):
    ref = ref.detach().float(); q = q.detach().float()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0), q.flatten().unsqueeze(0)).item()
    snr_db = 20 * torch.log10(ref.norm() / ((ref - q).norm() + 1e-12)).item()
    log.info("[%s]  max=%.4f  cos=%.5f  SNR=%.2fdB",
             name, (ref - q).abs().max().item(), cos, snr_db)
    return {"name": name, "cos": cos, "snr_db": snr_db}


def hadamard_matrix(n: int) -> torch.Tensor:
    assert n & (n - 1) == 0
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def run_pt2e(model, probe, calib):
    captured = torch.export.export(model, probe, strict=True).module()
    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=QcomChipset.SM8850,
    )
    quantizer.set_default_quant_config(
        QuantDtype.use_8a8w,
        is_conv_per_channel=True, is_linear_per_channel=True,
        act_observer=HistogramObserver,
    )
    prepared = prepare_pt2e(captured, quantizer)
    with torch.no_grad():
        for c in calib:
            if not isinstance(c, tuple): c = (c,)
            prepared(*c)
    converted = convert_pt2e(prepared)
    with torch.no_grad():
        out = converted(*probe)
    if isinstance(out, tuple): out = out[0]
    return out


def transform_inputs(inputs_list, transform_fn, idx=1):
    """Apply transform_fn to one positional argument (encoder_hidden_states is idx 1)."""
    out = []
    for c in inputs_list:
        if not isinstance(c, tuple): c = (c,)
        new_c = list(c)
        new_c[idx] = transform_fn(new_c[idx])
        out.append(tuple(new_c))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+",
                   default=["baseline", "smoothquant_pre", "hadamard_pre", "per_token_pre"])
    p.add_argument("--ncal", type=int, default=5)
    args = p.parse_args()

    log.info("Loading pipeline ...")
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    cal = torch.load(
        str(Path(__file__).parent / "calibration_data" / "calibration_transformer.pt"),
        weights_only=False)[:args.ncal]
    probe_orig = cal[0] if isinstance(cal[0], tuple) else (cal[0],)
    orig_state = copy.deepcopy(pipe.transformer.state_dict())

    def reset():
        pipe.transformer.load_state_dict(orig_state)

    # The model wraps the transformer
    reset()
    ref_model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
    log.info("=== fp32 reference (orig inputs) ===")
    with torch.no_grad():
        ref = ref_model(*probe_orig)
    if isinstance(ref, tuple): ref = ref[0]

    results = []

    if "baseline" in args.methods:
        log.info("\n############ BASELINE 8a8w (no transform) ############")
        reset()
        m = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
        out = run_pt2e(m, probe_orig, cal)
        results.append(diff("BASELINE", ref, out))

    # Compute SmoothQuant scale from calibration data
    if "smoothquant_pre" in args.methods or "hadamard_pre" in args.methods:
        # Stat: per-channel max-abs over all calibration samples
        ehs_max = torch.zeros(7680)
        for c in cal:
            ehs = c[1] if isinstance(c, tuple) else c[1]
            x = ehs.abs().reshape(-1, ehs.shape[-1])
            ehs_max = torch.maximum(ehs_max, x.max(dim=0).values)
        log.info("encoder_hidden_states per-channel max: min=%.3f max=%.3f median=%.3f",
                 ehs_max.min().item(), ehs_max.max().item(), ehs_max.median().item())

    if "smoothquant_pre" in args.methods:
        log.info("\n############ SMOOTHQUANT (pre-rescale on host) ############")
        reset()
        # SmoothQuant: s = ehs_max^alpha / w_max^(1-alpha), per channel
        ce = pipe.transformer.context_embedder
        W = ce.weight.data  # [out, in=7680]
        w_max = W.abs().max(dim=0).values.clamp(min=1e-5)
        x_max = ehs_max.clamp(min=1e-5)
        alpha = 0.5
        s = (x_max.pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-5)
        log.info("smoothquant s: min=%.4f max=%.4f median=%.4f",
                 s.min().item(), s.max().item(), s.median().item())

        # Fold s into weight; pre-divide input
        ce.weight.data = W * s.unsqueeze(0)
        s_buf = s.clone()
        new_cal = transform_inputs(cal, lambda t: t / s_buf)
        new_probe = transform_inputs([probe_orig], lambda t: t / s_buf)[0]

        # Math sanity: with these inputs the fp32 output should be identical to ref.
        m_check = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
        with torch.no_grad():
            out_fp = m_check(*new_probe)
        if isinstance(out_fp, tuple): out_fp = out_fp[0]
        log.info("fp32 sanity: max diff vs ref = %.2e",
                 (out_fp - ref).abs().max().item())

        out = run_pt2e(m_check, new_probe, new_cal)
        results.append(diff("SMOOTHQUANT_PRE", ref, out))

    if "hadamard_pre" in args.methods:
        log.info("\n############ HADAMARD (pre-rotate on host, block-diag, block=128) ############")
        reset()
        ce = pipe.transformer.context_embedder
        in_dim = ce.in_features
        block = 128
        n_blocks = in_dim // block
        H = hadamard_matrix(block)
        W = ce.weight.data
        out_, _ = W.shape
        Wb = W.reshape(out_, n_blocks, block)
        Wb_rot = torch.einsum("onb,bc->onc", Wb, H.to(Wb.dtype))
        ce.weight.data = Wb_rot.reshape(out_, in_dim)

        def rotate_input(t):
            prefix = t.shape[:-1]
            tb = t.reshape(*prefix, n_blocks, block)
            tb_rot = torch.einsum("...nb,bc->...nc", tb, H.to(tb.dtype))
            return tb_rot.reshape(*prefix, in_dim)

        new_cal = transform_inputs(cal, rotate_input)
        new_probe = transform_inputs([probe_orig], rotate_input)[0]

        m_check = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
        with torch.no_grad():
            out_fp = m_check(*new_probe)
        if isinstance(out_fp, tuple): out_fp = out_fp[0]
        log.info("fp32 sanity: max diff vs ref = %.2e",
                 (out_fp - ref).abs().max().item())
        log.info("rotated ehs range: %.2f", new_probe[1].abs().max().item())

        out = run_pt2e(m_check, new_probe, new_cal)
        results.append(diff("HADAMARD_PRE", ref, out))

    if "per_token_pre" in args.methods:
        log.info("\n############ PER-TOKEN L2 (pre-divide on host) ############")
        reset()
        # For per-token L2: x' = x / max(|x|, dim=-1, keepdim=True)
        # Then y' = x' @ W; true y = y' * s_token. We can't recover s_token
        # in the transformer pte (since each token has its own s, dependent
        # on input). Best simulation: feed normalized inputs and accept that
        # the output magnitude will be wrong by per-token factor — we report
        # the cosine similarity (which is scale-invariant) and SNR (which
        # measures relative error).
        def per_token_l2(t):
            s = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
            return t / s
        new_cal = transform_inputs(cal, per_token_l2)
        new_probe = transform_inputs([probe_orig], per_token_l2)[0]

        m_check = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
        with torch.no_grad():
            out_fp = m_check(*new_probe)
        if isinstance(out_fp, tuple): out_fp = out_fp[0]
        # Compare quantized run to fp32 of same transformed inputs (not orig ref)
        out = run_pt2e(m_check, new_probe, new_cal)
        # Use the new fp32 forward as "ref" for this method since input is changed
        results.append(diff("PER_TOKEN_PRE_vs_fp32_xformed", out_fp, out))
        # Also report cosine to original ref (this captures whether the model
        # distinguishes content despite scale; we expect lower because input changed)
        results.append(diff("PER_TOKEN_PRE_vs_origref", ref, out))

    log.info("\n=================== SUMMARY ===================")
    for r in results:
        log.info("  %-35s SNR=%6.2f dB  cos=%.5f", r["name"], r["snr_db"], r["cos"])

    out_path = Path(__file__).parent / "outlier_methods_v2_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
