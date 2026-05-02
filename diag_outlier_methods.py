"""Compare outlier-handling methods at 8a8w in host PT2E simulation.

Each method tries to recover signal at 8a8w that pure 8a8w loses (~SNR<<0).
Goal: which methods produce clean output (SNR > 15 dB)?

  A) Baseline 8a8w (no fix)
  B) Static SmoothQuant: per-channel scale on encoder_hidden_states + worst
     linears; fold scale into linear weights at export time.
  C) Hadamard rotation: orthogonal mixing on encoder_hidden_states; fold
     H^-1 into context_embedder.weight.
  D) Per-token L2 dynamic: divide each token by its max-abs before each
     problem linear, multiply after; supported ops only.
  E) Timestep-bucketed SmoothQuant: 2 buckets, separate s per bucket.

All methods use HistogramObserver and 5 calibration samples (mixed t).
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
log = logging.getLogger("methods")


def diff(name, ref, q):
    ref = ref.detach().float(); q = q.detach().float()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0), q.flatten().unsqueeze(0)).item()
    snr_db = 20 * torch.log10(ref.norm() / ((ref - q).norm() + 1e-12)).item()
    log.info("[%s]  max=%.4f  cos=%.5f  SNR=%.2fdB",
             name, (ref - q).abs().max().item(), cos, snr_db)
    return {"name": name, "cos": cos, "snr_db": snr_db}


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


def hadamard_matrix(n: int) -> torch.Tensor:
    """Sylvester construction; n must be power of 2."""
    assert n & (n - 1) == 0, f"{n} not power of 2"
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


# --- The instrumentation: we modify the *transformer module* in-place
# before each test. After test, restore from a deep-copied original.

def _find_context_embedder(transformer):
    return transformer.context_embedder


def apply_smoothquant(transformer, cal_samples, alpha=0.5):
    """Per-channel SmoothQuant on context_embedder (which sees encoder_hidden_states).

    Compute per-channel scale s = (max|x|)^alpha / (max|w|)^(1-alpha) per channel.
    Then divide x by s, multiply W by s. Mathematically y = (x/s) @ (s*W) = x@W.

    We can't fold "divide x by s" into the runtime graph cleanly, so we wrap
    context_embedder with a forward hook that does the division.
    """
    log.info("[SmoothQuant] collecting input stats for context_embedder ...")
    inputs_seen = []
    def cap_hook(_m, inputs, _out):
        x = inputs[0]
        inputs_seen.append(x.detach().abs())

    h = transformer.context_embedder.register_forward_hook(cap_hook)
    with torch.no_grad():
        for c in cal_samples:
            if not isinstance(c, tuple): c = (c,)
            transformer(c[0], c[1], c[2], c[3], c[4],
                        guidance=None, return_dict=False)
    h.remove()

    x_max = torch.zeros(transformer.context_embedder.in_features)
    for x in inputs_seen:
        flat = x.reshape(-1, x.shape[-1])
        x_max = torch.maximum(x_max, flat.max(dim=0).values)
    x_max = x_max.clamp(min=1e-5)

    W = transformer.context_embedder.weight.data  # [out, in]
    w_max = W.abs().max(dim=0).values.clamp(min=1e-5)

    # SmoothQuant scale per input channel
    s = (x_max.pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-5)

    # Fold s into the weight: W' = W * s
    transformer.context_embedder.weight.data = W * s.unsqueeze(0)
    log.info("[SmoothQuant] folded s into context_embedder.weight, range %.3f..%.3f",
             s.min().item(), s.max().item())

    # We need to divide the input x by s at runtime. Wrap the linear.
    orig_forward = transformer.context_embedder.forward
    s_buf = s.clone()
    def new_forward(x):
        return orig_forward(x / s_buf)
    transformer.context_embedder.forward = new_forward
    log.info("[SmoothQuant] wrapped context_embedder.forward to divide input by s")


def apply_hadamard(transformer):
    """Apply Hadamard rotation on context_embedder input (encoder_hidden_states)
    of shape [B, S, in=7680]. 7680 = 2^9 * 15 = not power of 2, can't use full
    Hadamard. Use blockwise Hadamard with block size 128 (each head_dim block
    rotates independently)."""
    in_dim = transformer.context_embedder.in_features
    block = 128  # head_dim
    n_blocks = in_dim // block
    assert in_dim == n_blocks * block, f"{in_dim} not divisible by {block}"

    H = hadamard_matrix(block)  # [128, 128]
    # Block-diagonal Hadamard: kron(I_n, H)
    # We don't materialize the full matrix; instead reshape and matmul in blocks.
    # W_new = (kron(I, H_T) @ W) @ kron(I, H) is just W with input axis permuted via H.
    # Simpler: y = (x @ W^T) where W: [out, in]. Define H_full = block-diag H.
    # New: x' = x @ H_full (rotates along last dim block-by-block).
    # Then y = x' @ (H_full^T @ W^T) so we set W'^T = H_full^T @ W^T → W' = W @ H_full.
    # Since H is symmetric and orthogonal: H^T = H, H @ H = I.
    W = transformer.context_embedder.weight.data  # [out, in]
    out, in_ = W.shape
    Wb = W.reshape(out, n_blocks, block)  # [out, n_blocks, 128]
    Wb_rot = torch.einsum("onb,bc->onc", Wb, H.to(Wb.dtype))  # rotate per block on last
    transformer.context_embedder.weight.data = Wb_rot.reshape(out, in_)

    # Apply x -> x @ block-diag(H) at runtime by wrapping forward.
    orig_forward = transformer.context_embedder.forward
    H_buf = H.clone()
    def new_forward(x):
        # x: [..., in] -> reshape to [..., n_blocks, 128] -> rotate -> reshape back
        prefix = x.shape[:-1]
        xb = x.reshape(*prefix, n_blocks, block)
        xb_rot = torch.einsum("...nb,bc->...nc", xb, H_buf.to(xb.dtype))
        return orig_forward(xb_rot.reshape(*prefix, in_))
    transformer.context_embedder.forward = new_forward
    log.info("[Hadamard] applied block-diag Hadamard (block=%d, n_blocks=%d) to context_embedder",
             block, n_blocks)


def apply_per_token_l2(transformer):
    """Per-token max-abs scaling around context_embedder.

    Forward:
      s = max(|x|, dim=-1, keepdim=True).clamp(min=1e-5)   # [B, S, 1]
      x' = x / s
      y = x' @ W
      y_out = y * s   (broadcast back)

    Note: y * s isn't mathematically equivalent to x @ W (the output channels
    don't have a single "scale per token" since W mixes channels). This is an
    approximation — but it's what naive per-token quant would do, and it gives
    quant a uniformly-scaled input. The downstream "y * s" must be on a
    different layer; we'll just do x' @ W and not multiply back, then compare.
    Actually the right form is: x' has |.|<=1 per token; y = x' @ W. Quantize
    y. Then post-multiply by s: y_real = y * s. The post-multiply happens on
    the small s tensor (broadcast). Both ops are supported.
    """
    orig_forward = transformer.context_embedder.forward
    def new_forward(x):
        s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_scaled = x / s
        y = orig_forward(x_scaled)
        return y * s
    transformer.context_embedder.forward = new_forward
    log.info("[PerTokenL2] wrapped context_embedder with per-token rescale")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+",
                   default=["baseline", "smoothquant", "hadamard", "per_token"],
                   choices=["baseline", "smoothquant", "hadamard", "per_token"])
    p.add_argument("--ncal", type=int, default=5)
    args = p.parse_args()

    log.info("Loading pipeline ...")
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    cal = torch.load(
        str(Path(__file__).parent / "calibration_data" / "calibration_transformer.pt"),
        weights_only=False)[:args.ncal]
    probe = cal[0] if isinstance(cal[0], tuple) else (cal[0],)

    # Snapshot original transformer state for restoration between tests.
    orig_state = copy.deepcopy(pipe.transformer.state_dict())

    results = []

    def reset_transformer():
        pipe.transformer.load_state_dict(orig_state)
        # Restore original forward functions (in case we wrapped them)
        # Reload from class to drop instance-level overrides.
        from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel  # noqa: F401
        ce = pipe.transformer.context_embedder
        if isinstance(ce, nn.Linear):
            # Drop instance forward override
            if "forward" in ce.__dict__:
                del ce.__dict__["forward"]

    # ref forward (use clean original)
    reset_transformer()
    ref_model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
    log.info("=== fp32 reference ===")
    with torch.no_grad():
        ref = ref_model(*probe)
    if isinstance(ref, tuple): ref = ref[0]

    if "baseline" in args.methods:
        log.info("\n############ A) BASELINE 8a8w ############")
        reset_transformer()
        m = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
        out = run_pt2e(m, probe, cal)
        if isinstance(out, tuple): out = out[0]
        results.append(diff("A_baseline", ref, out))

    if "smoothquant" in args.methods:
        log.info("\n############ B) SMOOTHQUANT (context_embedder) ############")
        reset_transformer()
        # Need to apply on the inner transformer, then wrap
        apply_smoothquant(pipe.transformer, cal, alpha=0.5)
        m = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
        out = run_pt2e(m, probe, cal)
        if isinstance(out, tuple): out = out[0]
        results.append(diff("B_smoothquant", ref, out))

    if "hadamard" in args.methods:
        log.info("\n############ C) HADAMARD (context_embedder) ############")
        reset_transformer()
        apply_hadamard(pipe.transformer)
        m = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
        out = run_pt2e(m, probe, cal)
        if isinstance(out, tuple): out = out[0]
        results.append(diff("C_hadamard", ref, out))

    if "per_token" in args.methods:
        log.info("\n############ D) PER-TOKEN L2 (context_embedder) ############")
        reset_transformer()
        apply_per_token_l2(pipe.transformer)
        m = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
        out = run_pt2e(m, probe, cal)
        if isinstance(out, tuple): out = out[0]
        results.append(diff("D_per_token", ref, out))

    log.info("\n=================== SUMMARY ===================")
    for r in results:
        log.info("  %-25s SNR=%6.2f dB  cos=%.5f", r["name"], r["snr_db"], r["cos"])

    out_path = Path(__file__).parent / "outlier_methods_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
