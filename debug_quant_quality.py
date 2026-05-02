#!/usr/bin/env python3
"""
Host-side debug tool: measure quantization-induced divergence in the FLUX.2
transformer against fp32 PyTorch reference.

Runs the exact same positional-tensor inputs through:
  (1) the unquantized fp32 transformer loaded from HuggingFace, and
  (2) a serialized transformer.pte (XNNPACK dynamic w8a8 or QNN HTP static w8a8),

then reports per-tensor correlation, max abs diff, mean abs diff, and
per-channel RMS. Useful for catching noise-producing quantization BEFORE
burning a 10-minute adb push.

Also supports saving the fp32 reference so that a later run against a fresh
transformer.pte can compare against the cached reference without re-loading
the 16 GB pipeline.

Usage
-----
    # 1. Capture fp32 reference (slow — loads full pipeline).
    python debug_quant_quality.py --mode reference --ref_out ./debug_ref.pt

    # 2. Compare an exported .pte against the cached reference (fast).
    python debug_quant_quality.py --mode compare \
        --ref ./debug_ref.pt \
        --pte ./exported_flux2_klein_xnnpack/transformer.pte

    # 3. One-shot: do both in one run.
    python debug_quant_quality.py --mode compare \
        --pte ./exported_flux2_klein_qnn_v81/transformer.pte
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("debug_quant")


def _load_portable_and_quant_ops():
    """Needed so the host ExecuTorch runtime can resolve dq-linear kernels."""
    from executorch.extension.pybindings import _portable_lib  # noqa: F401
    import site
    candidates = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        candidates.append(
            os.path.join(sp, "executorch", "kernels", "quantized",
                         "libquantized_ops_aot_lib.so")
        )
    for p in candidates:
        if os.path.isfile(p):
            torch.ops.load_library(p)
            log.info("loaded: %s", p)
            return
    log.warning("libquantized_ops_aot_lib.so not found — dq-linear calls may fail")


def _build_fixed_inputs(height=512, width=512, max_text_len=512, seed=0):
    """
    Build reproducible transformer inputs matching the export wrapper
    signature: (hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids).

    We construct these to be in-distribution enough that static-calibrated
    quantization exercises realistic activation ranges. Not a true pipeline
    trajectory — just reproducible and roughly the right scale.
    """
    from export_flux2_klein_xnnpack import (
        load_pipeline,
        _get_vae_scale_factor,
        _compute_latent_dims,
        _prepare_latent_ids_klein,
        _prepare_text_ids_klein,
    )

    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    vae_sf = _get_vae_scale_factor(pipe)
    patch_h, patch_w = _compute_latent_dims(height, width, vae_sf)

    in_channels = pipe.transformer.config.in_channels
    joint_dim = pipe.transformer.config.joint_attention_dim
    num_img_tokens = patch_h * patch_w

    g = torch.Generator(device="cpu").manual_seed(seed)
    hidden_states = torch.randn(1, num_img_tokens, in_channels, dtype=torch.float32, generator=g) * 0.7
    encoder_hidden_states = torch.randn(1, max_text_len, joint_dim, dtype=torch.float32, generator=g) * 0.02
    timestep = torch.tensor([1.0], dtype=torch.float32)
    img_ids = _prepare_latent_ids_klein(patch_h, patch_w, batch=1).to(torch.float32)
    txt_ids = _prepare_text_ids_klein(max_text_len, batch=1).to(torch.float32)

    log.info(
        "shapes: hs=%s ehs=%s ts=%s iid=%s tid=%s",
        list(hidden_states.shape), list(encoder_hidden_states.shape),
        list(timestep.shape), list(img_ids.shape), list(txt_ids.shape),
    )
    return pipe, (hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids)


def run_fp32_reference(pipe, sample_inputs):
    from export_flux2_klein_xnnpack import Flux2TransformerWrapper

    log.info("running fp32 reference forward …")
    t0 = time.time()
    model = Flux2TransformerWrapper(pipe.transformer).eval().cpu()
    with torch.no_grad():
        out = model(*sample_inputs)
    log.info("fp32 forward done in %.1fs, out shape=%s, mean=%+.4f std=%.4f",
             time.time() - t0, list(out.shape), out.mean().item(), out.std().item())
    return out


def run_pte_forward(pte_path, sample_inputs):
    from executorch.runtime import Runtime

    log.info("loading %s (%.2f GB) …", pte_path, os.path.getsize(pte_path) / 1e9)
    t0 = time.time()
    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")
    log.info("  load: %.1fs", time.time() - t0)

    t0 = time.time()
    outputs = method.execute(list(sample_inputs))
    out = outputs[0]
    if not isinstance(out, torch.Tensor):
        out = torch.as_tensor(out)
    log.info("  forward: %.1fs, out shape=%s, mean=%+.4f std=%.4f",
             time.time() - t0, list(out.shape), out.mean().item(), out.std().item())
    return out


def compare(ref: torch.Tensor, got: torch.Tensor, label: str):
    """Print correlation, per-token / per-channel / per-sample diagnostics."""
    assert ref.shape == got.shape, f"shape mismatch {ref.shape} vs {got.shape}"
    r = ref.detach().cpu().float().flatten()
    g = got.detach().cpu().float().flatten()

    # Correlation (Pearson)
    rm, gm = r.mean(), g.mean()
    corr = ((r - rm) * (g - gm)).sum() / (((r - rm).pow(2).sum().sqrt()) *
                                          ((g - gm).pow(2).sum().sqrt()) + 1e-12)

    # Error metrics
    abs_diff = (r - g).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    rms = (r - g).pow(2).mean().sqrt().item()
    ref_rms = r.pow(2).mean().sqrt().item()

    # Relative
    rel_rms = rms / (ref_rms + 1e-12)

    # SNR (signal-to-noise ratio, dB)
    snr_db = 10.0 * torch.log10(r.pow(2).mean() / ((r - g).pow(2).mean() + 1e-12)).item()

    print()
    print(f"=== {label} ===")
    print(f"  shape        : {list(ref.shape)}  ({ref.numel():,} elements)")
    print(f"  ref    : mean={rm.item():+.4f}  std={r.std().item():.4f}  rms={ref_rms:.4f}")
    print(f"  got    : mean={gm.item():+.4f}  std={g.std().item():.4f}  rms={g.pow(2).mean().sqrt().item():.4f}")
    print(f"  Pearson corr : {corr.item():+.6f}")
    print(f"  max abs diff : {max_abs:.4f}")
    print(f"  mean abs diff: {mean_abs:.4f}")
    print(f"  RMS diff     : {rms:.4f}")
    print(f"  relative RMS : {rel_rms:.4f}   ({rel_rms*100:.2f}%)")
    print(f"  SNR          : {snr_db:+.1f} dB")

    # Qualitative verdict
    if corr.item() > 0.99:
        print("  verdict: EXCELLENT (≈ reference quality)")
    elif corr.item() > 0.95:
        print("  verdict: GOOD (prompt-following likely intact)")
    elif corr.item() > 0.8:
        print("  verdict: DEGRADED (visible artifacts expected)")
    elif corr.item() > 0.3:
        print("  verdict: BAD (structure preserved, detail corrupted)")
    else:
        print("  verdict: NOISE (quantization recipe broken)")

    # Per-last-dim channel RMS — catches per-channel saturation.
    if ref.ndim >= 2:
        per_chan = (ref - got).pow(2).mean(dim=tuple(range(ref.ndim - 1))).sqrt()
        top = per_chan.topk(min(10, per_chan.numel())).indices
        print(f"  worst channels (last dim, by RMS):")
        for ci in top.tolist():
            c = ci
            print(f"    ch {c:5d} (vae_ch={c // 4}, patch_pos={c % 4})  "
                  f"rms={per_chan[c].item():.4f}  "
                  f"ref_abs_max={ref[..., c].abs().max().item():.4f}  "
                  f"got_abs_max={got[..., c].abs().max().item():.4f}  "
                  f"amp={got[..., c].abs().max().item() / max(ref[..., c].abs().max().item(), 1e-6):.3f}")

    # Per-token RMS — catches spatially-localized failure (bad patch position).
    if ref.ndim == 3 and ref.shape[1] >= 16:
        per_tok = (ref - got).pow(2).mean(dim=(0, 2)).sqrt()
        top = per_tok.topk(min(10, per_tok.numel())).indices
        print(f"  worst tokens (spatial position, by RMS):")
        for ti in top.tolist():
            # Assume 32x32 patch grid for 512x512 output (vae_sf=8, patch=2, so 32x32)
            y, x = ti // 32, ti % 32
            print(f"    tok {ti:5d} (y={y}, x={x})  rms={per_tok[ti].item():.4f}")

    # Per-VAE-channel aggregated (channels 0..3 -> vae_ch 0, 4..7 -> vae_ch 1, etc.).
    if ref.ndim == 3 and ref.shape[-1] % 4 == 0:
        n_vae = ref.shape[-1] // 4
        vae_view_ref = ref.reshape(*ref.shape[:-1], n_vae, 4)
        vae_view_got = got.reshape(*got.shape[:-1], n_vae, 4)
        per_vae = (vae_view_ref - vae_view_got).pow(2).mean(dim=(0, 1, 3)).sqrt()
        top = per_vae.topk(min(5, per_vae.numel())).indices
        print(f"  worst VAE channels (post-unpatchify, by RMS):")
        for vi in top.tolist():
            ref_amp = vae_view_ref[..., vi, :].abs().max().item()
            got_amp = vae_view_got[..., vi, :].abs().max().item()
            print(f"    vae_ch {vi:3d}  rms={per_vae[vi].item():.4f}  "
                  f"ref_abs_max={ref_amp:.4f}  got_abs_max={got_amp:.4f}")

    return {
        "corr": corr.item(),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rms": rms,
        "rel_rms": rel_rms,
        "snr_db": snr_db,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reference", "compare"], default="compare",
                    help="reference: save fp32 output. compare: run both and diff.")
    ap.add_argument("--pte", default=None, help=".pte to evaluate (required in compare mode unless --skip_pte)")
    ap.add_argument("--ref", default=None, help="cached fp32 reference .pt (optional; else recompute)")
    ap.add_argument("--ref_out", default=None, help="write fp32 reference to this .pt path")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--max_text_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _load_portable_and_quant_ops()

    # Load cached inputs + reference if available, else compute.
    ref_tensor = None
    sample_inputs = None

    if args.ref and os.path.isfile(args.ref):
        bundle = torch.load(args.ref, map_location="cpu", weights_only=False)
        sample_inputs = tuple(bundle["inputs"])
        ref_tensor = bundle["fp32_out"]
        log.info("loaded cached reference from %s", args.ref)

    pipe = None
    if ref_tensor is None:
        pipe, sample_inputs = _build_fixed_inputs(
            height=args.height, width=args.width,
            max_text_len=args.max_text_len, seed=args.seed,
        )
        ref_tensor = run_fp32_reference(pipe, sample_inputs)

    if args.ref_out:
        torch.save(
            {"inputs": list(sample_inputs), "fp32_out": ref_tensor.detach().cpu()},
            args.ref_out,
        )
        log.info("wrote reference bundle to %s", args.ref_out)

    if args.mode == "reference":
        return

    if args.pte is None:
        log.error("--pte is required in compare mode")
        sys.exit(2)
    if not os.path.isfile(args.pte):
        log.error("no such .pte: %s", args.pte)
        sys.exit(2)

    # Free pipeline before running PTE to save RAM.
    if pipe is not None:
        del pipe
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    got = run_pte_forward(args.pte, sample_inputs)
    compare(ref_tensor, got, label=os.path.basename(args.pte))


if __name__ == "__main__":
    main()
