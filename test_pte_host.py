#!/usr/bin/env python3
"""
Host-side smoke test for exported .pte files.

Loads each component with the ExecuTorch host Python runtime, runs one
forward pass with dummy tensors, and checks output shape. This verifies
that the .pte files are well-formed and that the delegate (XNNPACK)
actually links/initialises before we push to the phone.

Requires: executorch + numpy available in .venv.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def _load_quantized_ops_lib():
    """Register quantized_decomposed::* kernels with the host runtime.

    The text_encoder uses embedding_byte (from --embedding_quantize 8), and
    Runtime.get() won't find that kernel unless libquantized_ops_aot_lib.so
    is loaded first. That library in turn dlopens _portable_lib.so via a
    runpath relative to an activated venv, which fails under direct
    interpreter launch — so we import _portable_lib here first, which
    satisfies the symbol resolution in the process.
    """
    from executorch.extension.pybindings import _portable_lib  # noqa: F401
    import site
    candidates = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        candidates.append(os.path.join(sp, "executorch", "kernels", "quantized",
                                       "libquantized_ops_aot_lib.so"))
    for p in candidates:
        if os.path.isfile(p):
            torch.ops.load_library(p)
            print(f"loaded: {p}")
            return
    print(f"warn: libquantized_ops_aot_lib.so not found in {candidates}")


_load_quantized_ops_lib()

from executorch.runtime import Runtime


def load_and_run(pte_path: str, inputs):
    print(f"\n--- {os.path.basename(pte_path)} ({os.path.getsize(pte_path)/1e9:.2f} GB) ---")
    t0 = time.time()
    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")
    print(f"  load: {time.time()-t0:.1f}s")

    t0 = time.time()
    outputs = method.execute(inputs)
    dt = time.time() - t0
    print(f"  forward: {dt:.1f}s")

    for i, out in enumerate(outputs):
        shape = tuple(out.shape)
        dtype = out.dtype
        print(f"  out[{i}]: shape={shape} dtype={dtype}")
        arr = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
        print(f"         min={arr.min():.4f} max={arr.max():.4f} "
              f"mean={arr.mean():.4f} std={arr.std():.4f}")
    return outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--component",
                   choices=["all", "text_encoder", "transformer", "vae_decoder"],
                   default="all")
    args = p.parse_args()

    cfg_path = os.path.join(args.model_dir, "export_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    max_text_len = cfg["text_encoder"]["max_sequence_length"]
    in_ch = cfg["transformer"]["in_channels"]
    joint_dim = cfg["transformer"]["joint_attention_dim"]
    latent_ch = cfg["vae"]["latent_channels"]
    patch_h, patch_w = cfg["patch_dims"]
    num_tokens = patch_h * patch_w

    torch.manual_seed(42)

    if args.component in ("all", "text_encoder"):
        te_path = os.path.join(args.model_dir, "text_encoder.pte")
        if os.path.exists(te_path):
            ids = torch.ones(1, max_text_len, dtype=torch.long)
            mask = torch.ones(1, max_text_len, dtype=torch.long)
            load_and_run(te_path, (ids, mask))
        else:
            print(f"missing: {te_path}")

    if args.component in ("all", "transformer"):
        tf_path = os.path.join(args.model_dir, "transformer.pte")
        if os.path.exists(tf_path):
            hs = torch.randn(1, num_tokens, in_ch)
            ehs = torch.randn(1, max_text_len, joint_dim)
            ts = torch.tensor([0.5])
            iid = torch.zeros(1, num_tokens, 4)
            tid = torch.zeros(1, max_text_len, 4)
            load_and_run(tf_path, (hs, ehs, ts, iid, tid))
        else:
            print(f"missing: {tf_path}")

    if args.component in ("all", "vae_decoder"):
        vae_path = os.path.join(args.model_dir, "vae_decoder.pte")
        if os.path.exists(vae_path):
            lat = torch.randn(1, latent_ch, patch_h * 2, patch_w * 2)
            load_and_run(vae_path, (lat,))
        else:
            print(f"missing: {vae_path}")


if __name__ == "__main__":
    main()
