#!/usr/bin/env python3
"""
Prepare inputs for the C++ Flux2 on-device runner.

This script converts Python-side artefacts into simple binary formats that the
C++ runner (flux2_main) can load directly, without any Python / PyTorch / JSON
dependencies at runtime.

What it produces
----------------
1. ``prompt.bin``  – tokenized prompt (int64 IDs + attention mask)
2. ``bn_mean.bin`` – VAE batch-norm running mean  (float32)
3. ``bn_var.bin``  – VAE batch-norm running var    (float32)

Usage
-----
    python prepare_mobile.py \\
        --model_dir ./exported_flux2_klein \\
        --prompt "a cat sitting on a windowsill at sunset" \\
        --output_dir ./mobile_bundle
"""

import argparse
import json
import os
import struct

import numpy as np
import torch
from transformers import AutoTokenizer


def tokenize_prompt(tokenizer, prompt: str, max_seq_len: int):
    """Tokenize matching the pipeline's _get_qwen3_prompt_embeds logic."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )
    return inputs["input_ids"][0].astype(np.int64), inputs["attention_mask"][0].astype(np.int64)


def save_token_bin(path: str, input_ids: np.ndarray, attention_mask: np.ndarray):
    seq_len = len(input_ids)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", seq_len))
        f.write(input_ids.tobytes())
        f.write(attention_mask.tobytes())
    print(f"  {path}  (seq_len={seq_len}, non-pad={attention_mask.sum()})")


def save_float_bin(path: str, data: np.ndarray):
    data = data.astype(np.float32).ravel()
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(data)))
        f.write(data.tobytes())
    print(f"  {path}  ({len(data)} floats)")


def main():
    p = argparse.ArgumentParser(
        description="Prepare binary inputs for the C++ Flux2 on-device runner",
    )
    p.add_argument("--model_dir", required=True,
                   help="Exported model directory (with tokenizer/, vae_bn_stats.pt, export_config.json)")
    p.add_argument("--prompt", required=True,
                   help="Text prompt to tokenize")
    p.add_argument("--output_dir", default=None,
                   help="Where to write binary files (default: same as model_dir)")
    args = p.parse_args()

    model_dir = args.model_dir
    out_dir = args.output_dir or model_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---- read config for max_seq_len ------------------------------------
    cfg_path = os.path.join(model_dir, "export_config.json")
    max_seq_len = 512
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        max_seq_len = cfg.get("text_encoder", {}).get("max_sequence_length", 512)
    print(f"max_seq_len = {max_seq_len}")

    # ---- tokenize prompt ------------------------------------------------
    print("\n[1/2] Tokenizing prompt ...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    input_ids, attention_mask = tokenize_prompt(tokenizer, args.prompt, max_seq_len)
    save_token_bin(os.path.join(out_dir, "prompt.bin"), input_ids, attention_mask)

    # ---- convert BN stats to flat binary --------------------------------
    bn_pt = os.path.join(model_dir, "vae_bn_stats.pt")
    if os.path.exists(bn_pt):
        print("\n[2/2] Converting BN stats ...")
        stats = torch.load(bn_pt, map_location="cpu", weights_only=True)
        save_float_bin(os.path.join(out_dir, "bn_mean.bin"),
                       stats["running_mean"].numpy())
        save_float_bin(os.path.join(out_dir, "bn_var.bin"),
                       stats["running_var"].numpy())
    else:
        print("\n[2/2] vae_bn_stats.pt not found — skipping BN conversion")

    # ---- summary --------------------------------------------------------
    print(f"\nDone. Binary files written to {out_dir}/")
    print("Next steps:")
    print(f"  adb push {out_dir}/*.bin  /data/local/tmp/flux2/")
    print(f"  adb push {model_dir}/*.pte /data/local/tmp/flux2/")


if __name__ == "__main__":
    main()
