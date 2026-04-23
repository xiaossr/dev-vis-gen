#!/usr/bin/env python3
"""
Collect real calibration data for QNN PTQ quantization of FLUX.2-klein-4B.

Runs the PyTorch pipeline on a set of prompts and captures intermediate
tensors at each stage (text encoder, transformer, VAE decoder).  Saves
them to disk so the QNN export script can use them instead of random data.

Usage
-----
    # Collect from default prompts:
    python collect_calibration_data.py --output_dir ./calibration_data

    # From a file (one prompt per line):
    python collect_calibration_data.py --prompt_file prompts.txt --output_dir ./calibration_data

    # Custom prompts inline:
    python collect_calibration_data.py --output_dir ./calibration_data \
        --prompts "a cat on a roof" "a painting of mountains" "a photo of a city"

    # Control number of transformer timesteps sampled:
    python collect_calibration_data.py --output_dir ./calibration_data --num_timesteps 2
"""

import argparse
import logging
import os

import torch
import torch.nn as nn

from export_flux2_klein_xnnpack import (
    Qwen3TextEncoderWrapper,
    Flux2TransformerWrapper,
    VAEDecoderWrapper,
    VAEEncoderWrapper,
    load_pipeline,
    build_text_encoder_inputs,
    _compute_latent_dims,
    _get_vae_scale_factor,
    _prepare_latent_ids_klein,
    _prepare_text_ids_klein,
    _free_memory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("calibration")

# Diverse default prompts covering different styles and subjects
DEFAULT_PROMPTS = [
    "a cat sitting on a windowsill at sunset",
    "a cyberpunk city skyline at night with neon lights",
    "a watercolor painting of sunflowers in a vase",
    "a photo of an astronaut riding a horse on mars",
    "a close-up portrait of a woman with freckles, studio lighting",
    "an oil painting of a stormy ocean with a lighthouse",
    "a cute robot serving coffee in a futuristic cafe",
    "a snowy mountain landscape with a wooden cabin",
    "a macro photograph of a butterfly on a flower",
    "abstract geometric shapes in vibrant colors",
]


def tokenize_prompt(pipe, prompt: str, max_text_len: int):
    """Tokenize a single prompt and return (input_ids, attention_mask)."""
    tokenizer = pipe.tokenizer
    encoding = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_text_len,
        truncation=True,
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["attention_mask"]


def collect_text_encoder_data(pipe, prompts, max_text_len, device="cpu"):
    """Collect real tokenized inputs for the text encoder."""
    logger.info("Collecting text encoder calibration data (%d prompts) ...",
                len(prompts))
    samples = []
    for i, prompt in enumerate(prompts):
        input_ids, attention_mask = tokenize_prompt(pipe, prompt, max_text_len)
        # Save on CPU for portability (export runs on CPU)
        samples.append((input_ids.cpu(), attention_mask.cpu()))
        logger.info("  [%d/%d] tokenized: %r", i + 1, len(prompts), prompt[:50])
    return samples


def collect_transformer_data(
    pipe,
    text_encoder_wrapper,
    prompts,
    height,
    width,
    max_text_len,
    num_timesteps=2,
    dtype=torch.float32,
    device="cpu",
):
    """Collect real transformer inputs by running text encoder + sampling timesteps.

    For each prompt, encodes it through the text encoder, then creates
    transformer inputs at several timesteps across the denoising schedule.
    """
    t_cfg = pipe.transformer.config
    in_channels = t_cfg.in_channels
    vae_sf = _get_vae_scale_factor(pipe)
    patch_h, patch_w = _compute_latent_dims(height, width, vae_sf)
    num_tokens = patch_h * patch_w

    # Build positional IDs (same for all prompts at a given resolution)
    img_ids = _prepare_latent_ids_klein(patch_h, patch_w, 1).to(dtype).to(device)
    txt_ids = _prepare_text_ids_klein(max_text_len, 1).to(dtype).to(device)

    # Sample timesteps from the scheduler's range
    # Use linearly spaced values in [0, 1] to cover the full range
    timestep_values = torch.linspace(0.0, 1.0, num_timesteps)

    logger.info(
        "Collecting transformer calibration data "
        "(%d prompts x %d timesteps = %d samples) ...",
        len(prompts), num_timesteps, len(prompts) * num_timesteps,
    )

    samples = []
    for i, prompt in enumerate(prompts):
        # Get real prompt embeddings from text encoder
        input_ids, attention_mask = tokenize_prompt(pipe, prompt, max_text_len)
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        with torch.no_grad():
            prompt_embeds = text_encoder_wrapper(input_ids, attention_mask)

        for j, t_val in enumerate(timestep_values):
            # Random noise latents (different per sample)
            hidden_states = torch.randn(1, num_tokens, in_channels, dtype=dtype, device=device)
            timestep = torch.full((1,), t_val.item(), dtype=dtype, device=device)

            # Save on CPU for portability
            samples.append((
                hidden_states.cpu(),
                prompt_embeds.clone().cpu(),
                timestep.cpu(),
                img_ids.clone().cpu(),
                txt_ids.clone().cpu(),
            ))

        logger.info("  [%d/%d] encoded: %r", i + 1, len(prompts), prompt[:50])

    return samples


def collect_vae_decoder_data(
    pipe,
    transformer_wrapper,
    text_encoder_wrapper,
    prompts,
    height,
    width,
    max_text_len,
    num_steps=4,
    dtype=torch.float32,
    device="cpu",
):
    """Collect real VAE decoder inputs by running a few denoising steps.

    Runs partial denoising for each prompt to produce realistic latents
    that the VAE decoder would actually see.
    """
    t_cfg = pipe.transformer.config
    in_channels = t_cfg.in_channels
    vae_cfg = pipe.vae.config
    latent_ch = getattr(vae_cfg, "latent_channels", 32)
    vae_sf = _get_vae_scale_factor(pipe)
    patch_h, patch_w = _compute_latent_dims(height, width, vae_sf)
    num_tokens = patch_h * patch_w
    latent_h, latent_w = patch_h * 2, patch_w * 2

    img_ids = _prepare_latent_ids_klein(patch_h, patch_w, 1).to(dtype).to(device)
    txt_ids = _prepare_text_ids_klein(max_text_len, 1).to(dtype).to(device)

    # Simple linear sigma schedule for denoising
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1)

    logger.info("Collecting VAE decoder calibration data (%d prompts) ...",
                len(prompts))

    samples = []
    for i, prompt in enumerate(prompts):
        input_ids, attention_mask = tokenize_prompt(pipe, prompt, max_text_len)
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        with torch.no_grad():
            prompt_embeds = text_encoder_wrapper(input_ids, attention_mask)

        # Run denoising to get realistic latents
        latents = torch.randn(1, num_tokens, in_channels, dtype=dtype, device=device)
        with torch.no_grad():
            for step in range(num_steps):
                sigma = sigmas[step]
                timestep = torch.full((1,), sigma.item(), dtype=dtype, device=device)
                noise_pred = transformer_wrapper(
                    latents, prompt_embeds, timestep, img_ids, txt_ids,
                )
                # Euler step
                dt = sigmas[step + 1] - sigmas[step]
                latents = latents + noise_pred * dt

        # Unpack from (B, N, C) to spatial (B, C, H, W) for VAE
        # Reverse the patchify: (B, patch_h*patch_w, C) -> (B, C, patch_h, patch_w)
        latents_spatial = latents.permute(0, 2, 1).reshape(
            1, in_channels, patch_h, patch_w
        )

        # Apply BN un-normalization BEFORE unpatchify (BN has in_channels=128)
        if hasattr(pipe.vae, "bn"):
            bn = pipe.vae.bn
            mean = bn.running_mean.view(1, -1, 1, 1).to(device)
            var = bn.running_var.view(1, -1, 1, 1).to(device)
            eps = getattr(pipe.vae.config, "batch_norm_eps", 1e-5)
            latents_spatial = latents_spatial * torch.sqrt(var + eps) + mean

        # Unpatchify 2x2: (B, C=128, patch_h, patch_w) -> (B, latent_ch=32, latent_h, latent_w)
        # in_channels=128 = latent_ch(32) * 4 (2x2 patches)
        latents_unpatch = latents_spatial.reshape(
            1, latent_ch, 4, patch_h, patch_w
        )
        latents_unpatch = latents_unpatch.reshape(
            1, latent_ch, 2, 2, patch_h, patch_w
        )
        latents_unpatch = latents_unpatch.permute(0, 1, 4, 2, 5, 3).reshape(
            1, latent_ch, latent_h, latent_w
        )

        # Save on CPU for portability
        samples.append((latents_unpatch.cpu(),))
        logger.info("  [%d/%d] denoised: %r", i + 1, len(prompts), prompt[:50])

    return samples


def save_calibration_data(samples, component_name, output_dir):
    """Save calibration samples as a list of tensor tuples."""
    path = os.path.join(output_dir, f"calibration_{component_name}.pt")
    torch.save(samples, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info("Saved %d samples -> %s (%.1f MB)", len(samples), path, size_mb)
    return path


def main():
    p = argparse.ArgumentParser(
        description="Collect calibration data for QNN PTQ of FLUX.2-klein-4B",
    )
    p.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-4B",
                    help="HuggingFace model ID or local path")
    p.add_argument("--output_dir", default="./calibration_data",
                    help="Directory for calibration tensors")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--max_text_len", type=int, default=512)
    p.add_argument("--prompts", nargs="+", default=None,
                    help="Prompts to use (overrides defaults)")
    p.add_argument("--prompt_file", type=str, default=None,
                    help="File with one prompt per line")
    p.add_argument("--num_timesteps", type=int, default=2,
                    help="Timesteps to sample per prompt for transformer calibration")
    p.add_argument("--num_steps", type=int, default=4,
                    help="Denoising steps for VAE calibration data")
    p.add_argument("--component",
                    choices=["all", "text_encoder", "transformer", "vae"],
                    default="all",
                    help="Collect data for which component(s)")
    p.add_argument("--device", type=str, default=None,
                    help="Device to run on (e.g. 'cuda', 'cuda:0'). "
                         "Auto-detects GPU if available.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = torch.float32

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # Resolve prompts
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.prompts:
        prompts = args.prompts
    else:
        prompts = DEFAULT_PROMPTS

    logger.info("Using %d prompts for calibration", len(prompts))

    # Load pipeline
    pipe = load_pipeline(args.model_id, dtype=dtype)
    pipe = pipe.to(device)

    hidden_states_layers = [9, 18, 27]
    te_wrapper = Qwen3TextEncoderWrapper(
        pipe.text_encoder, hidden_states_layers=hidden_states_layers,
    ).eval().to(device)

    # ---- Text Encoder calibration data ----
    if args.component in ("all", "text_encoder"):
        te_samples = collect_text_encoder_data(
            pipe, prompts, args.max_text_len, device=device,
        )
        save_calibration_data(te_samples, "text_encoder", args.output_dir)
        del te_samples
        _free_memory()

    # ---- Transformer calibration data ----
    if args.component in ("all", "transformer"):
        tf_samples = collect_transformer_data(
            pipe, te_wrapper, prompts,
            args.height, args.width, args.max_text_len,
            num_timesteps=args.num_timesteps, dtype=dtype, device=device,
        )
        save_calibration_data(tf_samples, "transformer", args.output_dir)
        del tf_samples
        _free_memory()

    # ---- VAE Decoder calibration data ----
    if args.component in ("all", "vae"):
        tf_wrapper = Flux2TransformerWrapper(pipe.transformer).eval().to(device)
        vae_samples = collect_vae_decoder_data(
            pipe, tf_wrapper, te_wrapper, prompts,
            args.height, args.width, args.max_text_len,
            num_steps=args.num_steps, dtype=dtype, device=device,
        )
        save_calibration_data(vae_samples, "vae", args.output_dir)
        del tf_wrapper, vae_samples
        _free_memory()

    del te_wrapper, pipe
    _free_memory()

    logger.info("Calibration data collection complete -> %s/", args.output_dir)


if __name__ == "__main__":
    main()
