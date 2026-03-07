#!/usr/bin/env python3
"""
FLUX.2-klein-4B inference using ExecuTorch XNNPACK .pte files.

Pipeline:  text_encoder.pte → transformer.pte → vae_decoder.pte

Supports:
  - Text-to-image generation
  - Image-to-image editing (requires vae_encoder.pte)
  - Classifier-free guidance (for non-distilled Klein variants)

Usage
-----
    # Text-to-image:
    python run_flux2_klein_xnnpack.py \\
        --model_dir ./exported_flux2_klein \\
        --prompt "a cat sitting on a windowsill at sunset" \\
        --output output.png

    # Image-to-image editing:
    python run_flux2_klein_xnnpack.py \\
        --model_dir ./exported_flux2_klein \\
        --prompt "a cat sitting on a windowsill at sunset" \\
        --image reference.png \\
        --output edited.png
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("flux2_infer")


# ============================================================================
# 1.  Load ExecuTorch .pte models
# ============================================================================

def load_pte_model(path: str):
    """Load an ExecuTorch .pte program via the Python bindings."""
    from executorch.runtime import Runtime

    logger.info("Loading %s …", path)
    runtime = Runtime.get()
    program = runtime.load_program(path)
    method = program.load_method("forward")
    logger.info("  loaded (%s)", path)
    return method


# ============================================================================
# 2.  Positional-ID helpers  (match pipeline_flux2_klein.py exactly)
# ============================================================================

def prepare_latent_ids(patch_h: int, patch_w: int, batch: int = 1) -> torch.Tensor:
    """4D (T, H, W, L) positional IDs for image latent tokens.

    Input dims are *post-patchify* (i.e. latent_h//2, latent_w//2).
    Returns (B, patch_h * patch_w, 4) int64.
    """
    t = torch.arange(1)       # [0]
    h = torch.arange(patch_h)
    w = torch.arange(patch_w)
    l = torch.arange(1)       # [0]
    ids = torch.cartesian_prod(t, h, w, l)                 # (patch_h*patch_w, 4)
    return ids.unsqueeze(0).expand(batch, -1, -1)           # (B, N, 4)


def prepare_image_ids(patch_h: int, patch_w: int, num_images: int,
                      scale: int = 10, batch: int = 1) -> torch.Tensor:
    """4D positional IDs for reference-image latent tokens (img2img).

    Each reference image gets a unique T-coordinate: scale, 2*scale, …
    Returns (B, num_images * patch_h * patch_w, 4) int64.
    """
    id_list = []
    for img_idx in range(num_images):
        t_val = scale + scale * img_idx
        t = torch.tensor([t_val])
        h = torch.arange(patch_h)
        w = torch.arange(patch_w)
        l = torch.arange(1)
        ids = torch.cartesian_prod(t, h, w, l)
        id_list.append(ids)
    all_ids = torch.cat(id_list, dim=0)                     # (N_total, 4)
    return all_ids.unsqueeze(0).expand(batch, -1, -1)       # (B, N_total, 4)


def prepare_text_ids(seq_len: int, batch: int = 1) -> torch.Tensor:
    """4D (T, H, W, L) positional IDs for text tokens.

    Returns (B, seq_len, 4) int64.
    """
    out = []
    for _ in range(batch):
        t = torch.arange(1)
        h = torch.arange(1)
        w = torch.arange(1)
        s = torch.arange(seq_len)
        out.append(torch.cartesian_prod(t, h, w, s))       # (seq_len, 4)
    return torch.stack(out)                                 # (B, seq_len, 4)


def compute_latent_dims(height: int, width: int, vae_sf: int):
    """Compute post-patchify latent dims matching the pipeline exactly.

    Returns (patch_h, patch_w).
    """
    latent_h = 2 * (height // (vae_sf * 2))
    latent_w = 2 * (width  // (vae_sf * 2))
    return latent_h // 2, latent_w // 2


# ============================================================================
# 3.  Latent packing / unpacking  (match pipeline_flux2_klein.py exactly)
# ============================================================================

def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) → (B, C*4, H//2, W//2)"""
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(B, C * 4, H // 2, W // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C*4, H//2, W//2) → (B, C, H, W)"""
    B, C4, H2, W2 = latents.shape
    C = C4 // 4
    latents = latents.reshape(B, C, 2, 2, H2, W2)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(B, C, H2 * 2, W2 * 2)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) → (B, H*W, C)"""
    B, C, H, W = latents.shape
    return latents.reshape(B, C, H * W).permute(0, 2, 1)


def unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
    """Scatter packed tokens back to spatial form using positional IDs.

    x:     (B, N, C)
    x_ids: (B, N, 4)  — columns are (T, H, W, L)
    Returns (B, C, H, W).
    """
    out_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1
        flat = h_ids * w + w_ids
        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat.unsqueeze(1).expand(-1, ch), data)
        out = out.view(h, w, ch).permute(2, 0, 1)          # (C, H, W)
        out_list.append(out)
    return torch.stack(out_list, dim=0)                     # (B, C, H, W)


# ============================================================================
# 4.  Image utilities
# ============================================================================

def latents_to_pil(pixel_values: torch.Tensor) -> Image.Image:
    """Convert a (1, 3, H, W) float tensor in [-1, 1] to a PIL Image."""
    img = pixel_values[0].clamp(-1, 1).permute(1, 2, 0)    # (H, W, 3)
    img = ((img + 1.0) / 2.0 * 255.0).to(torch.uint8).cpu().numpy()
    return Image.fromarray(img)


def load_and_preprocess_image(path: str, height: int, width: int,
                              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Load an image, resize, and normalise to [-1, 1] for VAE encoding."""
    img = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)   # (1, 3, H, W)
    tensor = (tensor * 2.0 - 1.0).to(dtype)
    return tensor


# ============================================================================
# 5.  Batch-norm helpers (for VAE latent normalisation / un-normalisation)
# ============================================================================

def load_bn_stats(model_dir: Path, dtype: torch.dtype = torch.float32):
    """Load VAE batch-norm running stats from vae_bn_stats.pt."""
    bn_path = model_dir / "vae_bn_stats.pt"
    if not bn_path.exists():
        return None, None
    stats = torch.load(bn_path, map_location="cpu", weights_only=True)
    mean = stats["running_mean"].view(1, -1, 1, 1).to(dtype)
    var  = stats["running_var"].view(1, -1, 1, 1).to(dtype)
    return mean, var


def bn_normalise(latents: torch.Tensor, mean: torch.Tensor, var: torch.Tensor,
                 eps: float = 1e-5) -> torch.Tensor:
    """Apply batch-norm normalisation (for VAE encoding in img2img)."""
    std = torch.sqrt(var + eps)
    return (latents - mean) / std


def bn_unnormalise(latents: torch.Tensor, mean: torch.Tensor, var: torch.Tensor,
                   eps: float = 1e-5) -> torch.Tensor:
    """Undo batch-norm normalisation (for VAE decoding after denoising)."""
    std = torch.sqrt(var + eps)
    return latents * std + mean


# ============================================================================
# 6.  Text embeddings
# ============================================================================

def encode_prompt_ondevice(
    text_encoder_method,
    tokenizer,
    prompt: str,
    max_sequence_length: int,
    dtype: torch.dtype = torch.float32,
):
    """Tokenize a prompt and run the text_encoder.pte on-device.

    Tokenization matches the pipeline's _get_qwen3_prompt_embeds exactly:
    apply_chat_template → tokenize with max_length padding/truncation.
    Returns (prompt_embeds, txt_ids).
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
    )
    input_ids = inputs["input_ids"]            # (1, max_seq_len)
    attention_mask = inputs["attention_mask"]   # (1, max_seq_len)

    t0 = time.perf_counter()
    prompt_embeds = text_encoder_method.execute([
        input_ids.contiguous(),
        attention_mask.contiguous(),
    ])
    if isinstance(prompt_embeds, (list, tuple)):
        prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.to(dtype)
    dt = time.perf_counter() - t0
    logger.info("Text encoder: %.2f s → %s", dt, prompt_embeds.shape)

    txt_ids = prepare_text_ids(prompt_embeds.shape[1], batch=1).to(dtype)
    return prompt_embeds, txt_ids



# ============================================================================
# 7.  Image encoding (on-device via vae_encoder.pte, for img2img)
# ============================================================================

def encode_image_ondevice(
    vae_encoder_method,
    image_path: str,
    height: int,
    width: int,
    patch_h: int,
    patch_w: int,
    bn_mean: torch.Tensor,
    bn_var: torch.Tensor,
    bn_eps: float,
    dtype: torch.dtype = torch.float32,
):
    """Encode a reference image → patchify → BN-normalise → pack.

    Returns (image_latents, image_ids) ready to concatenate with noise tokens.
    """
    pixel_values = load_and_preprocess_image(image_path, height, width, dtype)
    logger.info("Image preprocessed: %s → %s", image_path, pixel_values.shape)

    # VAE encode
    t0 = time.perf_counter()
    raw_latents = vae_encoder_method.execute([pixel_values.contiguous()])
    if isinstance(raw_latents, (list, tuple)):
        raw_latents = raw_latents[0]
    dt = time.perf_counter() - t0
    raw_latents = raw_latents.to(dtype)
    logger.info("VAE encode: %.2f s → %s", dt, raw_latents.shape)

    # Patchify
    patched = patchify_latents(raw_latents)

    # BN-normalise (matches _encode_vae_image in pipeline)
    if bn_mean is not None and bn_var is not None:
        patched = bn_normalise(patched, bn_mean, bn_var, bn_eps)

    # Pack to sequence
    packed = pack_latents(patched)                          # (1, N, C)

    # Image positional IDs (T=10 for first image)
    image_ids = prepare_image_ids(patch_h, patch_w, num_images=1, batch=1)

    return packed, image_ids


# ============================================================================
# 8.  Sigma schedule (matches FlowMatchEulerDiscreteScheduler)
# ============================================================================

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute mu for time-shift sigma warping (from diffusers pipeline source)."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10  = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def time_shift_sigmas(sigmas: np.ndarray, mu: float) -> np.ndarray:
    """Apply exponential time-shift to sigma schedule."""
    return np.exp(mu) / (np.exp(mu) + (1.0 / sigmas - 1.0))


def build_sigma_schedule(num_steps: int, image_seq_len: int) -> list:
    """Build the full sigma schedule matching FlowMatchEulerDiscreteScheduler.

    The Klein scheduler applies resolution-dependent time-shift (mu) to
    the raw linspace sigmas.  Flow matching models are very sensitive to
    this — using unshifted sigmas produces garbage.

    Returns a list of (num_steps + 1) sigma values, ending with 0.0.
    """
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    mu = compute_empirical_mu(image_seq_len, num_steps)
    sigmas = time_shift_sigmas(sigmas, mu)
    result = sigmas.tolist()
    result.append(0.0)
    return result


# ============================================================================
# 9.  Denoising loop
# ============================================================================

def denoise(
    transformer_method,
    prompt_embeds: torch.Tensor,
    txt_ids: torch.Tensor,
    patch_h: int,
    patch_w: int,
    in_channels: int,
    num_steps: int = 4,
    guidance_scale: float = 1.0,
    is_distilled: bool = True,
    negative_prompt_embeds: torch.Tensor | None = None,
    negative_txt_ids: torch.Tensor | None = None,
    image_latents: torch.Tensor | None = None,
    image_ids: torch.Tensor | None = None,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> tuple:
    """Run the flow-matching Euler denoising loop.

    Supports:
      - Text-to-image (image_latents=None)
      - Image-to-image (image_latents concatenated with noise tokens)
      - Classifier-free guidance (guidance_scale > 1.0, non-distilled only)

    Returns (latents, latent_ids) where latents is (B, N, C) packed tokens
    and latent_ids is (B, N, 4) — only noise tokens, not image tokens.
    """
    batch = 1
    num_latent_channels = in_channels // 4
    num_noise_tokens = patch_h * patch_w

    # ---- initial noise -------------------------------------------------
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        batch, num_latent_channels * 4, patch_h, patch_w,
        dtype=dtype, generator=generator,
    )
    latent_ids = prepare_latent_ids(patch_h, patch_w, batch)
    latents = pack_latents(noise)  # (B, num_noise_tokens, in_channels)

    # ---- sigma schedule with time-shift --------------------------------
    sigmas = build_sigma_schedule(num_steps, num_noise_tokens)

    # ---- prepare positional IDs ----------------------------------------
    noise_ids_float = latent_ids.to(dtype)

    # For img2img: concatenate image tokens with noise tokens
    if image_latents is not None and image_ids is not None:
        combined_img_ids = torch.cat(
            [noise_ids_float, image_ids.to(dtype)], dim=1
        )
    else:
        combined_img_ids = noise_ids_float

    txt_ids_float = txt_ids.to(dtype)

    logger.info(
        "Denoising: %d steps, noise_tokens=%d, in_channels=%d, patch=(%d×%d)%s",
        num_steps, num_noise_tokens, in_channels, patch_h, patch_w,
        f", img2img_tokens={image_latents.shape[1]}" if image_latents is not None else "",
    )

    for i in range(num_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # The pipeline passes t/1000 where t = sigma*1000 (scheduler
        # timestep), so the transformer receives the raw sigma.  The
        # transformer internally does *1000 (baked into the .pte).
        timestep = torch.full((batch,), sigma, dtype=dtype)

        # Build hidden_states: noise tokens [+ image tokens for img2img]
        if image_latents is not None:
            hidden_states = torch.cat([latents, image_latents], dim=1)
        else:
            hidden_states = latents

        t0 = time.perf_counter()
        noise_pred = transformer_method.execute([
            hidden_states.contiguous(),
            prompt_embeds.contiguous(),
            timestep.contiguous(),
            combined_img_ids.contiguous(),
            txt_ids_float.contiguous(),
        ])
        if isinstance(noise_pred, (list, tuple)):
            noise_pred = noise_pred[0]

        # Only keep predictions for noise tokens (strip image-condition tokens)
        noise_pred = noise_pred[:, :num_noise_tokens, :]

        elapsed = time.perf_counter() - t0

        # Euler step: x_{t-1} = x_t + (sigma_next - sigma) * v_pred
        latents = latents + (sigma_next - sigma) * noise_pred

        logger.info("  step %d/%d  σ=%.4f→%.4f  (%.2f s)",
                     i + 1, num_steps, sigma, sigma_next, elapsed)

    return latents, latent_ids


# ============================================================================
# 10.  Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Fully on-device FLUX.2-klein-4B inference (ExecuTorch XNNPACK)",
    )
    p.add_argument("--model_dir", required=True,
                    help="Directory containing .pte files and export_config.json")
    p.add_argument("--prompt", required=True,
                    help="Text prompt")
    p.add_argument("--image", default=None,
                    help="Reference image path for image-to-image editing")
    p.add_argument("--guidance_scale", type=float, default=None,
                    help="CFG scale (default: from config; >1.0 enables CFG for non-distilled)")
    p.add_argument("--output", default="generated.png",
                    help="Output image path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_steps", type=int, default=None,
                    help="Override number of denoising steps (default: from config)")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    dtype = torch.float32

    # ---- load config ---------------------------------------------------
    cfg_path = model_dir / "export_config.json"
    if not cfg_path.exists():
        logger.error("Missing %s – run export_flux2_klein_xnnpack.py first.", cfg_path)
        return
    with open(cfg_path) as f:
        cfg = json.load(f)

    height         = cfg["height"]
    width          = cfg["width"]
    vae_sf         = cfg["vae_scale_factor"]
    in_channels    = cfg["transformer"]["in_channels"]
    is_distilled   = cfg.get("is_distilled", True)
    num_steps      = args.num_steps or cfg.get("num_inference_steps", 4)
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else cfg.get("guidance_scale", 1.0)
    bn_eps         = cfg.get("vae", {}).get("batch_norm_eps", 1e-5)
    patch_h, patch_w = compute_latent_dims(height, width, vae_sf)

    if guidance_scale > 1.0 and is_distilled:
        logger.warning(
            "guidance_scale=%.1f ignored for distilled model (is_distilled=True). "
            "Using guidance_scale=1.0.", guidance_scale
        )
        guidance_scale = 1.0

    # ---- load BN stats -------------------------------------------------
    bn_mean, bn_var = load_bn_stats(model_dir, dtype)
    if bn_mean is not None:
        logger.info("VAE batch-norm stats loaded")
    else:
        logger.warning("vae_bn_stats.pt not found — BN un-normalisation will be skipped")

    # ---- load .pte models ----------------------------------------------
    # Use transformer_img2img.pte for image editing, transformer.pte for t2i
    if args.image:
        transformer_pte = model_dir / "transformer_img2img.pte"
        if not transformer_pte.exists():
            logger.error(
                "Missing %s — re-export with --num_img2img_images 1 to "
                "enable image editing.", transformer_pte,
            )
            return
    else:
        transformer_pte = model_dir / "transformer.pte"

    required_files = [transformer_pte, model_dir / "vae_decoder.pte"]
    if args.image:
        required_files.append(model_dir / "vae_encoder.pte")

    for path in required_files:
        if not path.exists():
            logger.error("Missing %s", path)
            return

    transformer  = load_pte_model(str(transformer_pte))
    vae_decoder  = load_pte_model(str(model_dir / "vae_decoder.pte"))
    vae_encoder  = load_pte_model(str(model_dir / "vae_encoder.pte")) if args.image else None

    # Load text encoder .pte for on-device encoding
    text_encoder_pte = model_dir / "text_encoder.pte"
    if not text_encoder_pte.exists():
        logger.error("Missing %s — re-run export with: --component text_encoder", text_encoder_pte)
        return
    text_encoder = load_pte_model(str(text_encoder_pte))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    logger.info("Text encoder .pte loaded")

    # ---- 1. text embeddings --------------------------------------------
    logger.info("=" * 50)
    logger.info("TEXT EMBEDDINGS")
    logger.info("=" * 50)
    max_seq_len = cfg.get("text_encoder", {}).get("max_sequence_length", 512)

    prompt_embeds, txt_ids = encode_prompt_ondevice(
        text_encoder, tokenizer, args.prompt, max_seq_len, dtype,
    )

    # Negative prompt for CFG
    negative_prompt_embeds, negative_txt_ids = None, None
    if guidance_scale > 1.0 and not is_distilled:
        negative_prompt_embeds, negative_txt_ids = encode_prompt_ondevice(
            text_encoder, tokenizer, "", max_seq_len, dtype,
        )

    # ---- 2. encode reference image (img2img) ---------------------------
    image_latents, image_ids = None, None
    if args.image:
        logger.info("=" * 50)
        logger.info("IMAGE ENCODING (img2img)")
        logger.info("=" * 50)
        if bn_mean is None:
            logger.error("Image-to-image requires vae_bn_stats.pt for BN normalisation.")
            return
        image_latents, image_ids = encode_image_ondevice(
            vae_encoder, args.image, height, width,
            patch_h, patch_w, bn_mean, bn_var, bn_eps, dtype,
        )

    # ---- 3. denoise ----------------------------------------------------
    logger.info("=" * 50)
    logger.info("DENOISING")
    logger.info("=" * 50)
    latents, latent_ids = denoise(
        transformer_method=transformer,
        prompt_embeds=prompt_embeds,
        txt_ids=txt_ids,
        patch_h=patch_h,
        patch_w=patch_w,
        in_channels=in_channels,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        is_distilled=is_distilled,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_txt_ids=negative_txt_ids,
        image_latents=image_latents,
        image_ids=image_ids,
        seed=args.seed,
        dtype=dtype,
    )

    # ---- 4. unpack + un-normalise + unpatchify -------------------------
    logger.info("=" * 50)
    logger.info("VAE DECODE")
    logger.info("=" * 50)

    # Unpack from (B, N, C) → spatial (B, C, patch_h, patch_w) using IDs
    latents_spatial = unpack_latents_with_ids(latents, latent_ids)
    logger.info("Unpacked latents: %s", latents_spatial.shape)

    # Batch-norm un-normalisation (matches pipeline post-denoising code)
    if bn_mean is not None and bn_var is not None:
        latents_spatial = bn_unnormalise(latents_spatial, bn_mean, bn_var, bn_eps)
        logger.info("Applied batch-norm un-normalisation")

    # Unpatchify: (B, C*4, H//2, W//2) → (B, C, H, W)
    latents_unpatch = unpatchify_latents(latents_spatial)
    logger.info("Unpatchified latents: %s", latents_unpatch.shape)

    # ---- 5. VAE decode -------------------------------------------------
    t0 = time.perf_counter()
    pixel_values = vae_decoder.execute([latents_unpatch.contiguous()])
    if isinstance(pixel_values, (list, tuple)):
        pixel_values = pixel_values[0]
    dt = time.perf_counter() - t0
    logger.info("VAE decode: %.2f s → %s", dt, pixel_values.shape)

    # ---- 6. save image -------------------------------------------------
    image = latents_to_pil(pixel_values)
    image.save(args.output)
    logger.info("Saved %s (%d×%d)", args.output, image.width, image.height)
    print(f"\n  Output: {args.output}  ({image.width}×{image.height})")


if __name__ == "__main__":
    main()
