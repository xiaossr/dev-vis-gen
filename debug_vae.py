#!/usr/bin/env python3
"""
Debug script for VAE decoder generation in FLUX.2-klein pipeline.

This script isolates and tests just the VAE decoder component:
1. Loads the full FLUX.2-klein pipeline
2. Generates latents by running the pipeline with a prompt
3. Runs the VAE decoder on those latents
4. Saves the output image
5. Optionally tests the exported .pte VAE decoder

Usage:
    # Test PyTorch VAE decoder with default prompt
    python debug_vae.py

    # Test with custom prompt
    python debug_vae.py --prompt "a dog in a field"

    # Test both PyTorch and exported .pte VAE decoder
    python debug_vae.py --test_pte --pte_path ./exported_flux2_klein/vae_decoder.pte

    # Use custom resolution and inference steps
    python debug_vae.py --height 512 --width 512 --num_inference_steps 4

    # Save latents for later reuse
    python debug_vae.py --save_latents

    # Load specific latents from file (skip pipeline generation)
    python debug_vae.py --latents_path ./vae_debug_output/test_latents.pt
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vae_debug")


class VAEDecoderWrapper(nn.Module):
    """Wraps AutoencoderKLFlux2 for decode-only export."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents, return_dict=False)[0]


def load_pipeline(model_id: str = "black-forest-labs/FLUX.2-klein-4B"):
    """Load the full FLUX.2-klein pipeline."""
    from diffusers import Flux2KleinPipeline
    
    logger.info("Loading pipeline from '%s' ...", model_id)
    pipe = Flux2KleinPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float32
    )
    logger.info("Pipeline loaded successfully")
    logger.info("VAE config: latent_channels=%s, scaling_factor=%s, shift_factor=%s",
                getattr(pipe.vae.config, "latent_channels", None),
                getattr(pipe.vae.config, "scaling_factor", None),
                getattr(pipe.vae.config, "shift_factor", None))
    
    # Get batch norm stats if available
    bn_stats = None
    if hasattr(pipe.vae, "bn"):
        bn_stats = {
            "running_mean": pipe.vae.bn.running_mean.detach().cpu().float(),
            "running_var": pipe.vae.bn.running_var.detach().cpu().float(),
        }
        logger.info("VAE batch-norm stats: mean shape=%s, var shape=%s",
                    bn_stats["running_mean"].shape,
                    bn_stats["running_var"].shape)
    
    return pipe, bn_stats


def get_vae_scale_factor(vae) -> int:
    """Return the spatial down-scale factor of the VAE."""
    vae_cfg = vae.config
    if hasattr(vae_cfg, "block_out_channels"):
        return 2 ** (len(vae_cfg.block_out_channels) - 1)
    return 8


def compute_latent_dims(height: int, width: int, vae_sf: int):
    """Compute latent dimensions matching the pipeline.
    
    Pipeline does: height = 2 * (int(height) // (vae_scale_factor * 2))
    Returns (latent_h, latent_w) after patchification.
    """
    latent_h = 2 * (height // (vae_sf * 2))
    latent_w = 2 * (width // (vae_sf * 2))
    # These are the un-patchified dims that go to VAE decoder
    return latent_h, latent_w


def generate_latents_from_pipeline(pipe, prompt: str, height: int, width: int, 
                                   num_inference_steps: int = 4, seed: int = 42):
    """Generate latents by running the FLUX.2-klein pipeline.
    
    This runs the full diffusion process and captures the final latents
    before VAE decoding, giving us realistic latents to test the VAE with.
    """
    logger.info("Generating latents from pipeline:")
    logger.info("  Prompt: %r", prompt)
    logger.info("  Image size: %dx%d", height, width)
    logger.info("  Inference steps: %d", num_inference_steps)
    logger.info("  Seed: %d", seed)
    
    # Set up generator for reproducibility
    generator = torch.Generator().manual_seed(seed)
    
    # We'll monkey-patch the VAE decode to capture latents
    original_decode = pipe.vae.decode
    captured_latents = []
    
    def capture_decode(latents, *args, **kwargs):
        captured_latents.append(latents.detach().cpu().clone())
        return original_decode(latents, *args, **kwargs)
    
    pipe.vae.decode = capture_decode
    
    try:
        logger.info("Running pipeline inference...")
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
        
        logger.info("Pipeline inference complete")
        
        if captured_latents:
            latents = captured_latents[0]
            logger.info("  Captured latent shape: %s", latents.shape)
            logger.info("  Latent stats: min=%.3f, max=%.3f, mean=%.3f, std=%.3f",
                        latents.min(), latents.max(), latents.mean(), latents.std())
            return latents, image
        else:
            logger.error("Failed to capture latents from pipeline")
            return None, image
    finally:
        # Restore original decode
        pipe.vae.decode = original_decode


def decode_with_pytorch_vae(vae, latents: torch.Tensor):
    """Decode latents using PyTorch VAE."""
    logger.info("Decoding with PyTorch VAE ...")
    
    vae.eval()
    with torch.no_grad():
        output = vae.decode(latents, return_dict=False)[0]
    
    logger.info("  Output shape: %s", output.shape)
    logger.info("  Output stats: min=%.3f, max=%.3f, mean=%.3f, std=%.3f",
                output.min(), output.max(), output.mean(), output.std())
    
    return output


def decode_with_pte_vae(pte_path: str, latents: torch.Tensor):
    """Decode latents using exported .pte VAE decoder."""
    logger.info("Decoding with ExecuTorch .pte VAE ...")
    
    try:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
    except ImportError:
        logger.error("ExecuTorch not available. Install with: pip install executorch")
        return None
    
    if not os.path.exists(pte_path):
        logger.error("PTE file not found: %s", pte_path)
        return None
    
    logger.info("  Loading %s ...", pte_path)
    module = _load_for_executorch(pte_path)
    
    logger.info("  Running inference ...")
    output = module.forward((latents,))[0]
    
    logger.info("  Output shape: %s", output.shape)
    logger.info("  Output stats: min=%.3f, max=%.3f, mean=%.3f, std=%.3f",
                output.min(), output.max(), output.mean(), output.std())
    
    return output


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert output tensor (B, C, H, W) to PIL Image.
    
    Assumes tensor is in range [-1, 1] (typical VAE output).
    """
    # Take first image from batch
    img = tensor[0].detach().cpu().float()
    
    # Clamp and scale to [0, 255]
    img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)
    img = (img * 255.0).to(torch.uint8)
    
    # Convert to HWC format
    img = img.permute(1, 2, 0).numpy()
    
    return Image.fromarray(img)


def compare_outputs(pytorch_out: torch.Tensor, pte_out: torch.Tensor):
    """Compare PyTorch and PTE outputs."""
    logger.info("Comparing PyTorch vs PTE outputs:")
    
    diff = torch.abs(pytorch_out - pte_out)
    
    logger.info("  Max absolute difference: %.6f", diff.max())
    logger.info("  Mean absolute difference: %.6f", diff.mean())
    logger.info("  Median absolute difference: %.6f", diff.median())
    
    # Relative error
    rel_error = diff / (torch.abs(pytorch_out) + 1e-8)
    logger.info("  Max relative error: %.6f", rel_error.max())
    logger.info("  Mean relative error: %.6f", rel_error.mean())
    
    # Check if outputs are close
    rtol = 1e-3
    atol = 1e-5
    close = torch.allclose(pytorch_out, pte_out, rtol=rtol, atol=atol)
    logger.info("  Outputs close (rtol=%g, atol=%g): %s", rtol, atol, close)


def main():
    parser = argparse.ArgumentParser(
        description="Debug VAE decoder for FLUX.2-klein",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-4B",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--prompt", default="a cat sitting on a windowsill at sunset",
                        help="Prompt to generate image (and latents) from")
    parser.add_argument("--height", type=int, default=512,
                        help="Target image height")
    parser.add_argument("--width", type=int, default=512,
                        help="Target image width")
    parser.add_argument("--num_inference_steps", type=int, default=4,
                        help="Number of inference steps for generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for generation")
    parser.add_argument("--output_dir", default="./vae_debug_output",
                        help="Directory to save debug outputs")
    parser.add_argument("--latents_path", type=str, default=None,
                        help="Path to load latents from (skip pipeline generation)")
    parser.add_argument("--pte_path", required=True,
                        help="Path to exported VAE decoder .pte file")
    parser.add_argument("--save_latents", action="store_true",
                        help="Save the test latents to file")
    
    args = parser.parse_args()
    
    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pipeline
    pipe, bn_stats = load_pipeline(args.model_id)
    
    # Generate or load latents
    pipeline_image = None
    if args.latents_path and os.path.exists(args.latents_path):
        logger.info("Loading latents from %s", args.latents_path)
        data = torch.load(args.latents_path)
        if isinstance(data, dict):
            latents = data.get("latents", data)
        else:
            latents = data
        logger.info("  Loaded latent shape: %s", latents.shape)
    else:
        # Generate latents from pipeline
        latents, pipeline_image = generate_latents_from_pipeline(
            pipe, args.prompt, args.height, args.width, 
            args.num_inference_steps, args.seed
        )
        
        if latents is None:
            logger.error("Failed to generate latents")
            return
        
        if args.save_latents:
            latents_path = out_dir / "test_latents.pt"
            torch.save({"latents": latents, "prompt": args.prompt}, latents_path)
            logger.info("Saved test latents to %s", latents_path)
        
        # Save the pipeline-generated reference image
        if pipeline_image is not None:
            pipeline_img_path = out_dir / "pipeline_reference.png"
            pipeline_image.save(pipeline_img_path)
            logger.info("Saved pipeline reference image to %s", pipeline_img_path)
    
    # Test PyTorch VAE
    logger.info("\n" + "=" * 60)
    logger.info("TESTING PYTORCH VAE DECODER")
    logger.info("=" * 60)
    
    pytorch_output = decode_with_pytorch_vae(pipe.vae, latents)
    
    # Save PyTorch output
    pytorch_img = tensor_to_image(pytorch_output)
    pytorch_img_path = out_dir / "pytorch_vae_output.png"
    pytorch_img.save(pytorch_img_path)
    logger.info("Saved PyTorch VAE output to %s", pytorch_img_path)
    
    # Test PTE VAE
    logger.info("\n" + "=" * 60)
    logger.info("TESTING EXECUTORCH .PTE VAE DECODER")
    logger.info("=" * 60)
    
    pte_output = decode_with_pte_vae(args.pte_path, latents)
    
    if pte_output is not None:
        # Save PTE output
        pte_img = tensor_to_image(pte_output)
        pte_img_path = out_dir / "pte_vae_output.png"
        pte_img.save(pte_img_path)
        logger.info("Saved PTE VAE output to %s", pte_img_path)
        
        # Compare outputs
        logger.info("\n" + "=" * 60)
        logger.info("PYTORCH vs PTE COMPARISON")
        logger.info("=" * 60)
        compare_outputs(pytorch_output, pte_output)
        
        # Save difference image
        # diff_tensor = torch.abs(pytorch_output - pte_output)
        # # Normalize difference for visualization
        # diff_normalized = diff_tensor / (diff_tensor.max() + 1e-8)
        # diff_img = tensor_to_image(diff_normalized * 2.0 - 1.0)
        # diff_img_path = out_dir / "difference.png"
        # diff_img.save(diff_img_path)
        # logger.info("Saved difference image to %s", diff_img_path)
    else:
        logger.error("PTE decoding failed — cannot compare")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DEBUG COMPLETE")
    logger.info("=" * 60)
    logger.info("Output directory: %s", out_dir)
    logger.info("Files generated:")
    for f in sorted(out_dir.glob("*")):
        if f.is_file():
            logger.info("  - %s", f.name)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
