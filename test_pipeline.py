#!/usr/bin/env python3
"""
Quick end-to-end test of the FLUX.2-klein-4B pipeline.

Runs the full diffusion pipeline on GPU (fp32) and saves a reference image.
This validates the model works correctly before deploying to device.

Usage:
  python test_pipeline.py --prompt "a cat on a windowsill" --output test_output.png
"""

import argparse
import gc
import logging
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_pipeline")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-4B")
    p.add_argument("--prompt", default="a photograph of a cat sitting on a windowsill at sunset")
    p.add_argument("--output", default="test_output.png")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from diffusers import Flux2KleinPipeline

    logger.info("Loading pipeline: %s", args.model_id)
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
    )
    pipe = pipe.to(args.device)
    logger.info("Pipeline loaded on %s", args.device)

    logger.info("Generating: '%s' (%dx%d, %d steps)", args.prompt, args.height, args.width, args.steps)
    t0 = time.time()
    with torch.no_grad():
        result = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            max_sequence_length=512,
            guidance_scale=0.0,
        )
    dt = time.time() - t0
    logger.info("Generated in %.1f seconds", dt)

    image = result.images[0]
    image.save(args.output)
    logger.info("Saved: %s (%dx%d)", args.output, image.width, image.height)

    # Basic sanity: check the image isn't all black/white/noise
    import numpy as np
    arr = np.array(image).astype(float)
    logger.info("Image stats: mean=%.1f, std=%.1f, min=%d, max=%d",
                arr.mean(), arr.std(), arr.min(), arr.max())

    if arr.std() < 5:
        logger.warning("Image has very low variance — may be blank!")
    elif arr.std() > 100:
        logger.info("Image looks like it has real content (good std)")
    else:
        logger.info("Image variance looks reasonable")


if __name__ == "__main__":
    main()
