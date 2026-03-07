#!/usr/bin/env python3
"""Compare PyTorch pipeline vs ExecuTorch inference to find divergence."""

import json
import torch
import numpy as np
from pathlib import Path

# ---- Load export config ------------------------------------------------
config = json.loads(Path("exported_flux2_klein/export_config.json").read_text())
height, width = config["height"], config["width"]
max_text_len = config["max_text_len"]
in_channels = config["transformer"]["in_channels"]
vae_sf = config["vae_scale_factor"]
bn_eps = config["vae"]["batch_norm_eps"]

patch_h = height // (vae_sf * 2) * 2 // 2  # same as _compute_latent_dims
patch_w = width // (vae_sf * 2) * 2 // 2
print(f"patch_h={patch_h}, patch_w={patch_w}, in_channels={in_channels}")

# ---- Load pipeline in PyTorch ------------------------------------------
print("\n=== Loading PyTorch pipeline ===")
from diffusers import Flux2KleinPipeline
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.float32
)

# ---- Encode prompt (PyTorch) -------------------------------------------
print("\n=== Encoding prompt (PyTorch) ===")
prompt = "a cat sitting on a windowsill at sunset"
result = pipe.encode_prompt(prompt=prompt, max_sequence_length=max_text_len)
pt_prompt_embeds = result[0]
pt_text_ids = result[1]
print(f"PT prompt_embeds: {pt_prompt_embeds.shape}, dtype={pt_prompt_embeds.dtype}")
print(f"PT text_ids: {pt_text_ids.shape}, dtype={pt_text_ids.dtype}")

# ---- Load pre-computed embeddings --------------------------------------
print("\n=== Loading pre-computed embeddings ===")
emb_index = json.loads(Path("exported_flux2_klein/embeddings/index.json").read_text())
emb_file = emb_index.get(prompt)
emb_data = torch.load(f"exported_flux2_klein/embeddings/{emb_file}", map_location="cpu", weights_only=True)
et_prompt_embeds = emb_data["prompt_embeds"]
print(f"ET prompt_embeds: {et_prompt_embeds.shape}")

# Compare embeddings
emb_diff = (pt_prompt_embeds.float() - et_prompt_embeds.float()).abs()
print(f"Embedding diff: max={emb_diff.max():.6f}, mean={emb_diff.mean():.6f}")

# ---- Prepare noise (same seed) -----------------------------------------
print("\n=== Preparing noise ===")
seed = 42
num_latent_channels = in_channels // 4
gen = torch.Generator(device="cpu").manual_seed(seed)
noise = torch.randn(1, num_latent_channels * 4, patch_h, patch_w, dtype=torch.float32, generator=gen)
print(f"Noise shape: {noise.shape}")

# Pack latents
latents = noise.reshape(1, in_channels, patch_h * patch_w).permute(0, 2, 1)  # (1, 1024, 128)
print(f"Packed latents: {latents.shape}")

# ---- Prepare position IDs -----------------------------------------------
t = torch.arange(1)
h = torch.arange(patch_h)
w = torch.arange(patch_w)
l = torch.arange(1)
latent_ids = torch.cartesian_prod(t, h, w, l).unsqueeze(0)  # (1, 1024, 4)

# ---- Run ONE transformer step (PyTorch) ---------------------------------
print("\n=== Transformer step 1 (PyTorch) ===")
sigma = 1.0
timestep_pt = torch.full((1,), sigma).to(pt_prompt_embeds.dtype)

# Pipeline divides timestep by 1000 before passing to transformer
pt_out = pipe.transformer(
    hidden_states=latents.to(pt_prompt_embeds.dtype),
    encoder_hidden_states=pt_prompt_embeds,
    timestep=timestep_pt / 1000,  # pipeline divides by 1000
    img_ids=latent_ids,
    txt_ids=pt_text_ids,
    guidance=None,
    return_dict=False,
)[0]
pt_noise_pred = pt_out[:, :latents.shape[1], :]
print(f"PT noise_pred: {pt_noise_pred.shape}")
print(f"  min={pt_noise_pred.min():.4f}, max={pt_noise_pred.max():.4f}, mean={pt_noise_pred.mean():.4f}")

# ---- Run ONE transformer step (ExecuTorch) -------------------------------
print("\n=== Transformer step 1 (ExecuTorch) ===")
from executorch.runtime import Runtime

runtime = Runtime.get()
program = runtime.load_program("exported_flux2_klein/transformer.pte")
et_transformer = program.load_method("forward")

# The exported graph includes the transformer's internal ×1000, so we
# must divide by 1000 (same as the pipeline does).
timestep_et = torch.full((1,), sigma / 1000.0, dtype=torch.float32)

# Our inference casts IDs to float
img_ids_float = latent_ids.to(torch.float32)

# Build text_ids
t2 = torch.arange(1)
h2 = torch.arange(1)
w2 = torch.arange(1)
s2 = torch.arange(max_text_len)
txt_ids = torch.cartesian_prod(t2, h2, w2, s2).unsqueeze(0).to(torch.float32)

et_out = et_transformer.execute([
    latents.contiguous(),
    et_prompt_embeds.contiguous(),
    timestep_et.contiguous(),
    img_ids_float.contiguous(),
    txt_ids.contiguous(),
])
if isinstance(et_out, (list, tuple)):
    et_out = et_out[0]
et_noise_pred = et_out[:, :latents.shape[1], :]
print(f"ET noise_pred: {et_noise_pred.shape}")
print(f"  min={et_noise_pred.min():.4f}, max={et_noise_pred.max():.4f}, mean={et_noise_pred.mean():.4f}")

# ---- Compare ---------------------------------------------------------------
print("\n=== COMPARISON ===")
diff = (pt_noise_pred.float() - et_noise_pred.float()).abs()
print(f"Noise pred diff: max={diff.max():.4f}, mean={diff.mean():.4f}")
rel_diff = diff / (pt_noise_pred.float().abs() + 1e-8)
print(f"Relative diff:   max={rel_diff.max():.4f}, mean={rel_diff.mean():.4f}")

if diff.max() > 1.0:
    print("\n⚠️  LARGE DIVERGENCE — likely quantization or export issue")
elif diff.max() > 0.1:
    print("\n⚠️  Moderate divergence — quantization artifacts")
else:
    print("\n✅ Outputs are close — issue is likely in post-processing")
