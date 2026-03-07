import torch, numpy as np
from diffusers import Flux2KleinPipeline
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.float32)

# 1. Check scheduler sigmas
use_flow = getattr(pipe.scheduler.config, "use_flow_sigmas", False)
print("use_flow_sigmas:", use_flow)

from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
mu = compute_empirical_mu(1024, 4)
print("mu:", mu)

sigmas = np.linspace(1.0, 0.25, 4)
if use_flow:
    sigmas = None

if sigmas is not None:
    pipe.scheduler.set_timesteps(sigmas=sigmas.tolist(), mu=mu)
else:
    pipe.scheduler.set_timesteps(num_inference_steps=4, mu=mu)

print("sigmas:", pipe.scheduler.sigmas.tolist())
print("timesteps:", pipe.scheduler.timesteps.tolist())

# 2. Check VAE BN
bn = pipe.vae.bn
print("\nbn.affine:", bn.affine)
print("bn.num_features:", bn.num_features)
if bn.affine:
    print("bn.weight[:3]:", bn.weight[:3].tolist())
    print("bn.bias[:3]:", bn.bias[:3].tolist())
print("bn.running_mean[:3]:", bn.running_mean[:3].tolist())
print("bn.running_var[:3]:", bn.running_var[:3].tolist())
