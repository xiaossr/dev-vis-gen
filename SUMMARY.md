# FLUX.2-klein-4B DSP Deployment: Status Report

**Date:** 2026-04-12
**Project:** On-device diffusion model inference on Qualcomm Hexagon DSP
**Model:** FLUX.2-klein-4B (4B parameters)

## Objective

Run FLUX.2-klein-4B text-to-image generation entirely on a Qualcomm phone's DSP (Hexagon HTP), not just the CPU. This requires converting the model from PyTorch into a format the DSP hardware can execute, with INT8 quantization (8-bit weights and activations) to fit in the DSP's limited memory and exploit its fixed-point compute units.

## Pipeline Overview

The model has three components that must each be converted separately:

```
Text Encoder (Qwen3, 27 layers)  -->  Transformer (Flux2, 4B params)  -->  VAE Decoder
      prompt text                      denoising (4 steps)                  latents → image
```

The conversion pipeline for each component:

```
PyTorch model
    |
    v
ONNX graph  (torch.onnx.export — translate PyTorch ops into a standard IR)
    |
    v  <-- This is where it's failing
QAIRT converter  (Qualcomm's tool — translates ONNX ops into DSP-native ops)
    |
    v
Quantized DLC  (Qualcomm's model format, with INT8 weights + activations)
    |
    v
Runs on Hexagon DSP
```

## Current Status

| Component | PyTorch → ONNX | ONNX → QAIRT DLC | End-to-end |
|-----------|:-:|:-:|:-:|
| VAE Decoder | Done | Done | Working |
| Text Encoder | Done | Fails | Blocked |
| Transformer | Done | Fails | Blocked |

**The ONNX export works for all three components.** The blocker is the QAIRT converter — Qualcomm's tool that translates the ONNX graph into DSP-native operations. It fails on the transformer and text encoder with shape inference and graph canonicalization errors.

## Where Exactly It Fails (and Why)

The problem is **not** quantization. The QAIRT converter fails during graph compilation, before quantization even runs. Specifically:

1. **The QAIRT converter cannot handle certain ONNX graph patterns** produced by the diffusion model. These include: fused attention ops (SDPA), chunk/split operations, repeat_interleave, CastLike nodes, and Reshape ops with symbolic dimensions (`-1`, `0`).

2. **We've been systematically removing unsupported patterns** from the ONNX graph — replacing SDPA with manual matmul attention, replacing chunk() with explicit slicing, rewriting rotary embeddings, removing CastLike, etc. Each fix resolves one class of errors but reveals the next.

3. **The most likely remaining root cause** (identified today): the transformer's ONNX graph still contained Reshape operations with unresolved shape placeholders (`-1` meaning "infer this dimension"). The QAIRT converter's shape inference doesn't handle these correctly and produces invalid stride/shape calculations. The VAE export already resolved these (via `onnxsim`), which is why it works. This fix has now been applied to the transformer path but has not been tested yet.

4. **We also found a real bug in QAIRT SDK 2.45**: C++ objects hold raw pointers to Python/numpy data that gets garbage-collected, causing use-after-free crashes. We wrote a monkey-patch (`patch_qairt_reshape.py`) to work around this, but it suggests the SDK may have other bugs.

## Work Done (22 iterations)

| Iteration | What was changed | Result |
|-----------|-----------------|--------|
| 1-3 | Replace SDPA with manual matmul attention | New errors (SplitToSequence) |
| 4-8 | Replace chunk/split with explicit slicing in AdaLayerNorm, SwiGLU, Modulation | Fewer errors, but new ones (repeat_interleave) |
| 9-12 | Rewrite rotary position embeddings, option to disable | Still fails (CastLike) |
| 13-15 | Remove CastLike nodes from ONNX | Still fails (canonicalization) |
| 16-19 | Patch QAIRT Python bindings for memory safety, disable GC | More stable, same errors |
| 20-22 | Various attempts at broadcast/shape fixes | Same core canonicalization errors |
| **Today** | **Add Reshape shape resolution to transformer path** | **Not yet tested** |

## Alternative Path Available

A second export approach exists (`export_flux2_klein_qnn.py`) that bypasses ONNX entirely:

```
PyTorch → torch.export() → ExecuTorch QnnQuantizer → QnnPartitioner → .pte file
```

This uses Qualcomm's ExecuTorch integration instead of the ONNX-based QAIRT tools. It's written and ready but requires building ExecuTorch with QNN backend support, which hasn't been done yet.

## Open Questions / Decisions Needed

1. **Should we continue with QAIRT or pivot to ExecuTorch QNN?**
   - QAIRT: 22 iterations invested, close to root cause, but fighting SDK bugs
   - ExecuTorch QNN: Cleaner pipeline, but untested and requires build setup

2. **Is a newer QAIRT SDK available?** We're on 2.45.0, which has a confirmed memory corruption bug. A newer version might fix our remaining issues.

3. **Is there Qualcomm support available?** The errors look like SDK bugs in their graph canonicalizer. A Qualcomm contact or developer forum could save significant time.

## Immediate Next Steps

1. Test the Reshape resolution fix on the transformer (smoke test #23)
2. If that fails, try the CLI-based converter tools (`qairt-converter` + `qairt-quantizer`) as a fallback
3. If QAIRT remains blocked, build ExecuTorch with QNN support and try the alternative path

## Reference

- Working CPU baseline: `export_flux2_klein_xnnpack.py` (ExecuTorch + XNNPACK, fully working)
- QAIRT export script: `export_flux2_klein_qairt.py`
- ExecuTorch QNN script: `export_flux2_klein_qnn.py`
- Environment setup: `CONTEXT_4090.md`
- Detailed progress log: `PROGRESS.md`
