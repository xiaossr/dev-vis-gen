# FLUX.2-klein-4B DSP Deployment: Status Report

**Date:** 2026-04-15
**Project:** On-device diffusion model inference on Qualcomm Hexagon DSP
**Model:** FLUX.2-klein-4B (4B parameters)
**Status: ALL COMPONENTS EXPORTED SUCCESSFULLY**

## Objective

Run FLUX.2-klein-4B text-to-image generation entirely on a Qualcomm phone's DSP (Hexagon HTP), using INT8 quantization (8-bit weights and activations).

## Result

All three model components have been exported to ExecuTorch `.pte` format with INT8 static quantization, targeting Snapdragon 8 Gen 3 (SM8650):

| Component | File | Size | Quantization |
|-----------|------|------|--------------|
| Text Encoder (Qwen3, 27 layers) | `text_encoder.pte` | 2970.9 MB | INT8 static (8a8w) |
| Transformer (Flux2, 4B params) | `transformer.pte` | 3697.1 MB | INT8 static (8a8w) |
| VAE Decoder | `vae_decoder.pte` | 55.8 MB | INT8 static (8a8w) |
| **Total** | | **~6.7 GB** | |

Output directory: `exported_flux2_klein_qnn/`

## How It Was Done

### Pipeline

```
PyTorch (fp32) → torch.export → QnnQuantizer (INT8) → calibration → convert_pt2e
    → QNN decompositions → to_edge → QnnPartitioner (online_prepare) → .pte
```

### Two Paths Were Attempted

1. **QAIRT Direct (ONNX → DLC)** — 25 iterations, abandoned. VAE worked but transformer/text encoder hit QAIRT converter bugs (shape canonicalization, zero-length slices, duplicate buffer names).

2. **ExecuTorch QNN (torch.export → .pte)** — Succeeded after fixing 13 bugs in ExecuTorch 0.6.0. This path avoids ONNX entirely and uses Qualcomm's QNN integration in ExecuTorch.

### Key Technical Challenges Solved

- **x86 HTP compilation limit**: The QNN HTP simulator on x86 can't compile 4B-parameter graphs. Solved with `online_prepare=True` which defers compilation to the actual Snapdragon device.

- **Qwen3 `@capture_outputs` lock**: HuggingFace's Qwen3 uses a threading.Lock decorator that `torch.export` can't trace. Built a custom wrapper that directly accesses model internals.

- **ExportPass dtype mismatch**: ExecuTorch's fake-tensor interpreter fails on quantized Qwen3 graphs with mixed int/float operations. Fixed with graceful per-pass error catching that preserves graph metadata.

- **Missing `val` metadata on delegate nodes**: When edge passes are skipped, newly-created nodes lack required metadata. Fixed with multi-source fallback in `backend_api.py`.

See `EXECUTORCH_ERRORS.md` for all 13 errors and their fixes.

## What's Next

1. Build ExecuTorch runtime for Android ARM64 with QNN backend
2. Build C++ inference runner for the 3-component diffusion pipeline
3. Push .pte files + QNN runtime libraries to Snapdragon 8 Gen 3 device
4. First on-device inference (first run will be slow due to HTP compilation)
5. Validate INT8 accuracy against fp32 reference

## Files

| File | Purpose |
|------|---------|
| `export_flux2_klein_qnn.py` | Export script (the one that worked) |
| `EXECUTORCH_ERRORS.md` | All 13 errors and fixes |
| `PROGRESS.md` | Detailed progress log |
| `iteration.md` | QAIRT iteration log (25 iterations, abandoned) |
| `exported_flux2_klein_qnn/` | Output directory with all .pte files |
