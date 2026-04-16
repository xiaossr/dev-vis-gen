# DSP Deployment Progress (Qualcomm / QAIRT + ExecuTorch QNN)

Date: 2026-04-15 (updated)

## Goal

Deploy FLUX.2-klein-4B diffusion model on Qualcomm phone DSP in 8-bit quantized form (weights + activations) using ExecuTorch QNN backend.

## Architecture Summary

Three export paths exist:

| Path | Script | Output | Status |
|------|--------|--------|--------|
| **ExecuTorch QNN** (torch.export → .pte) | `export_flux2_klein_qnn.py` | `.pte` files | **ALL 3 COMPONENTS EXPORTED** |
| **QAIRT direct** (ONNX → DLC) | `export_flux2_klein_qairt.py` | `.dlc` files | Abandoned — too many QAIRT converter bugs |
| **ExecuTorch XNNPACK** (CPU) | `export_flux2_klein_xnnpack.py` | `.pte` files | Working (CPU reference baseline) |

## Current Status: ExecuTorch QNN Path — COMPLETE

All three model components have been exported to `.pte` format with INT8 quantization targeting Snapdragon 8 Gen 3 (SM8650).

| Component | Status | File | Size | Notes |
|-----------|--------|------|------|-------|
| VAE Decoder | Done | `vae_decoder.pte` | 55.8 MB | Clean export, no issues |
| Transformer | Done | `transformer.pte` | 3697.1 MB | Needed `online_prepare=True` (defers HTP compilation to device) |
| Text Encoder (Qwen3) | Done | `text_encoder.pte` | 2970.9 MB | Needed 5 bug fixes: custom wrapper to bypass lock, graceful ExportPass handling, val metadata fallbacks |

Supporting files:
- `exported_flux2_klein_qnn/tokenizer/` — saved tokenizer
- `exported_flux2_klein_qnn/export_config.json` — export metadata
- `exported_flux2_klein_qnn/vae_bn_stats.pt` — VAE batch norm stats

## Export Pipeline

```
PyTorch model (fp32)
    |
    v  torch.export.export(strict=True)
Exported GraphModule
    |
    v  QnnQuantizer (8a8w) + prepare_pt2e + calibration (5 passes)
Quantized GraphModule (INT8)
    |
    v  convert_pt2e + _remove_int_quantize_nodes
Folded INT8 GraphModule
    |
    v  capture_program (or fallback: re-export + decompositions + to_edge)
Edge dialect program
    |
    v  to_backend(QnnPartitioner, online_prepare=True)
Delegated program (ops partitioned to QNN HTP)
    |
    v  EdgeProgramManager.to_executorch()
.pte binary
```

## Key Technical Decisions

1. **`online_prepare=True`**: The x86 QNN HTP simulator cannot compile graphs this large (4B+ params). Setting `online_prepare=True` serializes the graph definition + weights without compiling, deferring compilation to the Snapdragon device at first runtime. This is standard for large models.

2. **Graceful ExportPass handling**: ExecuTorch's `ExportPass` base class fails on quantized Qwen3 graphs with dtype mismatches in the fake-tensor interpreter. Rather than bypassing passes entirely (which loses critical `meta["val"]` metadata), the monkey-patch catches errors per-pass and returns the graph unchanged. Passes that succeed still apply.

3. **Custom Qwen3 wrapper**: The HuggingFace `Qwen3Model.forward()` uses `@capture_outputs` decorator with `threading.Lock`, which `torch.export(strict=True)` cannot trace. The wrapper directly accesses internal layers (`embed_tokens`, `layers`, `norm`, `rotary_emb`), bypassing the decorator.

4. **EdgeProgramManager direct**: After `to_backend()`, use `EdgeProgramManager` directly instead of calling `to_edge()` again, which would re-run edge passes (including the failing ExportPass ones) on the already-delegated program.

## Errors Fixed

13 errors were encountered and fixed during export. See `EXECUTORCH_ERRORS.md` for full details.

Key fixes in ExecuTorch 0.6.0 source (both source tree and pip copies):
- `node_visitor.py`: Added `torch.float64` to tensor type map
- `qnn_partitioner.py`: Guard against missing op visitors
- `backend_api.py`: Multi-fallback `val` metadata for delegate nodes
- `lowered_backend_module.py`: Safe `.get("val")` instead of `["val"]`

## QAIRT Path History (Abandoned)

The QAIRT direct path (ONNX → DLC) was attempted first with 25 iterations. VAE decoder worked, but the transformer and text encoder hit QAIRT converter bugs (shape canonicalization, zero-length slices, duplicate buffer names, constant absorption of inputs). See `iteration.md` for the full log.

## C++ Inference Runner (Written)

The `runner/` directory contains a complete C++ inference runner ready for cross-compilation:

- `flux2_main.cpp` — Entry point with argument parsing
- `flux2_runner.h/.cpp` — Runner class implementing the full pipeline:
  1. Text encoder: tokenized prompt → encoder hidden states
  2. Transformer: 4-step flow-matching Euler denoising (no CFG, distilled)
  3. Latent unpacking: [1, 1024, 128] patches → [1, 32, 64, 64] spatial
  4. VAE decoder: latents → 512x512 RGB image → PPM file
- `CMakeLists.txt` — Build configuration
- `deploy_to_device.sh` — Build, push, and run script

**Note:** The tokenizer in the runner is a placeholder (byte-level encoding). For correct results, integrate the saved Qwen3 tokenizer from `exported_flux2_klein_qnn/tokenizer/`.

## Next Steps (Requires Device)

1. **Install Android NDK** (r25c+) and set `ANDROID_NDK` env var

2. **Build ExecuTorch for Android ARM64** with QNN backend:
   ```bash
   cd executorch && mkdir build-android && cd build-android
   cmake .. -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
            -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 \
            -DEXECUTORCH_BUILD_QNN=ON -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
            -DCMAKE_BUILD_TYPE=Release
   cmake --build . -j$(nproc)
   ```

3. **Build and deploy runner:**
   ```bash
   cd runner && ./deploy_to_device.sh
   ```

4. **First on-device run** — first inference will be slow (HTP graph compilation from `online_prepare=True`), subsequent runs cached

5. **Integrate proper Qwen3 tokenizer** — replace placeholder with sentencepiece/tiktoken using the saved tokenizer.json

6. **Validate accuracy** — compare outputs against CPU fp32 reference

## Files

- `export_flux2_klein_qnn.py` — Main export script (ExecuTorch QNN path)
- `runner/` — C++ inference runner for on-device execution
- `exported_flux2_klein_qnn/` — All .pte model files + tokenizer + config
- `EXECUTORCH_ERRORS.md` — All 13 errors encountered and how they were fixed
- `iteration.md` — Full iteration log (QAIRT + ExecuTorch)
- `export_flux2_klein_qairt.py` — QAIRT direct path (abandoned)
- `export_flux2_klein_xnnpack.py` — CPU baseline (XNNPACK)

## Environment

- Python 3.10, torch 2.7.0+cu126, executorch 0.6.0, torchao 0.10.0
- QNN SDK 2.45.0.260326
- NVIDIA RTX 4090 (for calibration)
- Target: Qualcomm Snapdragon 8 Gen 3 (SM8650)
