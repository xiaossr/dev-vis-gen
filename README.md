# FLUX.2-klein-4B On-Device Deployment

End-to-end guide for running FLUX.2-klein-4B text-to-image generation on an Android phone using ExecuTorch + XNNPACK.

## Overview

```
Host (Mac / Linux)                        Android Phone
─────────────────                         ──────────────
1. Export .pte models  ───adb push───►  text_encoder.pte
2. Tokenize prompt     ───adb push───►  transformer.pte
3. Build C++ runner    ───adb push───►  vae_decoder.pte
                                        prompt.bin, bn_*.bin
                                        flux2_main
                                            │
                                            ▼
                                        output.ppm
```

**Pipeline on device:** `prompt.bin → text_encoder.pte → transformer.pte → vae_decoder.pte → output.ppm`

## Prerequisites

- macOS (Apple Silicon recommended) or Linux
- Python 3.10+ with PyTorch, diffusers, transformers
- Android NDK r28c (for Android cross-compilation)
- An Android phone with USB debugging enabled (16GB+ RAM recommended)
- ExecuTorch source at `../executorch`

## Step 1: Export Models (Prepare .pte files)

> If you have already exported the models, skip to Step 2.

```bash
cd dev-vis-gen

# Export all components (text_encoder + transformer + vae_decoder):
python export_flux2_klein_xnnpack.py \
    --output_dir ./exported_flux2_klein

# With w4da8 quantization for text encoder, w8a8 for other components 
python export_flux2_klein_xnnpack.py \
    --output_dir ./exported_flux2_klein \
    --text_encoder_8da4w \
    --embedding_quantize 8 \
    --w8a8

# With w4da8 quantization for text encoder, fp32 for other components 
python export_flux2_klein_xnnpack.py \
    --output_dir ./exported_flux2_klein \
    --component text_encoder \
    --text_encoder_8da4w \
    --embedding_quantize 8

# Export with image-to-image support:
python export_flux2_klein_xnnpack.py \
    --output_dir ./exported_flux2_klein \
    --num_img2img_images 1
```

After export, `exported_flux2_klein/` contains:
```
text_encoder.pte       # Qwen3 text encoder
transformer.pte        # Flux2 MMDiT transformer
vae_decoder.pte        # VAE decoder
vae_bn_stats.pt        # Batch-norm running statistics
export_config.json     # Metadata
tokenizer/             # Qwen2TokenizerFast files
```

## Step 2: Prepare Inputs for Mobile

The C++ runner cannot run the Python tokenizer, so we pre-tokenize on the host
and convert BN stats to flat binary:

```bash
python prepare_mobile.py \
    --model_dir ./exported_flux2_klein \
    --prompt "a cat sitting on a windowsill at sunset" \
    --output_dir ./exported_flux2_klein
```

This produces:
- `prompt.bin` — tokenized prompt (int64 IDs + attention mask)
- `bn_mean.bin` — VAE batch-norm mean (float32)
- `bn_var.bin` — VAE batch-norm var (float32)

## Step 3: Build & Run on Your Computer (Validate)

### 3.1 Build ExecuTorch

> **Important:** Do NOT use `cmake --workflow llm-release` — the LLM preset enables
> sentencepiece / LLM runner targets that have a known `absl::string_view` build
> error on macOS.  The Flux2 runner only needs core ExecuTorch + XNNPACK.

```bash
cd ../executorch

# Clean any stale cache from a previous preset build
rm -rf cmake-out

cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DPYTHON_EXECUTABLE=python3 \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=OFF \
    -DEXECUTORCH_BUILD_COREML=OFF \
    -DEXECUTORCH_BUILD_MPS=OFF \
    -Bcmake-out .

cmake --build cmake-out -j16 --target install --config Release
```

### 3.2 Build Flux2 runner

```bash
# Run from the executorch root directory
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=python3 \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -Bcmake-out/examples/models/flux2 \
    examples/models/flux2

cmake --build cmake-out/examples/models/flux2 -j16 --config Release
```

### 3.3 Run locally

```bash
MODEL_DIR=../dev-vis-gen/exported_flux2_klein

cmake-out/examples/models/flux2/flux2_main \
    --model_dir $MODEL_DIR \
    --tokens $MODEL_DIR/prompt.bin \
    --output output.ppm \
    --steps 4 \
    --seed 42
```

Convert PPM to PNG (requires ImageMagick):
```bash
(first time) brew install imagemagick
magick output.ppm output.png
```

## Step 4: Run on Android Phone

### 4.1 Set Android NDK

```bash
export ANDROID_NDK=~/android-ndk-r28c
```

### 4.2 Build ExecuTorch for Android

```bash
cd ../executorch

rm -rf cmake-out-android

cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DPYTHON_EXECUTABLE=python3 \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=OFF \
    -DEXECUTORCH_BUILD_COREML=OFF \
    -DEXECUTORCH_BUILD_MPS=OFF \
    -Bcmake-out-android .

cmake --build cmake-out-android -j16 --target install --config Release
```

### 4.3 Build Flux2 runner for Android

```bash
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=python3 \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -Bcmake-out-android/examples/models/flux2 \
    examples/models/flux2

cmake --build cmake-out-android/examples/models/flux2 -j16 --config Release
```

### 4.4 Push files to phone

Connect your Android phone with USB debugging enabled, then:

```bash
adb shell mkdir -p /data/local/tmp/flux2

# Push .pte models
adb push ../dev-vis-gen/exported_flux2_klein/text_encoder.pte /data/local/tmp/flux2/
adb push ../dev-vis-gen/exported_flux2_klein/transformer.pte  /data/local/tmp/flux2/
adb push ../dev-vis-gen/exported_flux2_klein/vae_decoder.pte  /data/local/tmp/flux2/

# Push binary inputs
adb push ../dev-vis-gen/exported_flux2_klein/prompt.bin  /data/local/tmp/flux2/
adb push ../dev-vis-gen/exported_flux2_klein/bn_mean.bin /data/local/tmp/flux2/
adb push ../dev-vis-gen/exported_flux2_klein/bn_var.bin  /data/local/tmp/flux2/

# Push runner binary
adb push cmake-out-android/examples/models/flux2/flux2_main /data/local/tmp/flux2/
```

### 4.5 Run on phone

```bash
adb shell "cd /data/local/tmp/flux2 && \
    chmod +x flux2_main && \
    ./flux2_main \
        --model_dir . \
        --tokens prompt.bin \
        --output output.ppm \
        --steps 4 \
        --seed 42"
```

### 4.6 Pull the result

```bash
adb pull /data/local/tmp/flux2/output.ppm .
convert output.ppm output.png    # optional: convert to PNG
```

## Command-Line Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model_dir` | (required) | Directory containing `.pte` and `bn_*.bin` files |
| `--tokens` | (required) | Path to tokenized prompt binary |
| `--output` | `output.ppm` | Output image path (PPM format) |
| `--height` | `512` | Image height (must match export) |
| `--width` | `512` | Image width (must match export) |
| `--steps` | `4` | Number of denoising steps |
| `--seed` | `42` | Random seed for noise generation |
| `--vae_sf` | `8` | VAE scale factor |
| `--in_channels` | `128` | Transformer in_channels |
| `--max_seq_len` | `512` | Max text sequence length |

## Changing the Prompt

To use a different prompt, re-run `prepare_mobile.py` on the host and push the
new `prompt.bin` to the phone:

```bash
# On host
python prepare_mobile.py \
    --model_dir ./exported_flux2_klein \
    --prompt "a cyberpunk city at night" \
    --output_dir ./exported_flux2_klein

# Push new tokens
adb push ./exported_flux2_klein/prompt.bin /data/local/tmp/flux2/

# Run again
adb shell "cd /data/local/tmp/flux2 && ./flux2_main --model_dir . --tokens prompt.bin --output output.ppm"
```

## Notes

- **Memory**: The full pipeline in fp32 needs ~20+ GB. Use `--quantize` during
  export (int8) to reduce model size to ~4-5 GB, which fits on 16 GB phones.
- **Output format**: PPM is a simple uncompressed format. Use ImageMagick
  (`convert`) or Python (`PIL`) to convert to PNG/JPEG.
- **Noise generation**: The C++ runner uses `std::mt19937` for random noise.
  Results will differ from the Python runner (which uses PyTorch's RNG) even
  with the same seed, but both produce valid images.
- **Tokenization**: Pre-tokenization on the host is required because the Qwen3
  tokenizer (HuggingFace format) does not have a C++ implementation in
  ExecuTorch. Each new prompt needs a new `prompt.bin`.
