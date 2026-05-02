# FLUX.2-klein-4B on Qualcomm HTP — End-to-End Guide

Run FLUX.2-klein-4B on Snapdragon Hexagon NPU via ExecuTorch QNN backend.

**Working split** (Non-noise quality):
- **Text encoder** — XNNPACK CPU
- **Transformer** — XNNPACK CPU
- **VAE decoder** — QNN HTP, w8a8 quantized

## 1. Environment Setup

### Host machine (Linux x86_64 or Mac, Docker recommended)

```bash
# Conda env
conda create -n executorch python=3.10 -y
conda activate executorch
pip install -r requirements_export.txt

# Android NDK r28c
wget https://dl.google.com/android/repository/android-ndk-r28c-linux.zip
unzip android-ndk-r28c-linux.zip
export ANDROID_NDK=$(pwd)/android-ndk-r28c

# Qualcomm AI Engine Direct SDK (download from Qualcomm developer portal)
# https://qpm.qualcomm.com/  — search "Qualcomm AI Engine Direct"
export QNN_SDK_ROOT=/path/to/qairt/<version>
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH

# ExecuTorch source
git clone --recurse-submodules https://github.com/pytorch/executorch.git
cd executorch
./install_requirements.sh
```

## 2. Patches Required to ExecuTorch QNN Backend

The stock ExecuTorch QNN backend has issues that crash on the Flux model. Apply
these patches to `/root/executorch/backends/qualcomm/`:

### `_passes/lift_constant_scalar_operands.py`
- Add `aten.pow.Scalar: TensorOpInfo(aten.pow.Tensor_Tensor, False, False)` to `SCALAR_OPS`
- Add `or not hasattr(n.target, "_schema")` guard in `_lift()` loop
- In `_build_tensor_constant`, guard `node.args[0].meta` access with `isinstance(first_arg, fx.Node)` check

### `partition/qnn_partitioner.py`
- In `is_node_supported`, add early return if `node.target.__name__ not in self.node_visitors`

### `quantizer/rules.py`
- In `_mark_nodes_as_annotated`, add `if node is None: continue`

### `quantizer/annotators/htp_rules.py`
- In `LayerNorm.annotate`, guard `weight_node` and `bias_node` with `is not None`

### `builders/op_layer_norm.py`
- When weight is None, create implicit `torch.ones(normalized_shapes)` weight
- When bias is None, create implicit `torch.zeros(normalized_shapes)` bias

## 3. Collect Calibration Data

Real prompts produce better activation ranges than random data:

```bash
python collect_calibration_data.py \
    --output_dir ./calibration_data \
    --num_timesteps 4
```

Produces:
- `calibration_text_encoder.pt` — tokenized prompts
- `calibration_transformer.pt` — real prompt embeddings + noise at multiple timesteps
- `calibration_vae.pt` — denoised latents from running partial pipeline

## 4. Export Models

### Text encoder (XNNPACK CPU)
```bash
python export_flux2_klein_xnnpack.py \
    --component text_encoder \
    --text_encoder_8da4w --embedding_quantize 8 \
    --output_dir ./exported_flux2_klein
```

### Transformer (QNN HTP w8a8 quantized)
```bash
python export_flux2_klein_qnn.py \
    --soc SM8650 \
    --component transformer \
    --quantize --quant_dtype 8a8w \
    --calibration_dir ./calibration_data \
    --output_dir ./exported_flux2_klein
```

### VAE decoder (QNN HTP fp16, no quantization)
```bash
python export_flux2_klein_qnn.py \
    --soc SM8650 \
    --component vae \
    --output_dir ./exported_flux2_klein
```

### Prepare binary inputs
```bash
python prepare_mobile.py \
    --model_dir ./exported_flux2_klein \
    --prompt "a cat sitting on a windowsill at sunset" \
    --output_dir ./exported_flux2_klein
```

## 5. Build C++ Runner

Build ExecuTorch core with both QNN + XNNPACK backends:

```bash
cd /root/executorch
export ANDROID_NDK=/root/android-ndk-r28c

cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android-both \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -Bcmake-out-android-both .

cmake --build cmake-out-android-both -j16 --target install --config Release
```

Build the flux2 runner:

```bash
cmake \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=python3 \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -Bcmake-out-android-both/examples/models/flux2 \
    -S examples/models/flux2

cmake --build cmake-out-android-both/examples/models/flux2 -j16 --config Release
```

Output binary: `cmake-out-android-both/examples/models/flux2/flux2_qnn_main`

## 6. Push to Phone

Adjust `SRC` path in [push_htp.sh](push_htp.sh) to your local export folder, then:

```bash
bash push_htp.sh
```

Required files on device (`/data/local/tmp/flux2/htp/`):
- `text_encoder.pte`, `transformer.pte`, `vae_decoder.pte`
- `prompt.bin`, `bn_mean.bin`, `bn_var.bin`
- `flux2_qnn_main` (runner binary)
- `libqnn_executorch_backend.so`
- `libQnnHtp.so`, `libQnnHtpV75Stub.so`, `libQnnSystem.so` (from QNN SDK `lib/aarch64-android/`)
- `libQnnHtpV75Skel.so` (from QNN SDK `lib/hexagon-v75/unsigned/`)

> **SoC version**: V75 above is for SM8650. Adjust to `V73` (SM8550), `V79` (SM8750), etc.

## 7. Run on Phone

```bash
adb shell "cd /data/local/tmp/flux2/htp && \
    chmod +x flux2_qnn_main && \
    export LD_LIBRARY_PATH=. && \
    ./flux2_qnn_main \
        --model_dir . \
        --tokens prompt.bin \
        --output output.ppm \
        --steps 4 --seed 42 \
        --htp_performance_mode 3"
```

### Pull result

```bash
adb pull /data/local/tmp/flux2/htp/output.ppm .
python3 -c "from PIL import Image; Image.open('output.ppm').save('output.png')"
```

### Change prompt

Re-run only `prepare_mobile.py` with new prompt and push the new `prompt.bin`:

```bash
python prepare_mobile.py --model_dir ./exported_flux2_klein \
    --prompt "a cyberpunk city at night" --output_dir ./exported_flux2_klein
adb push exported_flux2_klein/prompt.bin /data/local/tmp/flux2/htp/
adb shell "cd /data/local/tmp/flux2/htp && export LD_LIBRARY_PATH=. && \
    ./flux2_qnn_main --model_dir . --tokens prompt.bin --output output.ppm"
```

## Runtime Flags (`flux2_qnn_main`)

| Flag | Default | Description |
|------|---------|-------------|
| `--model_dir` | (required) | Directory with `.pte` and `bn_*.bin` files |
| `--tokens` | (required) | Path to `prompt.bin` |
| `--output` | `output.ppm` | Output image path |
| `--height`, `--width` | `512` | Image dimensions (must match export) |
| `--steps` | `4` | Denoising steps (Klein is distilled, 4 is enough) |
| `--seed` | `42` | RNG seed |
| `--htp_performance_mode` | `3` | 0=default, 1=sustained, 2=burst, 3=high_perf, 4=power_saver |
| `--log_level` | `0` | QNN log verbosity 0–5 |

## Validate Quantization Quality (Optional)

Compare QNN transformer output against PyTorch reference:

```bash
python test_transformer_qnn.py --qnn_pte ./exported_flux2_klein/transformer.pte
```

Reports correlation, max/mean abs diff. Correlation < 0.9 means the quantized model is producing noise — try `--quant_dtype 16a8w` for higher precision activations.

## Known Issues

- **LayerNorm falls back to CPU** — QNN HTP rejects rank-3 inputs (Flux uses `(B, S, C)`). Lightweight, acceptable.
- **8 view_copy ops fall back to CPU** — dtype mismatch (uint8 in / fp32 out) at quantization boundaries.
- **~32 partitions** in transformer due to above. Each partition boundary involves quant/dequant which accumulates error in iterative denoising.
- **Pure noise output with w8a8 quantized transformer** — likely caused by partition boundary error accumulation. Workaround: use CPU transformer or try `16a8w`.
