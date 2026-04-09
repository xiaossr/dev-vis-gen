# Session Context: Flux2 QNN/DSP Export on 4090

Transfer this file to the 4090 machine before starting. It contains everything needed to continue the work.

## What This Project Is

Deploying **FLUX.2-klein-4B** (a 4B-parameter diffusion model) to Android phones, targeting the **Qualcomm Hexagon HTP/DSP** (not just the CPU).

- **Repo:** `dev-vis-gen` — ExecuTorch-based on-device deployment
- **Current state:** Working CPU deployment via ExecuTorch + XNNPACK. W8A8/8DA4W quantization exists for the CPU path.
- **Goal for this week:** Export the model components with INT8 static quantization for the **QNN HTP backend** (Qualcomm DSP), using `export_flux2_klein_qnn.py`.

## Why a 4090?

- FLUX.2-klein-4B is ~8GB at bf16, ~16GB at fp32
- INT8 static quantization requires calibration (20–50 forward passes through the full model)
- `torch.export()` of the transformer alone takes 30+ min on CPU
- GPU (CUDA) is needed for practical iteration time

## Repo Structure

```
dev-vis-gen/
├── export_flux2_klein_xnnpack.py  # EXISTING: CPU/XNNPACK export (already working)
├── export_flux2_klein_qnn.py      # NEW: QNN/HTP export script (write this session)
├── flux2_main.cpp                 # C++ Android runner (may need QNN backend linking)
├── run_flux2_klein_xnnpack.py     # Python inference runner (validation)
├── validate_pipeline.py           # Compare PyTorch vs ExecuTorch outputs
├── prepare_mobile.py              # Pre-tokenize prompts for C++ runner
├── README.md                      # Full deployment guide (CPU/XNNPACK path)
└── CONTEXT_4090.md                # This file
```

## Model Details

- **Model:** `black-forest-labs/FLUX.2-klein-4B`
- **Components:**
  - Text encoder: Qwen3ForCausalLM (27 layers, extracts hidden states at layers 9, 18, 27 → shape (B, 512, 15360))
  - Transformer: Flux2Transformer2DModel (in_channels=128, joint_attention_dim=15360)
  - VAE: AutoencoderKLFlux2 (latent_channels=32, scale_factor=8)
- **Klein specifics:** Always `guidance=None`, 4-step denoising, distilled (no CFG)

## Environment Setup on 4090 Machine

### 1. Clone the repo

```bash
git clone <your-repo-url> dev-vis-gen
cd dev-vis-gen
```

### 2. Install Python dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_export.txt
# requirements_export.txt includes: executorch, torchao, diffusers (git), transformers, etc.
```

### 3. Install ExecuTorch (with QNN support)

```bash
# Clone ExecuTorch at the same level as dev-vis-gen
cd ..
git clone https://github.com/pytorch/executorch.git
cd executorch
git submodule sync && git submodule update --init

# Install Python package
pip install . --no-build-isolation
```

### 4. Get QNN SDK

Download from: https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai
- Requires free Qualcomm developer account
- Download "Qualcomm AI Engine Direct SDK" (QNN SDK), version 2.28+
- Extract to e.g. `/opt/qnn/qnn-<version>/`

```bash
export QNN_SDK_ROOT=/opt/qnn/qnn-<version>
```

### 5. Build ExecuTorch with QNN backend

```bash
cd executorch
rm -rf cmake-out-qnn

cmake -DCMAKE_INSTALL_PREFIX=cmake-out-qnn \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DPYTHON_EXECUTABLE=python3 \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_XNNPACK=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=OFF \
    -DEXECUTORCH_BUILD_COREML=OFF \
    -DEXECUTORCH_BUILD_MPS=OFF \
    -Bcmake-out-qnn .

cmake --build cmake-out-qnn -j$(nproc) --target install --config Release
```

### 6. Build ExecuTorch for Android with QNN (for device deployment)

```bash
export ANDROID_NDK=~/android-ndk-r28c

cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android-qnn \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DPYTHON_EXECUTABLE=python3 \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_XNNPACK=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=OFF \
    -DEXECUTORCH_BUILD_COREML=OFF \
    -DEXECUTORCH_BUILD_MPS=OFF \
    -Bcmake-out-android-qnn .

cmake --build cmake-out-android-qnn -j$(nproc) --target install --config Release
```

## Running the QNN Export

### Option A: Export all components (recommended first attempt)

```bash
cd dev-vis-gen

python export_flux2_klein_qnn.py \
    --output_dir ./exported_flux2_klein_qnn \
    --soc_model SM8650 \
    --num_calibration_passes 20
```

Expected time on 4090:
- Text encoder: ~10-20 min (export + 20 calibration passes)
- Transformer: ~45-90 min (4B params, export + calibration)
- VAE: ~5-10 min

### Option B: Export one component at a time (safer, easier to debug)

```bash
# Start with the smallest (VAE), verify QNN pipeline works
python export_flux2_klein_qnn.py \
    --component vae \
    --soc_model SM8650 \
    --num_calibration_passes 10

# Then transformer
python export_flux2_klein_qnn.py \
    --component transformer \
    --soc_model SM8650 \
    --num_calibration_passes 20

# Then text encoder
python export_flux2_klein_qnn.py \
    --component text_encoder \
    --soc_model SM8650 \
    --num_calibration_passes 10
```

### SOC Model Selection

| Phone | Chipset | --soc_model |
|-------|---------|-------------|
| Samsung Galaxy S24 / S24+ | Snapdragon 8 Gen 3 | SM8650 |
| Samsung Galaxy S23 | Snapdragon 8 Gen 2 | SM8550 |
| Samsung Galaxy S22 | Snapdragon 8 Gen 1 | SM8450 |
| Pixel 8 Pro | Google Tensor G3 | (not Qualcomm, use XNNPACK) |

## Expected Outputs

```
exported_flux2_klein_qnn/
├── text_encoder.pte    # Qwen3 → HTP
├── transformer.pte     # Flux2Transformer → HTP
├── vae_decoder.pte     # VAE decode → HTP
├── vae_bn_stats.pt     # Batch-norm stats
├── tokenizer/          # Qwen3 tokenizer files
└── export_config.json  # Metadata (backend: qnn_htp, soc: SM8650)
```

## Deploying to Device

After export, the .pte files work with the SAME C++ runner (`flux2_main.cpp`), but the ExecuTorch runtime must be built with QNN support and the QNN libraries must be on the device.

```bash
# Push QNN shared libraries (from QNN SDK)
adb shell mkdir -p /data/local/tmp/flux2
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so /data/local/tmp/flux2/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so /data/local/tmp/flux2/
adb push $QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so /data/local/tmp/flux2/

# Push model files
adb push exported_flux2_klein_qnn/*.pte /data/local/tmp/flux2/
adb push exported_flux2_klein_qnn/vae_bn_stats.pt /data/local/tmp/flux2/

# Push tokenizer + pre-tokenize prompt
python prepare_mobile.py \
    --model_dir ./exported_flux2_klein_qnn \
    --prompt "a cat sitting on a windowsill" \
    --output_dir ./exported_flux2_klein_qnn
adb push exported_flux2_klein_qnn/prompt.bin /data/local/tmp/flux2/
adb push exported_flux2_klein_qnn/bn_*.bin /data/local/tmp/flux2/

# Push the QNN-enabled runner binary
adb push cmake-out-android-qnn/examples/models/flux2/flux2_main /data/local/tmp/flux2/

# Run
adb shell "cd /data/local/tmp/flux2 && chmod +x flux2_main && \
    LD_LIBRARY_PATH=. ./flux2_main --model_dir . --tokens prompt.bin --output output.ppm --steps 4"
```

Note: The Hexagon skel library version depends on the Snapdragon chip:
- 8 Gen 3 (SM8650): `libQnnHtpV75Skel.so` (Hexagon V75)
- 8 Gen 2 (SM8550): `libQnnHtpV73Skel.so` (Hexagon V73)
- 8 Gen 1 (SM8450): `libQnnHtpV69Skel.so` (Hexagon V69)

## Potential Issues & Fixes

### Issue: QnnQuantizer API changes
The QNN backend API changes between ExecuTorch versions. If `set_bit8_op_str_override` doesn't exist, check:
```python
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer
help(QnnQuantizer)
```
Look for methods like `set_global_op_override` or `set_quant_config`.

### Issue: Transformer too large for VRAM during calibration
If OOM during calibration, reduce batch size or use gradient checkpointing:
```bash
# Set PYTORCH_CUDA_ALLOC_CONF to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
Or export transformer with fewer calibration passes first:
```bash
python export_flux2_klein_qnn.py --component transformer --num_calibration_passes 5
```

### Issue: torch.export fails on attention ops
Some attention implementations use in-place ops that break tracing. If this happens, check if disabling `flash_attn` helps:
```python
# In the wrapper, force standard attention
pipe.transformer.config._attn_implementation = "eager"
```

### Issue: QnnPartitioner leaves many ops on CPU fallback
This means those ops aren't supported on HTP. Check ExecuTorch logs for "delegating" messages.
Key unsupported ops: some activation functions, dynamic shapes.
Resolution: Use `--skip_node_op_set` or keep those ops on CPU.

### Issue: Context binary generation fails
The QNN partitioner generates context binaries during export (online compilation).
This requires the target SOC to match the host; for cross-compilation you may need
to use the QNN SDK's offline compiler instead:
```bash
qnn-context-binary-generator \
    --model libFlux2Transformer.so \
    --backend libQnnHtp.so \
    --output_dir qnn_context_bins/ \
    --binary_file transformer_ctx
```

## Key Files to Read

- `export_flux2_klein_xnnpack.py` — Reference for wrapper classes and export pipeline
- `export_flux2_klein_qnn.py` — QNN version (the main script to run)
- `flux2_main.cpp` — C++ runner (may need modification to load QNN libraries)
- `README.md` — Original CPU deployment guide

## Validation After Export

```bash
# Compare PyTorch vs QNN ExecuTorch output (1 transformer step)
python validate_pipeline.py \
    --model_dir ./exported_flux2_klein_qnn \
    --component transformer

# Acceptable divergence: < 0.1 mean absolute diff (INT8 introduces ~1-2% error)
# If > 0.5, check calibration quality or quantization config
```

## Notes

- The C++ runner (`flux2_main.cpp`) uses sequential load/unload to prevent OOM.
  This is important for QNN too — don't try to load all three .pte files at once.
- QNN context binaries are SOC-specific. A .pte compiled for SM8650 will NOT
  run on SM8550. Need separate exports per device family.
- HTP performs best with static shapes (no dynamic dims). The current wrappers
  all use fixed shapes — this is correct.
- Keep `--height 512 --width 512` for the initial export. 768+ will significantly
  increase transformer sequence length and export time.
