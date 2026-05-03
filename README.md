# FLUX.2-klein-4B on Snapdragon

Export and run the 4B-parameter FLUX.2-klein rectified-flow image generator on a
Samsung Galaxy S26 Ultra (Snapdragon 8 Elite Gen 5 / SM8850) as a single standalone
ARM64 binary. The repo supports two backends:

| Backend   | Target              | Status                                        | Recipe |
|-----------|---------------------|-----------------------------------------------|--------|
| XNNPACK   | ARM CPU (NEON/i8mm) | Working, ships                                | text_encoder + transformer + vae, w8a8 dynamic |
| QNN HTP   | Hexagon V81 NPU     | Transformer `.pte` compiled, awaiting device test | linear-only-discard a16w8, host SNR +23.1 dB / cos 0.998 |

Both paths share the same on-device runner (`runner/flux2_main.cpp`) and the same
tokenizer / VAE batch-norm / `prompt.bin` preparation flow. The difference is which
ExecuTorch delegate is embedded in the `.pte` files — the runner is backend-agnostic.

## Layout

```
export_flux2_klein_xnnpack.py     XNNPACK CPU export (PT2E w8a8 dynamic)
export_flux2_klein_qnn.py         QNN HTP shared utilities (wrappers, calibration loaders, rotary head-split)
export_flux2_klein_qnn_v12.py     QNN HTP v1.2 export (default a16w8 single-context)
export_flux2_klein_qnn_lin_only.py  QNN HTP a8w8 linear-only-discard export (host +3.6 dB / cos 0.82)
export_flux2_klein_qnn_a16w8.py     QNN HTP a16w8 linear-only-discard export (host +23.1 dB / cos 0.998)
collect_calibration_data.py       Collect real-prompt activations for QNN static PTQ
prepare_mobile.py                 Tokenize a prompt to prompt.bin (+ BN stats copy)
test_pte_host.py                  Host-side smoke test for .pte files (ExecuTorch Runtime)

diag_*.py                         Host PT2E diagnostics — per-Linear SNR, observer sweep,
                                  block-level promotion, override-applies sanity checks, etc.

runner/flux2_main.cpp             On-device pipeline (sequential mmap/load of 3 .pte files)
runner/CMakeLists.txt             Builds flux2_runner against an ExecuTorch install tree
runner/deploy_to_device.sh        Build + push + run wrapper (auto-detects V75/V79/V81)

stage_phone_ship.sh               Stage a flat, self-contained phone bundle at flux2_phone_ship/
push_htp.sh                       Older direct-push script (kept for reference)

executorch/                       Local clone with QNN backend patches (see "Patches" below)
qairt/2.45.0.260326/              Qualcomm AI Engine Direct SDK runtime libraries

CONTEXT_FOR_AGENTS.md             Load-bearing context for follow-up work; read before diving in
V12_PATH.md                       v1.2.0 ExecuTorch setup notes (which venv, which patches)
```

## End-to-end pipeline

The runner (`runner/flux2_main.cpp`) does the following on device, loading at most
one model at a time to stay under memory budget:

```
1. Read prompt.bin (host-tokenised input_ids + attention_mask)
2. Load text_encoder.pte  -> forward -> keep prompt embeds + pooled embeds
3. Load transformer.pte   -> for t in schedule: forward(h, embeds, t, img_ids, txt_ids)
                              -> rectified-flow update of h
4. Unpatchify h -> latent [1, 32, H/8, W/8]
5. Load vae_decoder.pte   -> forward -> BN un-normalise -> clamp [-1,1] -> PPM
```

Peak working set is the transformer (~4 GB int8). Each `.pte` is `mmap`'d (lazy),
and the previous model's `Module` is destroyed before the next one loads.

## Export - XNNPACK CPU path (working)

```bash
# Export all three components with int8 weights + dynamic-int8 activations.
python export_flux2_klein_xnnpack.py \
    --output_dir ./exported_flux2_klein_xnnpack \
    --quantize \
    --component all
```

Quantization recipe used for every component:

- Weights: int8 per-output-channel symmetric, frozen at export time.
- Activations: int8 per-token symmetric, **dynamic** - XNNPACK runs
  `choose_qparams` at every forward and picks scales from the live min/max.
  No calibration dataset, no precomputed scales.
- Text encoder embedding table is additionally int8 per-channel (separate pass
  using `_QuantizedEmbedding` + `embedding_byte` on-device).

This is exactly what `get_symmetric_quantization_config(is_per_channel=True,
is_dynamic=True)` from `XNNPACKQuantizer` produces. The resulting graph has the
`quantized_decomposed.*` op triplet that `XnnpackPartitioner(per_op_mode=True,
config_precisions=DYNAMIC_QUANT)` delegates into XNNPACK's native dq-linear kernel.

Key implementation details in `export_component_to_xnnpack()`:

1. Wrap each submodule to a pure positional-tensor `forward` (`Qwen3TextEncoderWrapper`,
   `Flux2TransformerWrapper`, `VAEDecoderWrapper`). `torch.export` dislikes `**kwargs`,
   dict returns, and dataclass outputs.
2. Pre-export the fp32 module with `export_for_training(...).module()`.
3. `prepare_pt2e(gm, XNNPACKQuantizer)` - inserts observers on every linear's
   weight constant and activation input.
4. `_initialise_weight_observers(gm)` - walk the graph, find each weight observer
   whose single input is a `get_attr`, invoke the observer with just the tensor.
   We skip the activation-observer forward because for dynamic activations the
   observer is a `PlaceholderObserver` (no-op at calibration time). Running a full
   forward on a 4B transformer after `prepare_pt2e` re-triggers decomposition bugs
   and eats >30 GB RAM.
5. `convert_pt2e(gm)` - fuses observers into the Q/DQ op triplet.
6. `DuplicateDynamicQuantChainPass()(gm)` - needed when one activation feeds Q, K, V
   projections; duplicates the Q/DQ chain so each linear gets its own scale.
7. Final `torch.export.export()` inside `sdpa_kernel([SDPBackend.MATH]) + no_grad()` -
   the MATH SDPA backend decomposes cleanly under export.
8. `to_edge_transform_and_lower(XnnpackDynamicallyQuantizedPartitioner + fp32 fallback)`.
9. `edge_program.to_executorch()` -> write `.pte`.

### Text encoder quirk: Qwen3 causal mask

Qwen3's `create_causal_mask` builds `kv_arange = torch.arange(...) + kv_offset`
which gets promoted to float under PT2E re-export, and then
`padding_mask[batch_idx, kv_idx]` explodes with *"tensors used as indices must be
long, int, byte or bool"*. `Qwen3TextEncoderWrapper.forward` bypasses this by
pre-building the 4-D additive mask and passing it as a **dict**
(`attention_mask={"full_attention": additive}`), which hits the branch in
`Qwen3Model.forward` that takes the mask as-is and never calls `create_causal_mask`.

### Resulting artefacts

```
text_encoder.pte   ~3.94 GB   Qwen3-4B, int8 embeddings + w8a8 dynamic linears
transformer.pte    ~3.88 GB   Flux2Transformer2DModel, w8a8 dynamic linears
vae_decoder.pte    ~0.20 GB   AutoencoderKLFlux2 decoder, w8a8 dynamic linears
```

## Export - QNN HTP path (Hexagon DSP)

The current HTP path uses **ExecuTorch v1.2.0** + the patched local `executorch/`
tree. Setup notes in `V12_PATH.md`. Two production export scripts:

```bash
# 1. Collect real activations (once).
python collect_calibration_data.py \
    --output_dir ./calibration_data \
    --num_timesteps 4

# 2a. a16w8 export (RECOMMENDED — host SNR +23.1 dB / cos 0.998).
cd /tmp && \
  FLATC_EXECUTABLE=$REPO/.venv-et12/lib/python3.10/site-packages/executorch/data/bin/flatc \
  FLUX_ROTARY_HEAD_SPLIT=2 \
  $REPO/.venv-et12/bin/python $REPO/export_flux2_klein_qnn_a16w8.py \
    --output_dir $REPO/exported_flux2_klein_qnn_a16w8 \
    --calibration_dir $REPO/calibration_data

# 2b. a8w8 export (fallback — host SNR +3.6 dB / cos 0.82, half VTCM pressure).
$REPO/.venv-et12/bin/python $REPO/export_flux2_klein_qnn_lin_only.py ...
```

**Output:** single 3.94 GB `transformer.pte` for a16w8 (3.97 GB for a8w8). Both
target SM8850 (V81) and use a single QNN context — no transformer sharding
needed in this path.

### Recipe details

The a16w8 production script does:

1. `torch.export.export(model, sample_inputs, strict=True).module()` — capture.
2. `QnnQuantizer(backend=kHtpBackend, soc_model=SM8850).set_default_quant_config(
   QuantDtype.use_16a8w, is_linear_per_channel=True, act_observer=HistogramObserver)`.
3. **`add_discard_ops` on every quant_op except `aten.linear.default`,
   `aten.conv2d.default`, `aten.conv1d.default`** — the key step. By default
   `QnnQuantizer` annotates ~165 op types (mul, add, layer_norm, softmax, bmm,
   matmul, …); discarding the 162 non-Linear ops lets them flow in fp on HTP and
   eliminates compounding 256-level rounding through 25 transformer blocks.
   Mirrors torchao's `Int8DynamicActivationInt8WeightConfig` policy (Linear-only)
   on the QNN PT2E flow. Without this, host SNR is **−2.6 dB / cos 0.18**
   (uncorrelated). With it: **+23.1 dB / cos 0.998 at a16w8**.
4. `prepare_pt2e` → calibrate (5 real samples) → `convert_pt2e`.
5. `generate_htp_compiler_spec(use_fp16=False, use_dlbc=True)` →
   `to_edge_transform_and_lower_to_qnn` → `to_executorch` → write `.pte`.

### Two non-obvious gotchas to know about

**Rotary head-split** (`FLUX_ROTARY_HEAD_SPLIT=2`). FLUX's rotary embedding
multiplies a `(1, 1536, 24, 128)` tensor by a `(1, 1536, 1, 128)` cos/sin
broadcasting on the **head** dim. HTP's element-wise tiler doesn't fall back
across axes when its first-choice slice doesn't fit; at int16 the tile is ~9.4 MB
(over the 8 MB VTCM budget) and compile fails. The wrapper pre-splits the head
dim into N halves before the multiply (bit-exact identical math, verified at
FX-graph level) so each tile drops to ~4.7 MB and tiles cleanly. `N=2` is enough
for the current shapes.

**Two ExecuTorch venvs.** `.venv` ships v0.6.0 (incomplete for our path —
missing `to_edge_transform_and_lower_to_qnn`). `.venv-et12` ships v1.2.0 (the
correct one). The local `executorch/` tree is also at v1.2.0 with the patches
listed below; **keep it on `sys.path`** so its patched
`backends/qualcomm/builders/op_layer_norm.py` shadows the unpatched venv copy
(otherwise the partitioner crashes with `'NoneType' object has no attribute
'name'` on FLUX's `LayerNorm(elementwise_affine=False)` modules).

### What got tried and didn't help

For continuity with future work — these were dead ends:

- **Selective a16w8 promotion.** Built a per-Linear local SNR ranking, identified
  `context_embedder` (7.15 dB) and 25 output projections (10–25 dB) as outliers,
  promoted them to a16w8 individually and as a group. Verified the
  `set_submodule_qconfig_list` override actually applies (input observer
  quant_max goes 255 → 65535). Despite that, end-to-end SNR didn't move (~0 dB).
  **Lesson:** quant noise in this transformer is broadly distributed across all
  109 Linears; partial promotion shaves a fraction of cumulative error
  invisible end-to-end. Promote all or none.
- **Per-block promotion of D00–D04 to 16a8w.** Got *worse* SNR (−2.92 dB);
  boundary requant overhead at int8/int16 transitions ate the local gain.
- **Observer sweep** (Histogram / MinMax / MovingAvg / QNN default). All
  converged to similar SNR; observer choice not the bottleneck.
- **Host-side outlier mitigation** (SmoothQuant / Hadamard / per-token L2)
  before the `add_discard_ops` fix. No gain — masked by the over-annotation
  noise floor. Not retested afterward.
- **Older multi-context sharding** (5-shard transformer split) was needed under
  v0.6's monolithic-graph scheduler crashes. v1.2.0 + linear-only-discard
  compiles a single context cleanly; sharding deprecated for this path.

### Patches applied to `executorch/backends/qualcomm/`

Stock ExecuTorch QNN backend has several issues that crash on FLUX. Patched
locally in the in-tree `executorch/` clone (matches v1.2.0 + fixes):

| File | Change |
|------|--------|
| `serialization/qc_schema.py`, `qc_compiler_spec.fbs` | Add `HtpArch.V81 = 81` and `QcomChipset.SM8850 = 87`, plus `_soc_info_table` entry mapping SM8850 -> V81 / 8 MB VTCM. |
| `_passes/lift_constant_scalar_operands.py` | Add `aten.pow.Scalar` -> `pow.Tensor_Tensor` to `SCALAR_OPS`; guard `hasattr(n.target, "_schema")` in `_lift()`; guard `isinstance(first_arg, fx.Node)` before reading `.meta`. |
| `partition/qnn_partitioner.py` | In `is_node_supported`, early-return if `node.target.__name__ not in self.node_visitors`. |
| `quantizer/annotators.py`, `quantizer/annotators/htp_rules.py` | In `_mark_nodes_as_annotated`, `if node is None: continue`. LayerNorm.annotate guards `weight_node`/`bias_node` with `is not None`. |
| `builders/op_layer_norm.py` | When weight is `None`, synthesize `torch.ones(normalized_shape)`; same for bias with zeros. |
| `exir/operator/util.py` | Catch `AttributeError` around the torchao import (was crashing on `torch.ops.torchao.dequantize_affine` not being registered). |
| `third-party/ao` submodule | Bumped to v0.17.0-rc1 to match what executorch v1.2.0 expects (`torchao.quantization.pt2e.*`). |
| `builders/node_visitor.py`, `utils/utils.py`, `exir/backend/backend_api.py`, `exir/lowered_backend_module.py` | Misc. guards around shard-boundary metadata and fallback nodes (legacy from sharded path; harmless under single-context). |

These patches have not been submitted upstream.

## Build the device runner

```bash
export ANDROID_NDK=/path/to/android-ndk-r28c   # or r25c+
export QNN_SDK_ROOT=$(pwd)/qairt/2.45.0.260326
export EXECUTORCH_ROOT=$(pwd)/executorch

# One-time: build ExecuTorch for Android with both delegates.
cmake -B executorch/install-android \
    -S executorch \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
    -DCMAKE_INSTALL_PREFIX=executorch/install-android
cmake --build executorch/install-android -j"$(nproc)" --target install

# Build the flux2 runner against that install tree.
cmake -B runner/build-android -S runner \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 \
    -DCMAKE_BUILD_TYPE=Release \
    -Dexecutorch_DIR=executorch/install-android/lib/cmake/ExecuTorch \
    -DEXECUTORCH_ROOT=executorch
cmake --build runner/build-android -j"$(nproc)"
```

Output: `runner/build-android/flux2_runner` (static ARM64 binary).

## Stage + deploy

```bash
# 1. Stage a self-contained bundle (auto-detects V75 / V79 / V81 from export_config.json).
MODEL_DIR=./exported_flux2_klein_qnn_v81 ./stage_phone_ship.sh

# 2. Push + run.
ADB=./.tools/platform-tools/adb ./flux2_phone_ship/push.sh
adb shell "cd /data/local/tmp/flux2 && \
    export LD_LIBRARY_PATH=/data/local/tmp/flux2:\$LD_LIBRARY_PATH && \
    export ADSP_LIBRARY_PATH='/data/local/tmp/flux2;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp' && \
    ./flux2_runner --model_dir . --tokens prompt.bin \
                   --output output.ppm --steps 4 --seed 42"
adb pull /data/local/tmp/flux2/output.ppm ./
```

`stage_phone_ship.sh` copies:

- `*.pte`, `export_config.json`, `vae_bn_stats.pt`, `tokenizer/`, `prompt.bin`,
  `bn_mean.bin`, `bn_var.bin`, `flux2_runner`
- `libqnn_executorch_backend.so` (from `executorch/install-android/lib/`)
- `libQnnHtp.so`, `libQnnSystem.so`, `libQnnHtpPrepare.so`,
  `libQnnHtpNetRunExtensions.so`, `libQnnHtp${ARCH}Stub.so`
  (from `qairt/2.45.0.260326/lib/aarch64-android/`)
- All `qairt/2.45.0.260326/lib/hexagon-${arch}/unsigned/*.so` (Hexagon skel)
- `push.sh` (self-contained adb push script inside the bundle)

Where `${ARCH}` is `v75` (SM8650), `v79` (SM8750), or `v81` (SM8850) based on
`export_config.json#soc_model`.

## Runner flags

```
--model_dir   Directory with .pte + export_config.json + prompt.bin + bn_*.bin
--tokens      Path to prompt.bin (host-side tokenised)
--output      Output PPM path (default output.ppm)
--height W    Image dims; must match export (default 512)
--width H
--steps N     Rectified-flow steps (Klein is distilled, 4 is enough)
--seed N      RNG seed for initial noise
```

## Host-side smoke test

```bash
python test_pte_host.py --model_dir ./exported_flux2_klein_xnnpack
```

Loads each `.pte` in the ExecuTorch Python runtime, runs one forward pass with
dummy tensors, and checks the output shapes. Catches misformed `.pte` or
delegate-init failures before burning a 10-minute adb push.
