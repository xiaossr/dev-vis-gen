# FLUX.2-klein-4B on Snapdragon

Export and run the 4B-parameter FLUX.2-klein rectified-flow image generator on a
Samsung Galaxy S26 Ultra (Snapdragon 8 Elite Gen 5 / SM8850) as a single standalone
ARM64 binary. The repo supports two backends:

| Backend   | Target              | Status                           | Component split |
|-----------|---------------------|----------------------------------|-----------------|
| XNNPACK   | ARM CPU (NEON/i8mm) | Working                          | text_encoder + transformer + vae on CPU, w8a8 dynamic |
| QNN HTP   | Hexagon V81 NPU     | Transformer compile in progress  | text_encoder + vae on HTP, transformer sharded 5x |

Both paths share the same on-device runner (`runner/flux2_main.cpp`) and the same
tokenizer / VAE batch-norm / `prompt.bin` preparation flow. The difference is which
ExecuTorch delegate is embedded in the `.pte` files — the runner is backend-agnostic.

## Layout

```
export_flux2_klein_xnnpack.py    XNNPACK CPU export (PT2E w8a8 dynamic)
export_flux2_klein_qnn.py        QNN HTP export (PT2E w8a8 static, optional multi-context sharding)
collect_calibration_data.py      Collect real-prompt activations for QNN static PTQ
prepare_mobile.py                Tokenize a prompt to prompt.bin (+ BN stats copy)
test_pte_host.py                 Host-side smoke test for .pte files (ExecuTorch Runtime)

runner/flux2_main.cpp            On-device pipeline (sequential mmap/load of 3 .pte files)
runner/CMakeLists.txt            Builds flux2_runner against an ExecuTorch install tree
runner/deploy_to_device.sh       Build + push + run wrapper (auto-detects V75/V79/V81)

stage_phone_ship.sh              Stage a flat, self-contained phone bundle at flux2_phone_ship/
push_htp.sh                      Older direct-push script (kept for reference)

executorch/                      Local clone with QNN backend patches (see "Patches" below)
qairt/2.45.0.260326/             Qualcomm AI Engine Direct SDK runtime libraries
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

QNN needs *static* activation scales, so this path requires a calibration pass.

```bash
# 1. Collect real activations (once).
python collect_calibration_data.py \
    --output_dir ./calibration_data \
    --num_timesteps 4

# 2. Export. Target SM8850 = V81 = Snapdragon 8 Elite Gen 5.
python export_flux2_klein_qnn.py \
    --component all \
    --soc_model SM8850 \
    --quant_dtype 8a8w \
    --calibration_dir ./calibration_data \
    --output_dir ./exported_flux2_klein_qnn_v81 \
    --transformer_shards 5
```

Per-component pipeline in `export_component_to_qnn()`:

1. `torch.export.export()` -> GraphModule.
2. `_decompose_layer_norm()` - decompose `aten.native_layer_norm` into primitive
   ops. Stock HTP rejects rank-3 LN (Flux uses `(B, S, C)`), and the decomposed
   form stays on DSP.
3. `QnnQuantizer(QuantDtype.use_8a8w, MovingAverageMinMaxObserver)` +
   `prepare_pt2e`.
4. Calibration - either real activations from `--calibration_dir` or synthetic
   perturbations of `sample_inputs`.
5. `convert_pt2e`.
6. `_remove_int_quantize_nodes` - strip spurious `quantize_per_tensor` /
   `dequantize_per_tensor` nodes on integer tensors that would break re-export.
7. Re-export with `capture_program()` using QNN's decomp table and edge config.
   Falls back to direct `torch.export.export(strict=False)` + `qnn_edge_config()`
   if `capture_program` hits a dynamo regression.
8. **Multi-context sharding** (`--transformer_shards N`, transformer only):
   `_insert_flux_transformer_fallbacks()` walks the graph, finds transformer
   block boundaries via `nn_module_stack`, and inserts
   `exir_ops.edge.llama.fallback.default` nodes at `N-1` evenly-spaced
   block boundaries (e.g. blocks [5, 10, 15, 20] for 30 blocks / 5 shards).
   Adjacent tensors are tagged `QCOM_QUANTIZED_IO = torch.uint8` so the split
   stays fixed-point end-to-end. This is needed because a monolithic AOT compile
   of the whole transformer trips Qualcomm's host-side scheduler
   (`RouterX86 graph prepare failed 18`, 88+ *"could not create op"* errors).
9. `generate_htp_compiler_spec(use_fp16=..., use_dlbc=use_sharding,
   use_multi_contexts=use_sharding)` ->
   `QnnPartitioner(..., skip_node_op_set={"llama.fallback.default", ...})`.
10. `to_backend(edge_prog, qnn_partitioner)` - partitions on support and lowers
    each QNN partition to a separate context binary.
11. `update_spill_fill_size(delegated_ep)` if sharded - reserves a single
    spill/fill buffer sized to the max across shards.
12. `EdgeProgramManager.to_executorch()` -> `.pte`.

### Text encoder / VAE export notes

- Text encoder and VAE decoder export cleanly as single HTP contexts with
  `online_prepare=True` by default (graph preparation happens on device on first
  load). Both are sensitive to `fp16_components` - pass `--fp16_components vae`
  if the VAE's int8 output is visibly wrong, at the cost of 2x `.pte` size.
- The transformer is always `online_prepare=False` (pure AOT). On-device online
  prepare for the transformer was tried and turned out unreliable at this model
  size.

### Patches applied to `executorch/backends/qualcomm/`

Stock ExecuTorch QNN backend has several issues that crash on FLUX. Patched
locally in the in-tree `executorch/` clone:

| File | Change |
|------|--------|
| `serialization/qc_schema.py`, `qc_compiler_spec.fbs` | Add `HtpArch.V81 = 81` and `QcomChipset.SM8850 = 87`, plus `_soc_info_table` entry mapping SM8850 -> V81 / 8 MB VTCM. |
| `_passes/lift_constant_scalar_operands.py` | Add `aten.pow.Scalar` -> `pow.Tensor_Tensor` to `SCALAR_OPS`; guard `hasattr(n.target, "_schema")` in `_lift()`; guard `isinstance(first_arg, fx.Node)` before reading `.meta`. |
| `partition/qnn_partitioner.py` | In `is_node_supported`, early-return if `node.target.__name__ not in self.node_visitors`. |
| `quantizer/annotators.py` | In `_mark_nodes_as_annotated`, `if node is None: continue`. LayerNorm.annotate guards `weight_node`/`bias_node` with `is not None`. |
| `builders/op_layer_norm.py` | When weight is `None`, synthesize `torch.ones(normalized_shape)`; same for bias with zeros. |
| `builders/node_visitor.py`, `utils/utils.py`, `exir/backend/backend_api.py`, `exir/lowered_backend_module.py` | Misc. guards around shard-boundary metadata and fallback nodes. |

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
