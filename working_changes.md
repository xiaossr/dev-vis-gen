# FLUX.2-klein-4B on Snapdragon 8 Elite — working export

## 0. What we're doing and how

**Goal.** Run the 4-billion-parameter FLUX.2-klein diffusion pipeline **fully on a
Samsung Galaxy S26 Ultra (Snapdragon 8 Elite Gen 5)** with only the phone's CPU.
No server, no NPU/DSP dependency. The model should load, tokenize a prompt, run
4 rectified-flow sampling steps, and write a PPM image — all in one standalone
ARM64 binary.

**How.** ExecuTorch + XNNPACK (ARM NEON / i8mm CPU kernels). We take the HF
pipeline, wrap each sub-network to a plain positional-tensor forward, quantize
it with PyTorch's PT2E flow, serialize each to a `.pte` with XNNPACK as the
delegate, and link a small C++ runner that loads the three `.pte` files and
stitches the sampling loop.

Three artefacts ship to the phone:

```
text_encoder.pte    3.94 GB   Qwen3-4B, int8 embeddings + w8a8 dynamic linears
transformer.pte     3.88 GB   Flux2Transformer2DModel, w8a8 dynamic linears
vae_decoder.pte     0.20 GB   AutoencoderKLFlux2 (decoder), w8a8 dynamic linears
```

plus tokenizer files, VAE batch-norm stats, a tokenized `prompt.bin`, and
`flux2_runner` (139 MB static ARM64).

### Quantization recipe

All three components use the **same PT2E recipe**:

- **Weights:** int8 per-output-channel symmetric, frozen at export time.
- **Activations:** int8 per-token symmetric, **dynamic** — XNNPACK runs
  `choose_qparams` at every forward pass and picks scales from the live
  activation's min/max. No calibration dataset. No precomputed scales.
- Embedding table in the text encoder is also int8 per-channel (separate pass
  via `_QuantizedEmbedding` + `embedding_byte` kernel on-device).

This is exactly what `get_symmetric_quantization_config(is_per_channel=True,
is_dynamic=True)` from `XNNPACKQuantizer` gives you. It produces the
`quantized_decomposed.choose_qparams.tensor` / `quantize_per_tensor.tensor` /
`dequantize_per_tensor.tensor` op triplet, which the
`XnnpackPartitioner(config_precisions=DYNAMIC_QUANT, per_op_mode=True)`
recognises and delegates into XNNPACK dq-linear kernels.

### End-to-end export pipeline (one component)

1. **Load fp32 pipeline** with `diffusers`.
2. **Wrap** the submodule so its `forward` takes only positional tensors —
   `Qwen3TextEncoderWrapper`, `Flux2TransformerWrapper`, `VAEDecoderWrapper`.
   Wrapping matters because `torch.export` does not like `**kwargs` /
   dataclass returns / dicts.
3. **Pre-export the float module** with `export_for_training(...).module()`.
   This "unpacks" the nn.Module into a graph that PT2E can decorate.
4. **`prepare_pt2e(gm, XNNPACKQuantizer)`** — inserts observers on every linear's
   weight constant and on every linear's activation input.
5. **Observe weights only** (`_initialise_weight_observers`). For dynamic
   activations the activation observer is a `PlaceholderObserver` (no-op at
   calibration time — real scales happen on device), so we only need to run
   weight observers. Running a full calibration forward would re-trigger
   transformer decomposition bugs; we skip it safely.
6. **`convert_pt2e(gm)`** — fuses observers into `quantize` / `dequantize` /
   `choose_qparams` ops around each linear.
7. **`DuplicateDynamicQuantChainPass()(gm)`** — if one activation feeds
   multiple linears (e.g. Q, K, V projections all reading the same hidden
   state), this pass duplicates the Q/DQ chain so each linear gets its own
   scale and the partitioner can fuse cleanly.
8. **Final `torch.export.export(gm, sample_inputs)`** inside
   `sdpa_kernel([SDPBackend.MATH]) + torch.no_grad()` — the MATH backend is
   important because flash/efficient SDPA don't decompose cleanly under
   export. A `strict=False` fallback catches dynamo regressions.
9. **`to_edge_transform_and_lower`** with two partitioners: one DQ-quant aware,
   one greedy fp32 fallback. XNNPACK absorbs everything it can; anything left
   over goes to ExecuTorch's portable CPU kernels.
10. **Serialize** with `edge_program.to_executorch()` → write `.pte`.
    Uses `flatc` under the hood.

### On-device

The runner (`runner/flux2_main.cpp`) does:

```
1. load prompt.bin (token IDs + attention mask from host tokenization)
2. load text_encoder.pte          — scoped {}, destroyed before next model
3. run it once → keep prompt embeds
4. load transformer.pte           — scoped {}
5. for step in 0..3:
     forward(hidden_states, prompt_embeds, timestep, img_ids, txt_ids)
     update hidden_states via rectified-flow step
6. unpatchify hidden_states → latents
7. load vae_decoder.pte           — scoped {}
8. decode → BN stats normalize → clamp → write PPM
```

Each `.pte` is mmap'd, and only one is resident at a time. Peak working set is
~4 GB (the transformer), well under the S26 Ultra's 12–16 GB RAM budget.

---

## 1. Why our path works and April's didn't — the core change

**April's commit `f287d46` ("w8a8 for flux transformer and vae")** added
`apply_w8a8_quantization` which uses **torchao source transforms**:

```python
from torchao.quantization import (
    Int8DynamicActivationIntxWeightConfig,  # later Int8DynamicActivationInt8WeightConfig
    quantize_,
)
from torchao.utils import unwrap_tensor_subclass

quantize_(model, Int8DynamicActivationInt8WeightConfig(), filter_fn=is_linear)
unwrap_tensor_subclass(model)
```

That path swaps each `nn.Linear.weight` for a tensor subclass wrapper whose
matmul produces `aten._int_mm` plus torchao's own
`quant.choose_qparams_affine` / `quant.quantize_affine` /
`quant.dequantize_affine` ops. When that traced graph reaches the XNNPACK
partitioner, **the partitioner has no pattern for the `choose_qparams_affine`
op**. It's registered as "do not decompose" but also "not supported", so
`to_edge_transform_and_lower` aborts with

```
RuntimeError: Node quant_choose_qparams_affine_default with op
  <EdgeOpOverload: quant.choose_qparams_affine.default>
  was not decomposed or delegated.
```

That's the blocking failure in `logs/export_all.log` (line 46).

**Our fix: switch to PT2E graph-level quantization** — the path XNNPACK's
partitioner was actually designed around. Instead of torchao's tensor
subclasses, we use `prepare_pt2e` + `convert_pt2e` driven by
`XNNPACKQuantizer(is_per_channel=True, is_dynamic=True)`. This produces the
`quantized_decomposed.*` namespace ops (not `quant.*`), which
`XnnpackPartitioner(config_precisions=DYNAMIC_QUANT, per_op_mode=True)`
recognises and delegates into the native dq-linear kernel. `apply_w8a8_quantization`
is left in the file as dead code but **is never called** from `main()` —
`main()` now passes `dynamic_w8a8=True` into `export_component_to_xnnpack`,
which is where the full PT2E flow lives.

### Summary of the diff vs April

| Area | April (`f287d46`) | Ours |
|---|---|---|
| Quantizer | torchao source transforms on `nn.Linear` | PT2E graph pass (`XNNPACKQuantizer`) |
| Op namespace | `quant.choose_qparams_affine`, `aten._int_mm` | `quantized_decomposed.choose_qparams.tensor`, `quantize_per_tensor`, `dequantize_per_tensor` |
| XNNPACK partitioner | Can't delegate → abort | Delegates dq-linear natively |
| Calibration needed | No (dynamic activations) | No (dynamic activations) |
| Weight scope | linears only (good) | linears only |
| Embedding quant (TE) | separate `_QuantizedEmbedding` pass | **unchanged** — still works |
| `DuplicateDynamicQuantChainPass` | not run | run after `convert_pt2e` — needed for QKV shared input |
| SDPA backend during export | default | `sdpa_kernel([SDPBackend.MATH])` + `torch.no_grad()` |
| Causal mask construction in TE | HF's `create_causal_mask` is traced | **Bypassed** — we precompute a 4D additive mask in the wrapper |

Everything else (wrappers, VAE BN export, runner structure, sequential module
loading) is unchanged from April's scaffolding — those parts already worked.

---

## 2. Bugs we hit and how we patched them

All changes in this section are in `export_flux2_klein_xnnpack.py`,
`test_pte_host.py`, or `runner/` unless noted.

### 2.1 Unsupported torchao ops in the XNNPACK partitioner

**Symptom.** `RuntimeError: Node quant_choose_qparams_affine_default ... was
not decomposed or delegated.` (see `logs/export_all.log`).

**Cause.** torchao's `Int8DynamicActivationInt8WeightConfig` emits
`quant.choose_qparams_affine` + `aten._int_mm`, neither of which
XNNPACK's partitioner handles.

**Fix.** Replace the torchao source transform with PT2E. In
`export_component_to_xnnpack`, when `dynamic_w8a8=True`:

```python
model = export_for_training(model, sample_inputs).module()
q = XNNPACKQuantizer()
q.set_global(get_symmetric_quantization_config(is_per_channel=True,
                                                is_dynamic=True))
model = prepare_pt2e(model, q)
_initialise_weight_observers(model)
model = convert_pt2e(model)
DuplicateDynamicQuantChainPass()(model)
```

Partitioner is run with `config_precisions=DYNAMIC_QUANT, per_op_mode=True`.

### 2.2 Calibration forward on a 4B model re-triggers decomposition bugs

**Symptom.** `prepare_pt2e` wants activation observers fed. Running one
sample input through the 4B transformer after `prepare_pt2e` hit various
dynamo/decomposition failures and ate >30 GB RAM.

**Cause.** For `is_dynamic=True`, the activation observer is a
`PlaceholderObserver` — it *literally does nothing* at observe time because the
real scales will be computed on-device. The calibration forward is pure
overhead and exercises fragile code paths.

**Fix.** `_initialise_weight_observers(gm)` walks the graph, finds each
weight observer submodule whose single input is a `get_attr` (the constant
weight tensor), resolves the tensor by attribute walk, and invokes the
observer just on that. 253 weight observers initialized for the TE, 109 for
the transformer, 4 for the VAE — no activation forward needed, no OOM,
no decomposition regression.

### 2.3 Qwen3 causal mask becomes a float index under PT2E re-export

**Symptom.** After `convert_pt2e` on the text encoder, the final
`torch.export.export()` fails with

```
aten.index.Tensor ... tensors used as indices must be long, int, byte or bool
```

pointing into `transformers/masking_utils.py:157`:
`padding_mask[batch_idx, kv_idx]` — `kv_idx` comes in as float.

**Cause.** transformers' `create_causal_mask` (called from
`Qwen3Model.forward`) builds `kv_arange = torch.arange(kv_length) + kv_offset`.
Under PT2E's re-export, the fake-tensor pass promotes that to float somewhere
in the graph, and the subsequent tensor index explodes.

**Fix.** Bypass `create_causal_mask` entirely. In
`Qwen3TextEncoderWrapper.forward`, precompute the 4-D additive mask in the
wrapper:

```python
causal_bool  = torch.ones(T, T, dtype=torch.bool, device=...).tril()
pad_bool     = attention_mask.to(torch.bool).view(B, 1, 1, T)
mask_bool    = causal_bool.view(1, 1, T, T) & pad_bool
additive_mask = torch.where(mask_bool, 0.0, -inf)  # as float32

output = self.text_encoder(
    input_ids=input_ids,
    attention_mask={"full_attention": additive_mask},   # <-- dict form
    output_hidden_states=True,
    use_cache=False,
)
```

Passing `attention_mask` as a `dict` hits Qwen3Model line 403
(`if not isinstance(causal_mask_mapping := attention_mask, dict)`) — that
branch takes the mask as-is and never calls `create_causal_mask`.

### 2.4 `strict=False` fallback on the final export

Even with the mask fix, PT2E-converted models occasionally trip dynamo on
the second `torch.export`. We wrap the final `export(...)` in a
try/except and retry with `strict=False`, which uses the aot-dispatch path
and works on the already-traced graph. No-op when strict export succeeds.

### 2.5 Shared activations feeding multiple linears → partitioner splits awkwardly

**Symptom.** In attention blocks, the same input feeds three linears
(Q, K, V). After `convert_pt2e`, a single `choose_qparams` + `quantize` +
`dequantize` chain feeds into all three. The XNNPACK partitioner ends up
leaving Q/DQ nodes outside the delegated subgraph.

**Fix.** `DuplicateDynamicQuantChainPass()(model)` after `convert_pt2e`.
It walks users of each `dequantize` node and duplicates the upstream
Q/DQ chain per-consumer so every linear has its own chain. ExecuTorch ships
this pass in `backends/transforms/duplicate_dynamic_quant_chain.py`.

### 2.6 `flatc` not on PATH when launching via interpreter directly

**Symptom.** `FileNotFoundError: [Errno 2] No such file or directory: 'flatc'`
— right after "Lowering to XNNPACK backend …".

**Cause.** ExecuTorch's flatbuffers serializer calls `subprocess.run(['flatc',
...])`. `flatc` ships inside `.venv/bin/flatc`, but `nohup .venv/bin/python
…` doesn't put `.venv/bin` on PATH (you'd get that from `source
.venv/bin/activate`).

**Fix.** `_ensure_flatc_on_path()` runs at import time: if
`FLATC_EXECUTABLE` isn't already set, look in `sys.prefix/bin/flatc`, then
`site-packages/executorch/data/bin/flatc`, then `shutil.which('flatc')`, and
set the env var. No more flaky env-prefix requirement.

### 2.7 Host verification runtime missing `quantized_decomposed::embedding_byte.dtype_out`

**Symptom.** `test_pte_host.py` on `text_encoder.pte`:

```
Missing operator: [3] quantized_decomposed::embedding_byte.dtype_out
RuntimeError: loading method forward failed with error 0x14
```

**Cause.** The embedding-quantize pass emits `embedding_byte`, whose kernel
lives in `libquantized_ops_aot_lib.so`. The host ExecuTorch python runtime
doesn't load it by default.

**Fix.** In `test_pte_host.py`, `_load_quantized_ops_lib()`:

1. Import `executorch.extension.pybindings._portable_lib` first (the
   `.so` below has a runpath that assumes an activated venv; importing the
   pybinding module resolves its symbols in-process).
2. `torch.ops.load_library("<site-packages>/executorch/kernels/quantized/libquantized_ops_aot_lib.so")`.

This is purely for the host smoke test — the Android runner already links
`quantized_ops_lib` statically via `runner/CMakeLists.txt`.

### 2.8 Runner had a hard NEEDED on `libqnn_executorch_backend.so`

**Symptom.** On-device:

```
CANNOT LINK EXECUTABLE "./flux2_runner": cannot locate symbol
  "_ZN10executorch7runtime8internal17get_log_timestampEv"
  referenced by "/data/local/tmp/flux2/flux2_runner"
```

**Cause.** The runner's CMakeLists unconditionally linked
`qnn_executorch_backend` whenever the executorch install exported that
target. That target is a shared library, so the final binary gained a
dynamic dependency on `libqnn_executorch_backend.so`. The symbol
`get_log_timestamp` lives in that .so. Without pushing the .so and its
QNN HTP dependency chain, the dynamic linker fails.

But our `.pte` files don't use QNN at all — they're pure XNNPACK.

**Fix.** In `runner/CMakeLists.txt`:

```cmake
option(ENABLE_QNN "Link QNN backend (requires QNN HTP runtime on device)" OFF)
if(ENABLE_QNN AND TARGET qnn_executorch_backend)
  list(APPEND link_libraries qnn_executorch_backend)
  target_link_options_shared_lib(qnn_executorch_backend)
endif()
```

Rebuild. The resulting binary's NEEDED list is just
`[liblog.so, libm.so, libdl.so, libc.so]` — all Android system libraries.
No `.so` needs to ship alongside.

### 2.9 `mlock` failing on-device with ENOMEM

**Symptom.** Runner starts loading `text_encoder.pte`, then:

```
E executorch:mmap_data_loader.cpp:239] File text_encoder.pte (off=0x0):
  mlock(0x6fe8a72000, 341104) failed: Out of memory (12)
ERROR: text_encoder forward failed
```

**Cause.** Not real OOM. Android's default `RLIMIT_MEMLOCK` is ≈64 KB; the
`mmap_data_loader` tries to mlock a region an order of magnitude larger.
`mlock` returns `ENOMEM` meaning "rlimit exceeded", not "kernel OOM".

**Fix.** In `runner/flux2_main.cpp`, pass `Module::LoadMode::MmapUseMlockIgnoreErrors`
into all three `Module` constructors. The mmap still happens (so pages are
demand-paged from the `.pte` file as needed), but mlock failures are logged
and ignored. First-access is slightly slower (cold page fault) but the model
actually runs.

---

## Putting it together — what's new vs April

Code-side changes (small and localized):

- `export_flux2_klein_xnnpack.py`
  - `_ensure_flatc_on_path()` at module import.
  - `Qwen3TextEncoderWrapper.forward` precomputes a 4-D additive mask and
    passes it as `attention_mask={"full_attention": ...}` (dict form).
  - `export_component_to_xnnpack`: PT2E path for `dynamic_w8a8=True`,
    `_initialise_weight_observers`, `DuplicateDynamicQuantChainPass`,
    `sdpa_kernel(MATH)` + `no_grad` wrapping both exports,
    `strict=False` fallback.
  - `main()`: transformer and VAE both call `export_component_to_xnnpack(...,
    dynamic_w8a8=tf_use_w8a8, use_dynamic_quant_partitioner=tf_use_w8a8)`
    instead of `apply_w8a8_quantization(...)` followed by fp32 export.
  - `apply_w8a8_quantization` left in place as dead code but unused.

- `test_pte_host.py`
  - `_load_quantized_ops_lib()` preloads `_portable_lib` pybinding and
    `torch.ops.load_library("libquantized_ops_aot_lib.so")`.

- `runner/CMakeLists.txt`
  - `ENABLE_QNN` option, default OFF.

- `runner/flux2_main.cpp`
  - All three `Module` constructors now take
    `Module::LoadMode::MmapUseMlockIgnoreErrors`.

That's it. No new files, no external dependencies, no backend switch, no
custom ops. The whole deployment works end-to-end on CPU with a prompt-to-PPM
latency of ~2–5 minutes on the S26 Ultra (first run; subsequent runs closer
to the low end since file pages stay cached).
