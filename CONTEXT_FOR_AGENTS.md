# Context for Future Agents — FLUX.2-klein-4B on Qualcomm HTP

This file is the load-bearing context for anyone (human or LLM) picking up the
HTP NPU export work. It captures *what's true*, *why*, and *what to try next*.
Read this before reading individual scripts.

## Goal

Run FLUX.2-klein-4B (4B-param rectified-flow image diffusion) on a Samsung
Galaxy S26 Ultra (Snapdragon 8 Elite Gen 5, **SM8850, Hexagon V81 NPU**) via
ExecuTorch + QNN HTP backend. Currently the **CPU XNNPACK** path works
(`working_changes.md`); we're adding the **HTP** path for the transformer.

## Current state (2026-05-02)

### What works
- **CPU/XNNPACK path** for all three components (text encoder, transformer, VAE).
  Documented in `working_changes.md`. Ships and runs on phone.
- **HTP w8a8 transformer**: compiles, but with a bad quantization policy.
- **Linear-only-discard a8w8 host PT2E**: `+3.6 dB` SNR / `cos 0.82` —
  best result so far. **Not yet compiled to `.pte`.**

### Open
- Compile linear-only-discard config to `.pte` and verify it lowers cleanly.
- Promote `context_embedder` (and possibly 25 output projections) to a16w8 for
  more SNR.
- Reach torchao reference (+9.9 dB) or close enough for visually clean images.

## The two key bugs (and their fixes)

### Bug A: QnnQuantizer over-annotates → quant noise

**Symptom.** April's original a8w8 export compiled fine but gave **−2.6 dB SNR
/ cos 0.18** vs fp32 reference. Output uncorrelated with truth.

**Root cause.** `executorch.backends.qualcomm.quantizer.quantizer.QnnQuantizer`
inserts fake-quant nodes around **every op it knows about** — `mul`, `add`,
`layer_norm`, `softmax`, `bmm`, `matmul`, `cat`, `silu`, etc. (~163 op types).
Standard torchao recipe (`Int8DynamicActivationInt8WeightConfig`) only swaps
`nn.Linear` (~109 modules in FLUX). Annotating non-Linear ops compounds 256-
level rounding noise across hundreds of boundaries — particularly bad on:
- `softmax` outputs (probabilities < 1/256 round to 0)
- `layer_norm` outputs (precision lost immediately after normalization)
- attention `bmm`/`matmul` (double round-trip in QK^T → softmax → @V)
- residual `add`s with mismatched scales

**Fix.** `quantizer.add_discard_ops([...])` for the 162 non-Linear quant_ops.
Only `aten.linear.default`, `aten.conv2d.default`, `aten.conv1d.default` keep
annotation. Discarded ops still **run on HTP** — they just run in fp at the
boundary (no fake-quant inserted), then the next Linear's input observer takes
over. Reference: `diag_linear_only_quant.py`.

**Result.** −2.6 dB → +3.6 dB / cos 0.82, intermediate config (discard ~15
obvious ones): +1.5 dB / cos 0.76.

**Memory implication.** Discarding non-Linear ops saves **0 bytes on disk** —
they have no/tiny weights. The 4 GB Linear weights still quantize to int8.
Activations briefly run fp16 in VTCM for those ops; tiles are small enough.

### Bug B: VTCM tiling failure on rotary mul (only matters at int16)

**Symptom.** When promoting to a16w8 the QNN compiler errors at the rotary
`x * cos` mul: tile won't fit in 8 MB VTCM.

**Root cause.** Hexagon V81 has 8 MB VTCM (per-op tile scratchpad). HTP's
element-wise tiler uses `CHANNEL_SPLIT_SIZE = 256` (from
`qairt/.../optimize_defs.h`) and seems to commit to slicing one axis (innermost
or channel-equivalent). For rotary's
`(1, 1536, 24, 128) * (1, 1536, 1, 128)` — broadcast on dim=2 (heads):
- a8: ~4.7 MB tile, fits
- a16: ~9.4 MB tile, **doesn't fit**

The tiler doesn't fall back to slicing dim=1 (sequence) when its first choice
doesn't fit. Inference, not confirmed by source.

**Fix.** Manually pre-split the head dim before the multiply, run N smaller
muls, `cat`. Mathematically identical (verified bit-exact at FX-graph level —
`max abs diff = 0.0`). Set `FLUX_ROTARY_HEAD_SPLIT=2` to enable.
Implementation in `export_flux2_klein_qnn.py`.

**Important nuance.** Rotary tiling **was not blocking pure a8w8** — that
already compiled. The chunking is preventative for selective int16 promotion.

**Recipe template.** Same trick (manual axis split + cat) likely applies to
other broadcast-on-middle-dim element-wise ops if they fail at int16. Suspects:
modulation muls (adaLN-Zero), any attention bmm with broadcast on heads.

## Per-Linear SNR ranking (from `diag_per_linear_snr.py`)

Local int8 simulation (per-tensor input × per-channel weight) on each of the
109 Linears. **Two clear patterns:**

1. **`context_embedder` is the outlier**: SNR 7.15 dB / err_norm 4140 vs
   next-worst 12.8 dB / 1635. Single biggest contributor to remaining noise.
   Likely cause: T5/Qwen text-encoder output has wide dynamic range.

2. **Output projections cluster at 10–25 dB**:
   - `single_transformer_blocks.*.attn.to_out` (20 modules, 10–24 dB)
   - `transformer_blocks.*.ff_context.linear_out` (5 modules, 10–15 dB)
   - `transformer_blocks.*.ff.linear_out` (5 modules, 20–26 dB)

   Their **inputs** are post-softmax / post-GELU activations with outliers —
   classic SmoothQuant target.

3. Everything else (Q/K/V, modulation, embedders, fused QKV+MLP) is fine
   (30–50 dB).

## What was tried and what didn't help

| Experiment | Result |
|---|---|
| Observer sweep (Histogram / MinMax / MovingAvg / QNN default) | All converged to similar SNR; observer choice not the bottleneck |
| Per-block promotion D00–D04 to 16a8w | **−2.92 dB** (worse) — boundary requant overhead defeats local gain |
| Host-side SmoothQuant / Hadamard / per-token L2 (`diag_outlier_methods*`) | No gain at default annotation; not retested after `add_discard_ops` |
| Auto-skip via shape heuristic (`FLUX_ENABLE_AUTO_SKIP`) | Too aggressive at int16 (skipped tilable matmuls); gated behind env var |
| `aggressive_skip` of all unsupported ops | Helps — but `add_discard_ops` is cleaner |
| Promoting `encoder_hidden_states` placeholder via `set_submodule_qconfig_list` | Predicate matched 0 nodes — placeholders don't have nn_module_stack |
| Naive w8a8 PyTorch on 4090 (subagent-built) | Reference: 9.9 dB static / 17.4 dB dynamic — confirms torchao recipe is achievable, not HTP-specific |

## Things to try next (in suggested order)

1. **Compile linear-only-discard a8w8 to `.pte`** — does the host config
   actually lower cleanly to QNN HTP? Open question. Key validation step.

2. **Promote `context_embedder` to a16w8**, leave everything else at a8w8.
   Cheapest test of the diagnosis. Expected: meaningful host SNR jump if
   diagnosis is correct.

3. **Promote the 25 output projections** (`*.attn.to_out`,
   `*.ff_context.linear_out`) — full a16w8 on the noisy set. Each
   promotion may trigger downstream broadcast-mul tiling failures (modulation,
   attn bmm); apply rotary-style chunking patches as needed.

4. **SmoothQuant on those 25 Linears as alternative to int16 promotion** —
   no memory cost, may close some of the gap. Skipped in this session per user
   direction; reconsider if int16 promotion gets blocked.

5. **Investigate residual ~6 dB gap** vs torchao reference. Likely candidates:
   QNN HTP power-of-2 scale rounding, symmetric-only constraint, or
   per-channel-vs-per-tensor differences in act observer placement.

## Things known not to work (or "stop signs")

- **Dynamic per-token activation quant**: `QnnQuantizer` doesn't expose this
  flag in our build. Don't waste time — it's the single biggest lever in
  torchao's reference (+7.5 dB delta), but the API isn't there.
- **Wholesale a16w8** on everything: 2× activation traffic, ~2× HVX cycles,
  blows multiple VTCM budgets. Not worth pursuing.
- **Auto-skip via heuristic**: too brittle. Use explicit module-stack
  predicates or op-list discards.

## Critical files and entry points

| File | What it does |
|---|---|
| `export_flux2_klein_qnn.py` | Main wrapper + rotary chunking patch. `FLUX_ROTARY_HEAD_SPLIT` env var. |
| `export_flux2_klein_qnn_v12.py` | Auto-skip path (gated, opt-in via `FLUX_ENABLE_AUTO_SKIP=1`) |
| `diag_linear_only_quant.py` | The big-win experiment. Discards 162 non-Linear ops. **+3.6 dB**. |
| `diag_per_linear_snr.py` | Per-Linear local SNR ranking. Output: `per_linear_snr.json`. |
| `diag_block_promote.py` | Module-stack-predicate based promotion (template for `context_embedder` work). |
| `diag_block_snr.py` | Per-block round-trip int8 simulation. Shows D03 cliff. |
| `diag_check_rotary_export.py` | Verifies rotary chunking produces different shape distribution. |
| `working_changes.md` | CPU/XNNPACK shipping path (already works). Don't touch unless requested. |
| `april_export_original.py` | April Hu's original baseline. Reference. |
| `april_patches.md` | Diff log of what we changed vs April's. |

## Environment notes

- **Python**: `.venv/bin/python` (not on PATH — must `source .venv/bin/activate`
  or invoke directly).
- **QNN SDK**: `LD_LIBRARY_PATH` must include both `qairt/.../lib/x86_64-linux-clang`
  AND `.local-libs-14/usr/lib/x86_64-linux-gnu` (libc++).
- **`configure_local_tooling()`** in `export_flux2_klein_qnn.py` sets these.
  Always import + call it before `QnnQuantizer`.
- **Calibration data**: `calibration_data/calibration_transformer.pt` (gitignored,
  316 MB). Tuple-of-inputs format, 5 samples typical.
- **SoC**: `QcomChipset.SM8850` (Snapdragon 8 Elite Gen 5, V81).
  *Don't* default to V75/V79 — different VTCM, different op coverage.

## Glossary

- **VTCM**: Vector Tightly-Coupled Memory. 8 MB per-op scratchpad on V81.
  Where every HTP op streams its tile. **Working memory, not DRAM.**
- **PT2E**: PyTorch 2 Export quantization flow:
  `prepare_pt2e → calibrate → convert_pt2e`. Inserts fake-quant during prepare;
  converts to real Q/DQ during convert.
- **Fake quantization**: simulation node `dequant(quant(x))` that round-trips
  through int8 representation but stores result as fp. After `convert_pt2e`
  becomes real Q/DQ (still computed in fp on host CPU; runs on int8 silicon
  on device). Numerically equivalent.
- **Annotation**: in PT2E, "annotating" an op means inserting fake-quant nodes
  at its inputs/outputs.
- **Discard op**: removes annotation for that op type — op runs in fp at the
  boundary, no quant noise added there.
- **a8w8 / w8a8**: 8-bit activations, 8-bit weights. Same thing different
  ordering.
- **a16w8**: 16-bit activations, 8-bit weights. Used for selective promotion.
