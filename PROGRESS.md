# DSP Deployment Progress (Qualcomm / QAIRT)

Date: 2026-04-12 (updated)

This file summarizes current progress, what we tried, and the issues blocking QAIRT DSP deployment for the FLUX.2-klein-4B diffusion transformer in W8A8.

## Goal

Deploy the diffusion model on Qualcomm phone DSP in 8-bit quantized form (weights + activations) using QAIRT/QNN.

## Architecture Summary

Two export paths exist in this repo:

| Path | Script | Output | Status |
|------|--------|--------|--------|
| **QAIRT direct** (ONNX → DLC) | `export_flux2_klein_qairt.py` | `.dlc` files | Active; VAE working, transformer blocked |
| **ExecuTorch QNN** (torch.export → .pte) | `export_flux2_klein_qnn.py` | `.pte` files | Written but untested (needs ExecuTorch + QNN SDK build) |
| **ExecuTorch XNNPACK** (CPU) | `export_flux2_klein_xnnpack.py` | `.pte` files | Working (reference baseline) |

The QAIRT direct path (`export_flux2_klein_qairt.py`) is the one with active iteration (22 transformer smoke tests so far). The ExecuTorch QNN path is a cleaner pipeline but depends on ExecuTorch being built with QNN support, which hasn't been validated yet.

## Current Status

| Component | ONNX Export | QAIRT Conversion | DLC File |
|-----------|-------------|------------------|----------|
| VAE decoder | Working | Working | `vae_decoder.dlc` (in `tmp_qairt_smoke4/`) |
| Text encoder | Working (TorchScript → ONNX bundle) | Blocked | No `.dlc` produced |
| Transformer | Working (dynamo ONNX export) | Blocked | No `.dlc` produced (22 smoke tests) |

## What We Tried

### 1) Replace SDPA with Manual Attention
Reason: QAIRT/QNN has limited support for fused SDPA ops and related patterns.

Actions:
- Implemented manual attention with explicit `matmul -> softmax -> matmul`.
- Replaced diffusers attention processors for both self-attn and cross-attn.

Outcome:
- Export works; QAIRT conversion still fails later due to shape/canonicalization issues.

### 2) Remove Chunk/Split/Repeat Patterns Not Supported by QAIRT
Reason: QAIRT converter is sensitive to `SplitToSequence`, `Chunk`, and some broadcast/shape patterns.

Actions:
- Patched diffusers components to avoid `chunk()`:
  - `AdaLayerNormZero`, `AdaLayerNormZeroSingle`, `AdaLayerNormContinuous`
  - `Flux2SwiGLU`
  - `Flux2Modulation`
  - `transformer_flux2` fused projection split functions
- Replaced chunk/split with explicit slice-based ops (pure slicing on last dim).
- Disabled `split_with_sizes` where it produced `SplitToSequence`.

Outcome:
- SplitToSequence errors reduced, but QAIRT still fails with other canonicalization errors.

### 3) Rotary Position Embedding Changes
Reason: `repeat_interleave` and rotary helpers caused unsupported ops.

Actions:
- Replaced `get_1d_rotary_pos_embed` to use `stack+reshape` instead of `repeat_interleave`.
- Implemented `_export_apply_rotary_emb` using reshape + stack.
- Added option to **disable rotary entirely** for QAIRT conversion to test minimal graph.

Outcome:
- Disabling rotary reduced some converter errors, but conversion still fails.
- **Note: `DISABLE_ROTARY_FOR_QAIRT = True` is still set**, meaning even if conversion succeeds the model will produce incorrect results. This must be re-enabled before shipping.

### 4) CastLike Removal
Reason: QAIRT does not support ONNX `CastLike`.

Actions:
- Post-process ONNX to replace `CastLike` with `Cast`.
- Save with external data to avoid size limits.

Outcome:
- CastLike errors resolved, but converter still fails later.

### 5) QAIRT Python API Stability Patch
Reason: Repeated segfaults / invalid IR errors indicated tensor lifetime issues.

Actions:
- Patched QAIRT Python bindings to hold references to tensor shapes/attributes (`patch_qairt_reshape.py`).
- Disabled Python GC during `qairt.convert()` calls.

Outcome:
- Improved stability, but conversion still fails with shape/canonicalization errors.

## Key Issues Blocking Transformer Conversion

Converter errors observed (examples):
- `canonicalizeOp: invalid stride 1 for begin 0 and end 0 at axis 0`
- `SplitToSequence` unsupported (partially mitigated but may still appear indirectly)
- Broadcast shape mismatch (e.g., `1536 vs 1535` or `127`)
- Invalid MatMul shapes (from inferred shape propagation, not runtime)

Root cause hypothesis: **unresolved Reshape shapes**. The transformer ONNX path was NOT running shape resolution on Reshape ops (replacing `-1`/`0` placeholders with concrete dims). The VAE path does this via `simplify_onnx` and works. This is now fixed (see below).

## Fixes Applied (2026-04-12)

1. **Bug fix: `_export_get_fused_projections` returned `(None,)` instead of `None`** for encoder_query/key/value when no encoder_hidden_states. `(None,)` is a truthy tuple, which could cause downstream issues in attention processing.

2. **Added Reshape shape resolution to transformer path.** The VAE used `simplify_onnx` (which includes `_resolve_reshape_shapes`), but the transformer path only did shape inference + CastLike rewrite. Now `resolve_onnx_reshapes()` runs after CastLike rewrite, replacing `-1`/`0` in Reshape shape constants with concrete values. This is likely the most impactful fix for the QAIRT canonicalization errors.

3. **Added CLI fallback conversion path** (`--use_cli` flag). Uses `qairt-converter` + `qairt-quantizer` CLI tools as an alternative to the Python API (`qairt.convert`). The CLI tools sometimes handle edge cases differently.

4. **Refactored ONNX save logic** into shared `_save_onnx_model` helper to handle external data fallback consistently.

5. **`simplify_onnx` now supports `has_external_data` parameter** for models with external weight files (like the transformer).

## Files Modified

- `export_flux2_klein_qairt.py`
  - Manual attention processors.
  - Patch hooks for diffusers to remove chunk/split.
  - Rotary and timestep embedding modifications.
  - CastLike rewrite + ONNX shape inference + Reshape resolution.
  - QAIRT conversion flow (Python API + CLI fallback).
- `patch_qairt_reshape.py`
  - Keeps tensor/shape/attributes references alive in QAIRT Python bindings.

## Next Steps (Prioritized)

### Immediate (try now)
1. **Re-run transformer export** with the Reshape resolution fix. This is the highest-impact change.
   ```bash
   ./run_qairt_export.sh --component transformer --output_dir ./tmp_qairt_transformer_smoke23
   ```
2. **Try CLI-based conversion** if Python API still fails:
   ```bash
   ./run_qairt_export.sh --component transformer --use_cli --output_dir ./tmp_qairt_transformer_smoke23_cli
   ```

### If transformer conversion still fails
3. **Try `onnxsim` on the transformer** (full simplification, not just Reshape resolution). This is heavier but may fold more problematic patterns:
   ```python
   from onnxsim import simplify
   # Requires loading the full model with external data — memory-intensive
   ```
4. **Try opset 14 instead of 17** for the transformer ONNX export. Lower opsets produce simpler graphs (fewer fused ops).
5. **Try legacy ONNX export** (`dynamo=False`) instead of dynamo export. The dynamo exporter produces different graph structure that may be harder for QAIRT.
6. **Re-enable rotary embeddings** once base conversion works — `DISABLE_ROTARY_FOR_QAIRT = True` must be flipped to `False` before the model is usable.

### If QAIRT path proves unworkable
7. **Pivot to ExecuTorch QNN path** (`export_flux2_klein_qnn.py`). This requires:
   - Building ExecuTorch with QNN support (see `CONTEXT_4090.md`)
   - QNN SDK 2.28+ installed
   - But avoids the ONNX intermediate step entirely (torch.export → QnnPartitioner → .pte)

## Smoke Test History

All transformer ONNX exports are in `tmp_qairt_transformer_smokeXX/`, with increasing patch history.
- smoke1–22: Various patch iterations, all ONNX exports succeed, QAIRT conversion fails
- **smoke23**: Next run, with Reshape resolution fix

VAE smoke tests: `tmp_qairt_smoke1–4/` (smoke4 has working `vae_decoder.dlc`)
Text encoder smoke tests: `tmp_qairt_text_smoke1–6/` (all failed at QAIRT conversion)

## Decisions Needed

1. **QAIRT vs ExecuTorch QNN:** If the Reshape fix doesn't unblock the transformer, should we pivot to the ExecuTorch QNN path? The ExecuTorch path is cleaner (no ONNX intermediate) but requires building ExecuTorch with QNN support.

2. **QAIRT SDK version:** Currently using 2.45.0. Is a newer version available from Qualcomm? SDK bugs are a real possibility given the C++ memory corruption issue we already found.

3. **Rotary embeddings:** Currently disabled for QAIRT. Once conversion works, need to re-enable and verify the export-friendly implementation produces correct results vs. the original.

4. **Target resolution:** Currently exporting at 512x512. Going to 768+ will significantly increase transformer sequence length. Should we validate at 512 first before attempting higher resolutions?
