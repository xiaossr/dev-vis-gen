# Iteration Log

Tracking each attempt to get the QAIRT transformer conversion working.

## Iteration 23 — Reshape resolution fix (2026-04-12)

**Starting point:** 22 prior iterations all fail at QAIRT conversion with shape/canonicalization errors. ONNX export succeeds. Today's fix added `resolve_onnx_reshapes()` to replace -1/0 shape placeholders in Reshape ops with concrete values.

**Action:** Run transformer export with the new fix.

**Result:** Reshape fix worked (1464 shapes resolved). Hit infra issues:
- `infer_shapes()` fails on >2GB model → fixed with `infer_shapes_path`
- ONNX rejects symlinks/hardlinks for external data → fixed by reusing same-dir data file

Converter now gets further but hits: `canonicalizeOp: invalid stride 1 for begin 0 and end 0 at axis 0` at `node_Slice_597`. This is a zero-length Slice (begin=0, end=0), which QAIRT doesn't handle.

**Status:** Partial progress. Reshape fix unblocked loading. New blocker: zero-length Slice ops.

---

## Iteration 24 — Fix zero-length Slice ops (2026-04-12)

**Starting point:** QAIRT converter fails at `node_Slice_597` — a Slice with begin=0, end=0 (zero-length output). QAIRT's canonicalizer rejects this.

**Action:** Found all 310 Slice ops have non-constant params (dynamo exporter creates Constant->Cast->Reshape chains instead of inlining values). Wrote `_fold_constant_chains()` pass to evaluate these chains and replace with static initializers. Folded 930 params.

**Result:** Converter got further — past canonicalization. New error: `Duplicate buffer name, t_10 already exists`. Dynamo's `t_N` naming collides with QAIRT internal names.

**Fix:** Added `_rename_conflicting_tensors()` to prefix all `t_N` names with `flux_`.

**Result:** 
```
INFO_CONVERSION_SUCCESS: Conversion completed successfully
Total MACs: 6949404686
Total Params Count: 3875536896
```

**CONVERSION WORKS!** But quantization (calibration) fails: QNN CPU backend sees only 3 inputs instead of 5. `img_ids` and `txt_ids` were absorbed as constants during optimization.

**Status:** Conversion unblocked. Quantization calibration input mismatch.

---

## Iteration 25 — Fix quantization calibration inputs (2026-04-12)

**Starting point:** Converter succeeds. Quantizer's calibration inference fails because `img_ids` and `txt_ids` were optimized away (treated as constants by QAIRT since they were static in the ONNX). Need to ensure all 5 inputs remain as graph inputs.

**Action:** Investigate how to prevent QAIRT from absorbing inputs as constants.

**Status:** QAIRT path abandoned in favor of ExecuTorch QNN path.

---

## ExecuTorch QNN Path (2026-04-14 — 2026-04-15)

**Decision:** Pivoted from QAIRT direct (ONNX → DLC) to ExecuTorch QNN (torch.export → .pte). The ExecuTorch path avoids ONNX entirely and directly uses Qualcomm's QNN integration.

### VAE Decoder — Exported successfully
- Clean export, 55.8 MB .pte file
- Fixed 3 bugs in ExecuTorch QNN backend (LiftConstantScalarOperands, annotate_layer_norm, op_layer_norm)

### Transformer — Exported successfully
- 3697.1 MB .pte file
- Fixed 5 additional bugs (int64 quantize nodes, eval on exported model, float64 key error, missing op visitors, x86 HTP compilation failure)
- Key fix: `online_prepare=True` to defer HTP compilation to device

### Text Encoder (Qwen3) — Exported successfully (2026-04-15)
- 2970.9 MB .pte file
- Fixed 5 more bugs:
  - Error 9: Qwen3's `@capture_outputs` uses threading.Lock → custom wrapper bypassing decorator
  - Error 10: Module rename `i64_to_i32` → `tensor_i64_to_i32`
  - Error 11: `get_decomp_table()` needs `None` arg
  - Error 12: ExportPass dtype mismatch → graceful per-pass error catching (try/except instead of no-op)
  - Error 13: Missing `val` metadata on delegate nodes → multi-fallback in backend_api.py + EdgeProgramManager instead of second to_edge()

**ALL 3 COMPONENTS EXPORTED. Total: ~6.7 GB of .pte files.**

See `EXECUTORCH_ERRORS.md` for detailed error documentation.
