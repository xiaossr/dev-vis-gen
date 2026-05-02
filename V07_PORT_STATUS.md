# ExecuTorch v0.7.0 QNN Port — Status & Blocker

## Where we are

The v0.7.0 port **passes every v0.6.0 fatal**: RouterX86 multi-softmax, the
on-device memmove segfault, dtype mismatches. The pipeline reaches QNN's HTP
AOT compile step. **But the compile is pathologically slow** — 36+ minutes at
opt level 1 on the simplest possible test (0 double + 1 single transformer
block) without completing partition 1 of 5.

## What was done

- `.venv-et07/` fresh venv with executorch 0.7.0, torchao 0.12.0, torch 2.8.0.
- `executorch/` tree checked out at `v0.7.0` tag.
- Ported April's export script to `export_flux2_klein_qnn_v07.py` using the
  v0.7.0 `to_edge_transform_and_lower_to_qnn` helper (the path April's code
  assumed; our v0.6.0 didn't have it).
- Applied V81/SM8850 to `qc_schema.py`, `qc_compiler_spec.fbs`,
  `get_soc_to_arch_map`, `get_soc_to_chipset_map`.
- Patched `lift_constant_scalar_operands.py` for non-Node args.
- Patched `qnn_partitioner.py` to return False on ops with no visitor.
- Patched `insert_io_qdq.py`'s `q_dq_map` to include dequant variants.
- Patched `node_visitor.get_data_type` to treat fp64 as fp32.
- Added empty `__init__.py` where v0.7.0 omitted them so `pkg_resources` can
  resolve .fbs schema files.
- Ported `_remove_int_quantize_nodes` from v0.6.0 to strip Q/DQ on arange
  outputs.
- Rewrote LN decomposition's eps scalar to use a float32 `scalar_tensor` to
  avoid fp64 promotion.
- **Binary-patched `PyQnnManagerAdaptor.so` at offset 0x8396c** to change the
  host AOT optimization level from 3.0 to 1.0 (matching the aarch64 device
  setting). Confirmed patch took effect — log now says `Running level=1
  optimization.` instead of `level=3`.

## The remaining blocker

Even with opt level 1, the QNN HTP compile grinds:
- 100% single-threaded CPU.
- 65 GB resident memory.
- Log freezes at `Processing Method(0): (1/5)` for 36+ minutes.
- No crash, no forward-progress indicator. Not clear if it would eventually
  complete.

This doesn't look like a solvable-in-Python bug. It's inside Qualcomm's HTP
compiler processing our specific graph. Possible underlying causes:
- Our LN decomposition produces an op pattern the HTP compiler's optimization
  passes loop on.
- Our quant annotation layout creates a pathological partition structure.
- A specific op combination in Flux attention triggers exhaustive enumeration
  in the HTP scheduler.

None are debuggable without Qualcomm's source or support. April's setup
presumably avoided this but we don't have her exact dependency tree.

## Files to keep

- `export_flux2_klein_qnn_v07.py` — v0.7.0 port.
- `diagnose_mini_v07.py` — mini reproducer.
- `V07_PORT_STATUS.md` — this file.
- `.venv-et07/` — if deleted, 15 min to reinstall.
- `executorch/` at v0.7.0 with patches — `git stash` would throw away our
  patches; either commit them or keep the working tree.

## Files to kill

If we're done with the NPU path:
- Above plus `V07_PORT_STATUS.md`.
- The v0.6.0 port (`export_flux2_klein_qnn.py`, `diagnose_mini_transformer.py`).
- `mini_v07*.pte`, `mini_*.pte`, `mini_1single.pte`, `mini_fp16.pte`,
  `mini_blk2.pte`, etc.

## Suspected noise sources (orthogonal to compile)

If anyone ever gets a `transformer.pte` produced on the NPU, these are April's
documented noise sources that still need addressing:

1. Synthetic calibration (`torch.randn_like`) — picks scales from unrealistic
   distributions. Fix: `--calibration_dir` with real prompts through the
   PyTorch pipeline.
2. No explicit activation observer — defaults saturate. Fix: percentile
   observer.
3. `set_per_channel_linear_quant(True)` not called. Fix: call it.
4. Attention softmax quantized by default. Fix: `add_discard_ops([softmax])`.

## Honest recommendation

Ship `exported_flux2_klein_xnnpack/` (all three components on CPU). That
bundle is known-working, cleanly quantized (measured 0.987 correlation vs
fp32), and produces real images. Document the NPU attempt in this file so the
next person can pick up where this stops.
