# Handoff — FLUX.2-klein on Snapdragon 8 Elite Gen 5 (Hexagon V81 HTP)

## Goal
Run FLUX.2-klein-4B (4B-param image diffusion) on a Samsung Galaxy S26 Ultra
(SoC: SM8850, NPU: Hexagon V81) via ExecuTorch QNN HTP backend. April Hu is
the device-side tester; we hand her `.pte` files, she runs them through a
prebuilt runner at `/data/local/tmp/flux2/htp/flux2_qnn_main`.

## Current state (2026-05-14)

Three transformer `.pte` files have been compiled. Host PT2E numbers are
measured on a single calibration sample vs fp32 same-input reference:

| File (dir)                                | Policy            | Size    | Host SNR / cos       |
|-------------------------------------------|-------------------|---------|----------------------|
| `exported_flux2_klein_qnn_lin_only/`      | w8a8 lin-only-discard | 3.97 GB | +3.6 dB / 0.82  |
| `exported_flux2_klein_qnn_a16w8/`         | w8a16 lin-only-discard | 3.94 GB | +23.1 dB / 0.998 |
| `exported_flux2_klein_qnn_a16w4/`         | w4a16 lin-only-discard | 2.0 GB  | +7.5 dB / 0.91 (compile threw dtype warnings; device-untested) |

The w8a16 number (+23.1 dB / 0.998) is the headline — beats torchao static
(+9.9) and dynamic (+17.4) references, with no manual broadcast-mul chunking
beyond the existing rotary head-split.

text_encoder and decoder were already host-verified XNNPACK w8a8 (see
`MEMORY.md` → `project_flux2_dsp.md`).

## Active blocker — April's device test failed

April ran one of the `.pte`s on device. Log is at
`device_logs/log.rtf` (RTF) and stripped to `device_logs/log.txt`.

**What happened:**
1. text_encoder.pte ran fine, produced `prompt_embeds: 512 x 7680 (2.9 s)`.
2. Noise gen ran fine.
3. transformer.pte step crashed at *graph-context init*, before any forward.

**Single error pattern, repeated ~80× (once per transformer partition):**
```
[ERROR] QnnDsp Error from rpc transport when doing buffer mapping, map result: 8003
[ERROR] QnnDsp Failed to map input/output buffer on NSP
[ERROR] QnnDsp ERROR happen when trying to allocate buffers for graph forward
[ERROR] QnnDsp Failed to initialize graph memory
[ERROR] QnnDsp Failed to initialize graph with id N context M ... err 1002
[ERROR] QnnDsp Context create from binary failed ... err 1002
[ERROR] QnnDsp Context N failed on pd 0
```

**Diagnosis: rpcmem exhaustion at graph init — NOT VTCM, NOT total .pte size.**

- `8003` = `AEE_ERPC_NORESOURCE` from FastRPC. The host-side
  rpcmem/ION pool used to map I/O buffers into DSP-visible address space
  ran out.
- `1002` = `QNN_CONTEXT_ERROR_INIT_FAILED` (downstream).
- This is at *context create*, not during execution — so VTCM (which is
  only touched inside op execution) is *not* the constraint.
- Cause: ~80 partitions × per-partition activation I/O tensors, doubled by
  int16 activations vs int8. Cumulative rpcmem footprint > pool budget.

The `magic number 0x5678abcd ... but get 0x2000000` line is `[INFO]`, not
an error — benign QNN v1.2.0 protocol-version notice.

**Unknown: which `.pte` is actually on her device.** Ask her to run:
```
adb shell ls -lh /data/local/tmp/flux2/htp/transformer.pte
```
- 3.97 GB → w8a8
- 3.94 GB → w8a16
- 2.0 GB → w4a16

## Recommended next steps (ordered by cost)

1. **Confirm which .pte April ran**, then have her try `transformer_w8a8.pte`
   (half-size activation buffers from int8 acts). If it still fails →
   it's per-partition fixed overhead, not bytes. If it works → confirmed
   activation-bytes problem, then we have to choose: ship w8a8 quality, or
   re-export with fewer/larger partitions.
2. **Try lower `--htp_performance_mode`** (3 → 1 or 2). Aggressive perf
   modes can pre-allocate more buffer headroom.
3. **Re-export with fewer partitions** — real code change to the partition
   policy in the export script. Multi-hour compile.
4. **Re-export at lower seq_len** (e.g. 256 instead of 512) — halves activation
   buffer sizes, but needs the runner's prompt path adjusted too.

## Repo / env essentials

- Working dir: `/data/home/thanush/dev-vis-gen`
- Use venv `.venv-et12` (ExecuTorch v1.2.0). **Do not** use `.venv`
  (v0.6.0, incompatible with our local executorch tree).
- Local `executorch/` checkout carries patches the v1.2.0 site-packages
  version lacks (LayerNorm None-weight/bias fixes in
  `backends/qualcomm/builders/op_layer_norm.py`,
  `backends/qualcomm/quantizer/annotators/htp_rules.py`).
  **Keep `_REPO` on `sys.path`** in export scripts — removing it falls back
  to the unpatched venv copy and fails at partitioner with
  `AttributeError: 'NoneType' object has no attribute 'name'`.
- Required env var for transformer export: `FLUX_ROTARY_HEAD_SPLIT=2`
  (head-chunks rotary embedding so it fits 8 MB VTCM tile).
- Recommended cwd for export: `/tmp` (avoids FLATC path conflicts; see
  README for full invocation).
- torchao must be vendored at v0.17.0-rc1 (`executorch/third-party/ao` on
  sys.path before installed torchao). Installed torchao 0.10 lacks
  `quantization.pt2e`.

## Key scripts

- `export_flux2_klein_qnn_lin_only.py` — w8a8 linear-only-discard export
- `export_flux2_klein_qnn_a16w8.py` — w8a16 (same recipe, `use_16a8w`)
- `export_flux2_klein_qnn_a16w4.py` — w4a16 (`use_16a4w`)
- `diag_linear_only_a16w8.py` — host PT2E SNR test (the +23.1 dB result)
- `diag_check_promotion_actually_applied.py` — verifies
  `set_submodule_qconfig_list` actually changes observer bit-width
- `diag_promote_context_embedder.py` — host test of selective promotion
  (result: 0 dB gain; quant noise broadly distributed, not concentrated)
- `diag_per_linear_snr.py` — per-Linear local SNR ranking
  (`per_linear_snr.json`). Misleading: predicts neither end-to-end
  sensitivity nor which Linears to promote.

## Things tried that didn't help (so don't redo)

- **Selective a16w8 promotion** of "noisy" Linears (context_embedder,
  top-5 output projections, all 31 problem Linears): 0 dB end-to-end gain
  despite observers verifiably switching to int16. Quant noise is broadly
  distributed across all 109 Linears, not concentrated in the per-Linear-SNR
  outliers.
- Per-Linear local SNR ranking does *not* predict end-to-end sensitivity.

## Reference files

- `CONTEXT_FOR_AGENTS.md` — older context dump (bug history, glossary, etc.)
- `README.md` — QNN HTP section has current invocation, recipe, gotchas
- `MEMORY.md` (in `~/.claude/.../memory/`) — project memory

## Communication state

- Email draft to Muyang Li (mentor) and Song Han (PI) was prepared earlier
  summarizing the w8a16 breakthrough. Status unknown (not tracked here).
- April is awaiting a path forward on the rpcmem failure.
