# The v1.2.0 path that actually works

## What got us here
April's `export_flux2_klein_qnn.py` (commit `342a6cd`) uses ExecuTorch APIs that
didn't exist in v0.6.0 or v0.7.0 — specifically `QnnQuantizer(backend=, soc_model=)`
and `set_default_quant_config`. Those were added March 10, 2026 in commit
`7824373c7` ("Backend awareness quantizer"). April's April 15 commit targeted
that newer tree. We were fighting v0.6.0's completely different API.

Upgrading to v1.2.0 (released April 2026) gives us:
- April's exact quantizer constructor and methods.
- Native support for `aten.native_layer_norm.default` — no decomposition needed.
  5+ partitions collapse to 2 on a 3-block mini, which is why compile completes
  in minutes not hours.
- SM8850 / V81 enum entries already in the schema.

## Setup
- `.venv-et12/` — Python 3.10, executorch 1.2.0 pip wheel, torchao 0.12.0,
  torch 2.8.0, diffusers/transformers.
- `executorch/` tree — checked out at tag `v1.2.0` with patches (see below).
- Compiled QNN bindings (`PyQnnManagerAdaptor.cpython-310-x86_64-linux-gnu.so`,
  903 KB) copied from the pip wheel into
  `executorch/backends/qualcomm/python/`.
- fbs schemas copied from pip wheel into `executorch/exir/_serialize/` and
  `executorch/schema/` (pip wheel has them; source tree doesn't).

## Patches applied to the v1.2.0 tree
All in `executorch/backends/qualcomm/`:

| File | Fix | Why |
|------|-----|-----|
| `quantizer/rules.py::_mark_nodes_as_annotated` | `if node is None: continue` | FLUX's LayerNorm(elementwise_affine=False) yields `None` weight/bias; stock code AttributeErrors |
| `quantizer/annotators/htp_rules.py::LayerNorm.annotate` | Skip `weight_node`/`bias_node` annotation when they're `None`; don't add `None` to `nodes_to_mark_annotated` | Same root cause as above |
| `builders/op_layer_norm.py::define_node` | When `node.args[2]` (weight) or `node.args[3]` (bias) is None, synthesize `torch.ones`/`torch.zeros` of `normalized_shape` as static tensors via `define_tensor(wrapper_idx=1/2)` | QNN's LayerNorm op requires weight and bias; Python-side synthesis gives them the identity values that match the PyTorch semantics of missing weight/bias |
| `partition/qnn_partitioner.py::is_node_supported` | Early return `False` if `node.target.__name__ not in self.node_visitors` | `aten.reciprocal`, `aten.var` etc. have no QNN visitor; stock code `KeyError`s. Fall back to CPU portable kernel instead. |

## Environment variables required at export time
- `QNN_SDK_ROOT=$(pwd)/qairt/2.45.0.260326`
- `LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$(pwd)/.local-libs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`
  (libc++ for linking the QNN host .so)
- `FLATC_EXECUTABLE=$(pwd)/.venv/lib/python3.10/site-packages/executorch/data/bin/flatc`
  (v0.6.0's native flatc binary; v1.2.0's wrapper script tries to `from
  executorch.data.bin import flatc` which fails when site-packages is shadowed)

## Running an export
```bash
PY=$(pwd)/.venv-et12/bin/python
export QNN_SDK_ROOT=$(pwd)/qairt/2.45.0.260326
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$(pwd)/.local-libs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export FLATC_EXECUTABLE=$(pwd)/.venv/lib/python3.10/site-packages/executorch/data/bin/flatc
cd /tmp   # so our local tree doesn't shadow site-packages
$PY /tmp/diag_v12.py    # mini reproducer; ~8 min wall clock
```

The reproducer lives in `/tmp/diag_v12.py` and imports wrappers from the repo
while using the pip-installed v1.2.0 executorch. It patches in V81/SM8850 via
the v1.2.0 enum (already present) and builds a 3-block mini.

## What works
- `mini_v12.pte` — 721 MB, w8a8 AOT, SM8850 target. First w8a8 compile that has
  ever completed for us. 2 partitions total (vs 5+ on v0.7.0).

## What's next
1. Full transformer export (25 blocks): use a full-sized model instead of the
   mini. Expected ~20-40 min based on mini's ~8 min.
2. Push to device, test.
3. If noise: real calibration data via `--calibration_dir`, then 16a8w,
   then softmax discard.

## Files
- `/tmp/diag_v12.py` — mini reproducer (keep).
- `mini_v12.pte` — proof-of-life artefact.
- `V12_PATH.md` — this file.
- `V07_PORT_STATUS.md` — still accurate for the v0.7.0 port we abandoned.
