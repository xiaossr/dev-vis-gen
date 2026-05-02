# Deep audit of April's branch (`origin/aprilhuu/htp_staged` / commit `342a6cd`)

Her branch is 1 commit ahead of `main`, containing exactly these files:

| File | Type |
|------|------|
| `QNN_HTP_GUIDE.md` | new |
| `collect_calibration_data.py` | new |
| `executorch/examples/models/flux2/CMakeLists.txt` | modified |
| `executorch/examples/models/flux2/flux2_qnn_main.cpp` | new |
| `export_flux2_klein_qnn.py` | new |
| `push_htp.sh` | new |

The `executorch/` directory is gitignored, so nothing about her ExecuTorch
tree is in her commit. Her changes *under* `executorch/examples/models/flux2/`
are not to the backend — they're to the example runner ExecuTorch ships.

---

## 1. Patches she documented but didn't commit

Her `QNN_HTP_GUIDE.md` Section 2 explicitly lists the edits she applied to
`/root/executorch/backends/qualcomm/` locally. These are the patches we spent
hours rediscovering:

| File | Fix | Status on our v1.2.0 tree |
|------|-----|---------------------------|
| `_passes/lift_constant_scalar_operands.py` | `aten.pow.Scalar -> pow.Tensor_Tensor` in `SCALAR_OPS`; `or not hasattr(n.target, "_schema")` in `_lift()`; `isinstance(first_arg, fx.Node)` in `_build_tensor_constant` | All three applied |
| `partition/qnn_partitioner.py` | Early return False if `node.target.__name__ not in self.node_visitors` | Applied |
| `quantizer/rules.py` | `if node is None: continue` in `_mark_nodes_as_annotated` | Applied |
| `quantizer/annotators/htp_rules.py` | LayerNorm guard `weight_node`/`bias_node` with `is not None` | Applied |
| `builders/op_layer_norm.py` | Synthesize `torch.ones(normalized_shapes)` / `torch.zeros(normalized_shapes)` when weight/bias are None | Applied |

**Every patch she listed, we now have.** Nothing missing from her documented
patches. She may have had more, undocumented ones.

## 2. What her `export_flux2_klein_qnn.py` does (methodology)

Quant config (lines ~260-300):
```python
quantizer = QnnQuantizer(
    backend=QnnExecuTorchBackendType.kHtpBackend,
    soc_model=chipset,
)
quantizer.set_default_quant_config(qnn_quant_dtype)
```

**Critical:** she does NOT pass `is_linear_per_channel=True`. Her linears get
per-TENSOR weight quantization. That's much lossier than per-channel on a
3072×3072 Linear. We just added this option in our script — it's an
improvement over April's code.

**She also doesn't pass `act_observer`.** Stock default is probably
`PerTensorMinMaxObserver` — picks absolute extremes, fragile to outliers.
`MovingAverageMinMaxObserver` (what we just added) is more robust.

Calibration (lines ~300-350):
- If `--calibration_dir` provided: iterates the saved samples.
- Otherwise: `torch.randn_like(inp)` for every float input — random gaussian
  around zero, completely unlike real transformer activations.
- Default `--num_calibration 10`. Not many passes.

HTP compile spec (lines ~355-370):
```python
backend_options = generate_htp_compiler_spec(use_fp16=use_fp16)
compiler_specs = generate_qnn_executorch_compiler_spec(
    soc_model=chipset,
    backend_options=backend_options,
)
```

**She uses pure defaults.** In particular:
- `use_dlbc=False` (Deep Learning Bandwidth Compression off)
- `use_multi_contexts=False` (no shard-fill across contexts)
- `use_weight_sharing=False`
- `use_slc_allocator=False` (System Level Cache Allocator off; only some SoCs)
- `use_mha2sha=False` (Multi-Head→Single-Head Attention transform off)

**`use_mha2sha` is interesting and worth trying.** On V81 it may restructure
attention in a way that fits HTP better numerically — potentially helps the
w8a8 noise issue. April didn't use it.

No sharding, no explicit fallback boundaries, no explicit SDPA
decomposition — she trusts v1.2.0's default passes.

Lowering:
```python
delegated_program = to_edge_transform_and_lower_to_qnn(m, sample_inputs, compiler_specs)
et_program = delegated_program.to_executorch()
```

Single helper call, same as we now use.

## 3. What her runner does differently from ours

Her `flux2_qnn_main.cpp` has two things our `flux2_main.cpp` lacks:

**a. Calls `runtime::runtime_init()` explicitly.** We actually do this too
(line 275 of ours), so parity.

**b. Sets QNN runtime options at startup:**
```cpp
BackendOptions<3> backend_options;
backend_options.set_option(QNN_RUNTIME_LOG_LEVEL, FLAGS_log_level);
backend_options.set_option(QNN_RUNTIME_HTP_PERFORMANCE_MODE,
                           FLAGS_htp_performance_mode);
set_option(QNN_BACKEND, backend_options.view());
```

Default `--htp_performance_mode 3` = high-performance. Her invocation
explicitly passes it on the command line:
```
./flux2_qnn_main ... --htp_performance_mode 3
```

**Our runner doesn't set this.** Without it, HTP runs in whatever the stock
default is. From `qc_schema.py`: `kHtpDefault = 0`, which per the comment
"sets no configurations on the HTP". The HTP could be in low-power or
throttled mode, slower inference and potentially different numerical
behaviour (some HTP modes skip optimisations).

**Action:** port her `BackendOptions` setup into our runner, or rebuild
using her `flux2_qnn_main.cpp`.

## 4. What her push script tells us

From `push_htp.sh`:
```bash
# adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV75Stub.so $DST/
# adb push $QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so $DST/
```

**She used V75 libs — SM8650, not V81 / SM8850.** The user says April ran on
the same S26 Ultra phone we have (SM8850), but her push script is targeting
V75. Two possible explanations:
1. The script is a stale template she never updated for V81.
2. She was initially testing on an older SoC and the guide/script reflect
   that history.

If the latter, her "pure noise with w8a8 transformer" observation was on V75,
not V81. V81 might behave differently. Worth asking her.

Also note her comment at the top:
```
# Text encoder: XNNPACK CPU
# Transformer: QNN HTP quantized
# VAE decoder: QNN HTP fp16
```

But her `QNN_HTP_GUIDE.md` "Working split" says **transformer on XNNPACK CPU**,
not HTP. The push script reflects her attempted config, the guide reflects
what actually worked.

## 5. What her `collect_calibration_data.py` produces (the quality lever)

For transformer calibration:
- 10 default prompts (diverse styles: cyberpunk, watercolor, portrait, etc.)
- Each prompt: run through real text encoder → `prompt_embeds`
- 2 timesteps per prompt, linearly spaced in [0, 1]
- Fresh random `latents` per sample
- Total: 20 transformer calibration samples with REAL prompt_embeds at
  multiple timesteps

This is dramatically better than her default path's `torch.randn_like` for
every float input. Per her guide: *"Real prompts produce better activation
ranges than random data."*

**We have never run `collect_calibration_data.py`.** Every export we've done
used the synthetic-random default. This is probably the single biggest
noise-reduction lever we haven't pulled yet.

## 6. Open questions that would be good to ask April

Already listed. Adding a few more based on the audit:

1. What ExecuTorch commit hash did you build against (or pip version)?
2. What QAIRT version?
3. Did the "pure noise" observation come from V75 or V81 hardware?
4. Did you try `use_mha2sha`, `use_dlbc`, or `use_slc_allocator` in the
   HTP spec?
5. Did you ever get past "pure noise" with `--calibration_dir` + real
   prompts, or was it noise even with real calibration?
6. Was the noise present on first-step output, or did it accumulate over
   the 4 denoising steps?
7. For the VAE-on-HTP fp16 path that you said worked — what was the quality
   bar? (i.e. are we aiming to match that image or exceed it?)

## 7. Noise causes — ranked by severity, for our setup

Given we're now matching April's methodology + adding per-channel weights +
better observer, the remaining noise suspects (big → small):

**Rank 1: Calibration data.** Synthetic random vs real prompt trajectories
is a massive delta for static quantization. Easy fix: run
`collect_calibration_data.py`, pass `--calibration_dir`.

**Rank 2: Softmax saturation.** Int8 softmax saturates aggressively — one
dominant key dominates attention, rest rounds to zero. Possible mitigations:
- `quantizer.add_discard_ops([torch.ops.aten._safe_softmax.default,
  torch.ops.aten.softmax.int])` — keeps softmax in fp16/fp32
- `use_mha2sha=True` in HTP spec — restructure MHA to SHA, may help scheduling

**Rank 3: Partition boundary requant.** With v1.2.0 native LN support we're
down to 2 partitions total. Much better than April's ~32. Each boundary is
int8→fp32→int8 — fewer boundaries = less error.

**Rank 4: Accumulator precision in attention softmax exp().** QK^T in int8,
exp() approximated. Limited precision.

**Rank 5: Runtime HTP performance mode.** If `kHtpDefault=0` sets different
numerics than `kHtpHighPerformance=3`, switching could silently change
output. (Unlikely but cheap to test.)

## 8. Concrete actions that *add* over April's methodology

What we have that April didn't:
- `is_linear_per_channel=True` and `is_conv_per_channel=True` in quantizer
  config. Per-channel weight quant is strictly more accurate; we just
  enabled it.
- `MovingAverageMinMaxObserver` for activations. More robust than stock
  `PerTensorMinMaxObserver` to outliers.
- Our XNNPACK CPU path is a KNOWN-GOOD reference for `debug_quant_quality.py`
  to compare against (April didn't have this tool).

What April had that we don't yet:
- `--htp_performance_mode 3` in the runner.
- Real calibration data. We've never run `collect_calibration_data.py`.

## 9. Why the transformer compile is taking so long

Tangential but relevant — the full transformer compile at opt_level=3 is
pathologically slow on our setup (the mini took 8 min; full is still
running 2h+). The opt_level is hardcoded in
`HtpGraphCustomConfig.cpp::CreateGraphCustomConfig` — x86 host uses 3,
aarch64 device uses 1. We binary-patched v0.6.0's `.so` earlier (worked);
v1.2.0's `.so` has 3 occurrences of the 3.0f constant instead of 1 (not
all are the opt_level — some are other HTP constants). Safe patching
requires disassembly to confirm the right offset.

April presumably let the compile run long, or had a host machine
sufficiently beefy it didn't matter, or patched the opt_level binary
(unlikely — she would have documented it).

## 10. Bottom line

The working methodology converges on:
1. ExecuTorch v1.2.0 with the 5 backend patches April documented (all applied).
2. Her export_flux2_klein_qnn.py recipe, with per-channel weights added.
3. Run `collect_calibration_data.py` and pass `--calibration_dir`.
4. In the runner, set `QNN_RUNTIME_HTP_PERFORMANCE_MODE=3`.
5. If still noisy, try `use_mha2sha=True`, or fall back to 16a8w for
   transformer, or discard softmax from quantization.

We're already doing (1), (2) with improvements, and (3) is the next lever
to pull.

---

## Final status (2026-04-24)

After applying **all** documented patches and many more iterations:

- **mini-transformer (3 blocks) w8a8 AOT:** compiles in 8 min on v1.2.0 with our patches. Proof the recipe works at small scale.
- **full transformer (25 blocks) w8a8 AOT:** host compile hangs 2+ hours with no progress, RAM growing to 200 GB. Same behavior with online_prepare=True. Sharding via `llama.fallback` did NOT split into multiple HTP partitions (partitioner still reports 1/2 total), so the slow path is still taken. We don't yet know why it's superlinear.
- **full VAE fp16 (no quant) on v1.2.0:** compiles in 3 seconds. 198 MB. Confirms HTP compile itself is fast at level=1 when not doing w8a8 static quant on a big graph.
- **full VAE w8a8:** also hangs (tested, >1h30m no progress).

### Bundle that's ready to push (2026-04-24)

`flux2_phone_ship/` — **pure XNNPACK CPU**: text_encoder, transformer, and VAE all on ARM CPU via XNNPACK dynamic w8a8. 7.7 GB. Runs everywhere, produces correct images, but is slow (minutes per image).

This is *not* April's shipping config — hers had VAE on QNN HTP fp16 — but it's the equivalent safe baseline. The VAE swap to QNN HTP fp16 is a small latency improvement (~0.2 GB of the ~4 GB weight sum runs on NPU instead of CPU), not a game-changer.

### What we'd need to unblock the real goal (transformer on HTP)

The only path known to work today is:
1. Ask April what exact version of executorch/qairt she built with (her pip cache or git hash). Our best guess is v1.2.0 but the symptoms suggest her version has something we don't.
2. OR: upgrade our runner + runtime library to v1.2.0 (rebuild the Android libqnn_executorch_backend.so from source) so we can use the v1.2.0 .pte files we can produce for small models.
3. OR: investigate the superlinear compile on v1.2.0 x86 host — likely a Qualcomm bug specific to some op pattern in the full transformer.

All three are non-trivial and shouldn't block shipping the XNNPACK bundle.
