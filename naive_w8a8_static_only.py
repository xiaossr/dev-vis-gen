"""Run only the static per-tensor w8a8 variant (the QNN-mimicking control).

Reuses the StaticW8A8Linear / driver code from naive_w8a8_snr.py but skips the
torchao variants that already finished successfully.
"""
import sys
from pathlib import Path
import torch

REPO = Path("/data/home/thanush/dev-vis-gen")
sys.path.insert(0, str(REPO))

from naive_w8a8_snr import (  # noqa: E402
    DEVICE, DTYPE, CAL_PATH, snr_metrics, to_dev, run_forward,
    make_fresh_wrapper, _replace_linears_static, _set_observing,
    _finalize_static,
)
from export_flux2_klein_qnn import (  # noqa: E402
    Flux2TransformerWrapper, build_transformer_inputs, load_pipeline,
)

print(f"[setup] device={DEVICE} dtype={DTYPE}")
pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=DTYPE)
sample_inputs = build_transformer_inputs(pipe, 512, 512, 512, dtype=DTYPE)

cal_data = torch.load(CAL_PATH, map_location="cpu", weights_only=False)
probe = cal_data[0]
probe_dev = to_dev(probe, DEVICE, DTYPE)

print("[ref] running bf16 forward...")
base = Flux2TransformerWrapper(pipe.transformer).eval().to(DEVICE)
ref = run_forward(base, probe_dev)
print(f"[ref] |ref|={ref.norm().item():.3f}")

print("[stat] hand-rolled per-tensor static w8a8 ...")
stat = make_fresh_wrapper(pipe).to(DEVICE)
n = _replace_linears_static(stat)
print(f"[stat] replaced {n} nn.Linear modules")
_set_observing(stat, True)
print(f"[stat] calibrating on {len(cal_data)} samples ...")
with torch.no_grad():
    for i, sample in enumerate(cal_data):
        stat(*to_dev(sample, DEVICE, DTYPE))
        if (i + 1) % 5 == 0:
            print(f"  cal {i+1}/{len(cal_data)}")
_finalize_static(stat)
print("[stat] running quantized forward ...")
out_stat = run_forward(stat, probe_dev)
m_stat = snr_metrics(ref, out_stat)
print(f"[stat] SNR={m_stat['snr_db']:.2f} dB  cos={m_stat['cos']:.4f}")
