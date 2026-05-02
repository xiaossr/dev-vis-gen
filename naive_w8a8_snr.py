"""
Naive w8a8 SNR sweep on FLUX.2-klein-4B transformer.

Goal: answer whether FLUX is intrinsically un-quantizable at w8a8, or whether
QNN's specific PT2E (per-tensor static) is the problem.

Variants:
  1. weight-only int8           (activations bf16)        -> bound from weights alone
  2. dynamic w8a8 (per-token)   (torchao default)         -> activations quantized at runtime
  3. static  w8a8 (per-tensor)  (calibrated, hand-rolled) -> mirrors QNN's scheme

Reference: the unquantized model in the same dtype. We use bf16 because the
4B-param fp32 model does not fit comfortably on a single 4090. The int8 noise
floor we are measuring is 30+ dB louder than any bf16 vs fp32 mismatch, so the
choice of reference dtype does not change the conclusion.
"""

from __future__ import annotations

import math
import sys
import gc
from pathlib import Path

import torch

REPO = Path("/data/home/thanush/dev-vis-gen")
sys.path.insert(0, str(REPO))

from export_flux2_klein_qnn import (  # noqa: E402
    Flux2TransformerWrapper,
    build_transformer_inputs,
    load_pipeline,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
CAL_PATH = REPO / "calibration_data" / "calibration_transformer.pt"


# -----------------------------------------------------------------------------#
# Metrics                                                                       #
# -----------------------------------------------------------------------------#
def snr_metrics(ref: torch.Tensor, out: torch.Tensor) -> dict:
    ref = ref.float().flatten()
    out = out.float().flatten()
    diff = ref - out
    rms_ref = ref.norm().item()
    rms_err = diff.norm().item()
    snr_db = 20.0 * math.log10(rms_ref / max(rms_err, 1e-30))
    cos = torch.nn.functional.cosine_similarity(ref.unsqueeze(0), out.unsqueeze(0)).item()
    return dict(
        snr_db=snr_db,
        cos=cos,
        rms_ref=rms_ref,
        rms_err=rms_err,
    )


# -----------------------------------------------------------------------------#
# Static per-tensor w8a8 (mirrors QNN PT2E scheme)                              #
# -----------------------------------------------------------------------------#
class StaticW8A8Linear(torch.nn.Module):
    """Per-channel int8 weight + per-tensor static int8 input.

    Drop-in replacement for nn.Linear; calibrated input scale `act_scale` is
    populated by an MinMax observer pass before inference.
    """

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        # per-output-channel weight scale (symmetric)
        with torch.no_grad():
            w = lin.weight.detach()
            w_max = w.abs().amax(dim=1).clamp(min=1e-8)
            w_scale = w_max / 127.0
            w_int = torch.round(w / w_scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        self.register_buffer("w_int", w_int)
        self.register_buffer("w_scale", w_scale.to(torch.float32))
        self.bias = lin.bias  # keep fp
        # placeholders set by the observer pass
        self.register_buffer("act_scale", torch.tensor(1.0))
        self.register_buffer("act_max", torch.tensor(0.0))
        self.observing = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observing:
            cur = x.detach().abs().amax().float()
            if cur > self.act_max:
                self.act_max.copy_(cur)
            w_dq = self._dequant_w().to(x.dtype)
            return torch.nn.functional.linear(x, w_dq, self.bias)

        # quantize -> matmul -> dequant
        scale = self.act_scale
        x_int = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
        x_dq = x_int.to(x.dtype) * scale.to(x.dtype)
        w_dq = self._dequant_w().to(x.dtype)
        return torch.nn.functional.linear(x_dq, w_dq, self.bias)

    def _dequant_w(self) -> torch.Tensor:
        # restore in fp32, caller can cast
        return self.w_int.to(torch.float32) * self.w_scale.unsqueeze(1)

    def finalize(self) -> None:
        self.act_scale.copy_(self.act_max.clamp(min=1e-8) / 127.0)
        self.observing = False


def _replace_linears_static(module: torch.nn.Module) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            new = StaticW8A8Linear(child).to(child.weight.device)
            setattr(module, name, new)
            n += 1
        else:
            n += _replace_linears_static(child)
    return n


def _set_observing(module: torch.nn.Module, val: bool) -> None:
    for m in module.modules():
        if isinstance(m, StaticW8A8Linear):
            m.observing = val


def _finalize_static(module: torch.nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, StaticW8A8Linear):
            m.finalize()


# -----------------------------------------------------------------------------#
# Driver                                                                        #
# -----------------------------------------------------------------------------#
def to_dev(inputs, device, dtype):
    out = []
    for x in inputs:
        x = x.to(device)
        if x.is_floating_point():
            x = x.to(dtype)
        out.append(x)
    return tuple(out)


def run_forward(model, inputs):
    with torch.no_grad():
        return model(*inputs).detach().float().cpu()


def make_fresh_wrapper(pipe):
    """Build a wrapper from a fresh deep copy so quantization passes don't taint."""
    import copy
    transformer = copy.deepcopy(pipe.transformer)
    return Flux2TransformerWrapper(transformer).eval()


def main():
    print(f"[setup] device={DEVICE} dtype={DTYPE}")
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=DTYPE)
    # We only need the transformer + its config; drop the rest to save memory.
    sample_inputs = build_transformer_inputs(pipe, 512, 512, 512, dtype=DTYPE)

    cal_data = torch.load(CAL_PATH, map_location="cpu", weights_only=False)
    print(f"[setup] {len(cal_data)} calibration samples")
    probe = cal_data[0]
    probe_dev = to_dev(probe, DEVICE, DTYPE)

    # ---- 1. fp reference -----------------------------------------------------#
    print("\n[ref] running bf16 forward...")
    base = Flux2TransformerWrapper(pipe.transformer).eval().to(DEVICE)
    ref = run_forward(base, probe_dev)
    print(f"[ref] out shape {tuple(ref.shape)}, |ref|={ref.norm().item():.3f}")

    # we don't need the original transformer in fp anymore, but we keep `pipe` for
    # rebuilding wrappers from deepcopy

    # ---- 2. Weight-only int8 (torchao) ---------------------------------------#
    print("\n[woi8] applying torchao Int8WeightOnlyConfig ...")
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    woi8 = make_fresh_wrapper(pipe).to(DEVICE)
    quantize_(woi8, Int8WeightOnlyConfig())
    out_woi8 = run_forward(woi8, probe_dev)
    m_woi8 = snr_metrics(ref, out_woi8)
    print(f"[woi8] SNR={m_woi8['snr_db']:.2f} dB  cos={m_woi8['cos']:.4f}")
    del woi8
    torch.cuda.empty_cache(); gc.collect()

    # ---- 3. Dynamic w8a8 (torchao) -------------------------------------------#
    print("\n[dyn] applying torchao Int8DynamicActivationInt8WeightConfig ...")
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig
    dyn = make_fresh_wrapper(pipe).to(DEVICE)
    quantize_(dyn, Int8DynamicActivationInt8WeightConfig())
    out_dyn = run_forward(dyn, probe_dev)
    m_dyn = snr_metrics(ref, out_dyn)
    print(f"[dyn] SNR={m_dyn['snr_db']:.2f} dB  cos={m_dyn['cos']:.4f}")
    del dyn
    torch.cuda.empty_cache(); gc.collect()

    # ---- 4. Static w8a8 (per-tensor, calibrated) -----------------------------#
    print("\n[stat] hand-rolled per-tensor static w8a8 ...")
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

    # ---- summary -------------------------------------------------------------#
    print("\n" + "=" * 60)
    print("SNR summary (vs bf16 reference)")
    print("=" * 60)
    for tag, m in [("weight-only int8 (a=bf16)", m_woi8),
                   ("dynamic w8a8 (per-token)", m_dyn),
                   ("static  w8a8 (per-tensor)", m_stat)]:
        print(f"  {tag:32s}  SNR={m['snr_db']:7.2f} dB  cos={m['cos']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
