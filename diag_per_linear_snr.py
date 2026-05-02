"""Per-Linear SNR analysis.

For each nn.Linear in FLUX, simulate per-tensor int8 input quant + per-channel
int8 weight quant (matching torchao Int8DynamicActivationInt8WeightConfig
recipe), measure local SNR contribution.

Outputs:
  - ranked list (worst SNR first) -> candidates to keep fp16
  - ranked list (largest err_norm) -> largest absolute contributors

Both rankings written to per_linear_snr.json.
"""
import json, logging, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from export_flux2_klein_qnn import (
    Flux2TransformerWrapper, configure_local_tooling, load_pipeline,
)
configure_local_tooling()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("psnr")


def quant_per_tensor_int8(x):
    s = x.abs().max() / 127.0
    s = s.clamp(min=1e-8)
    return (torch.round(x / s).clamp(-128, 127) * s).to(x.dtype)


def quant_per_channel_int8(w):
    # weight [out, in]; per output-channel symmetric int8
    s = w.abs().amax(dim=1, keepdim=True) / 127.0
    s = s.clamp(min=1e-8)
    return (torch.round(w / s).clamp(-128, 127) * s).to(w.dtype)


def main():
    log.info("Loading ...")
    pipe = load_pipeline("black-forest-labs/FLUX.2-klein-4B", dtype=torch.float32)
    cal = torch.load(
        str(Path(__file__).parent / "calibration_data" / "calibration_transformer.pt"),
        weights_only=False)[:1]
    probe = cal[0] if isinstance(cal[0], tuple) else (cal[0],)

    transformer = pipe.transformer.eval().cpu()

    # Capture every Linear's input + fp32 output
    captures = {}
    handles = []
    n_linear = 0
    for name, m in transformer.named_modules():
        if isinstance(m, torch.nn.Linear):
            n_linear += 1
            def make_hook(n):
                def h(_m, inp, out):
                    captures[n] = (inp[0].detach().clone(), out.detach().clone())
                return h
            handles.append(m.register_forward_hook(make_hook(name)))

    log.info("fp32 forward to capture %d Linears ...", n_linear)
    with torch.no_grad():
        transformer(
            hidden_states=probe[0], encoder_hidden_states=probe[1], timestep=probe[2],
            img_ids=probe[3], txt_ids=probe[4], guidance=None, return_dict=False,
        )
    for h in handles: h.remove()
    log.info("Captured %d Linears", len(captures))

    rows = []
    for name, m in transformer.named_modules():
        if not isinstance(m, torch.nn.Linear): continue
        if name not in captures: continue
        x, y = captures[name]
        w = m.weight.detach().float()
        b = m.bias.detach().float() if m.bias is not None else None

        xq = quant_per_tensor_int8(x.float())
        wq = quant_per_channel_int8(w)
        yq = xq @ wq.T
        if b is not None: yq = yq + b

        ref = y.float(); q = yq.float()
        snr = (20 * torch.log10(ref.norm() / ((ref - q).norm() + 1e-12))).item()
        cos = torch.nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0), q.flatten().unsqueeze(0)).item()
        out_norm = ref.norm().item()
        err_norm = (ref - q).norm().item()
        rows.append({
            "name": name,
            "shape_in": list(x.shape),
            "shape_w": list(w.shape),
            "snr_db": snr,
            "cos": cos,
            "out_norm": out_norm,
            "err_norm": err_norm,
        })

    log.info("\n=== ALL %d LINEARS ===", len(rows))
    for r in sorted(rows, key=lambda r: r["snr_db"]):
        log.info("%-65s SNR=%6.2f cos=%.4f w=%s",
                 r["name"], r["snr_db"], r["cos"], tuple(r["shape_w"]))

    log.info("\n=== WORST 15 BY SNR ===")
    for r in sorted(rows, key=lambda r: r["snr_db"])[:15]:
        log.info("%-65s SNR=%6.2f cos=%.4f", r["name"], r["snr_db"], r["cos"])

    log.info("\n=== WORST 15 BY ERR_NORM (global contribution proxy) ===")
    for r in sorted(rows, key=lambda r: -r["err_norm"])[:15]:
        log.info("%-65s SNR=%6.2f err=%9.3f out=%9.3f w=%s",
                 r["name"], r["snr_db"], r["err_norm"], r["out_norm"], tuple(r["shape_w"]))

    out_path = Path(__file__).parent / "per_linear_snr.json"
    out_path.write_text(json.dumps(rows, indent=2))
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
