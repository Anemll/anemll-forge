"""E2 (20 Nov 2026): INT8 activation scales for the adapter MLPs (W4A8), calibrated on the GPTQ calibration inputs
(jetspec_pp/calib_inputs: 4096 sampled rows of every adapter Linear's input, gsm8k traces 8-15, held out from the
eval sets 0-7). In each matrix's own rotated basis (R x, the sign of adapter_lut4hg/<name>.weight.npz): per tensor,
the clip A minimizing the mean squared INT8 error over a grid of A / max|R x| in {1, 0.95, ..., 0.3}; also the error
of per-token scaling (s = max|R x_t| / 127) for comparison. Writes act_scales_tensor.json {linear name: A} (the
replay's JSP_ACTQ_SCALES), ane_ascale.json {block: {h: max(A_gate, A_up), a: A_down}} (the ANE timing build's
MLP_ASCALE: one scale for the shared gate / up input), report.json (relative error per tensor, both recipes).
    PY e2_act_scales.py OUT_DIR"""
import json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant_adapter_gptq import rot_rows  # noqa: E402

C = Path("/Volumes/SSD4TB/anemll-forge-research/jetspec_pp")
BLOCKS = ["streams.0", "streams.1", "streams.2", "streams.3", "extra_stream_layers.3.0", "extra_stream_layers.3.1",
          "extra_stream_layers.3.2", "streams.4", "extra_stream_layers.4.0", "extra_stream_layers.4.1",
          "extra_stream_layers.4.2", "extra_stream_layers.4.3"]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
A, ane, rep = {}, {}, {}


def fq(x, s):
    return torch.clamp(torch.round(x / s), -127, 127) * s


for b in BLOCKS:
    for m in ("gate_proj", "up_proj", "down_proj"):
        name = f"{b}.mlp.{m}"
        x = torch.from_numpy(np.load(C / "calib_inputs" / f"{name}.npy").astype(np.float32))
        z = np.load(C / "adapter_lut4hg" / f"{name}.weight.npz")
        xr = rot_rows(x, torch.from_numpy(z["sign"].astype(np.float32)))
        nrm = xr.pow(2).sum()
        mx = float(xr.abs().max())
        best = None
        for r in np.arange(1.0, 0.29, -0.05):
            e = float((fq(xr, r * mx / 127) - xr).pow(2).sum() / nrm)
            if best is None or e < best[1]:
                best = (float(r), e)
        st = xr.abs().amax(-1, keepdim=True).clamp_min(2 ** -9) / 127
        e_tok = float((fq(xr, st) - xr).pow(2).sum() / nrm)
        A[name] = best[0] * mx
        rep[name] = {"rows": int(x.shape[0]), "max_abs_rot": mx, "clip_ratio": best[0], "A": A[name],
                     "rel_err_tensor": best[1] ** 0.5, "rel_err_token": e_tok ** 0.5,
                     "rel_err_tensor_noclip": float((fq(xr, mx / 127) - xr).pow(2).sum() / nrm) ** 0.5}
        print(f"{name:42s} max|Rx| {mx:8.3f} clip {best[0]:.2f}: rel err per tensor {best[1] ** 0.5:.4f} "
              f"(no clip {rep[name]['rel_err_tensor_noclip']:.4f}), per token {e_tok ** 0.5:.4f}", flush=True)
    ane[b] = {"h": max(A[f"{b}.mlp.gate_proj"], A[f"{b}.mlp.up_proj"]), "a": A[f"{b}.mlp.down_proj"]}
(out / "act_scales_tensor.json").write_text(json.dumps(A, indent=1))
(out / "ane_ascale.json").write_text(json.dumps(ane, indent=1))
(out / "report.json").write_text(json.dumps(rep, indent=1))
print("mean rel err: per tensor %.4f, per token %.4f" % (np.mean([r["rel_err_tensor"] for r in rep.values()]),
                                                       np.mean([r["rel_err_token"] for r in rep.values()])), flush=True)
