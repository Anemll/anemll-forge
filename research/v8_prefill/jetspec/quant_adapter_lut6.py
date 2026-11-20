"""B8 (20 Nov 2026): the JetSpec++ adapter's Linear weights as LUT6 in the ANE build's form (lut_tensor.lut_quantize:
per-tensor 64-entry palette + per-output-channel scale), one .npz per tensor (lut fp16 (64, 1), idx uint8, scale fp16)
in OUT_DIR, plus the norms (fp16) the blocks need. Tensor by tensor on the CPU (low RAM).
    PY quant_adapter_lut6.py OUT_DIR [bits]"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lut_tensor import lut_quantize, lut_dequantize  # noqa: E402

SRC = "/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/release/adapter/head.safetensors"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
bits = int(sys.argv[2]) if len(sys.argv) > 2 else 6
f = safe_open(SRC, "pt")
rep = {}
t0 = time.time()
for k in f.keys():
    t = f.get_tensor(k)
    if t.ndim == 2 and "norm" not in k:
        lut, idx, sc = lut_quantize(t, bits)
        rel = float((lut_dequantize(lut, idx, sc) - t.float()).norm() / t.float().norm())
        np.savez(out / f"{k}.npz", lut=lut.numpy(), idx=idx.numpy(), scale=sc.numpy())
        rep[k] = {"shape": list(t.shape), "rel_err": round(rel, 5)}
    else:
        np.save(out / f"{k}.npy", t.float().numpy().astype(np.float16))
        rep[k] = {"shape": list(t.shape), "kept": "fp16"}
(out / "report.json").write_text(json.dumps({"bits": bits, "source": SRC, "seconds": round(time.time() - t0), "tensors": rep}, indent=1))
errs = [v["rel_err"] for v in rep.values() if "rel_err" in v]
print(f"{len(errs)} matrices at LUT{bits}: rel err mean {np.mean(errs):.4f} max {np.max(errs):.4f}; {len(rep) - len(errs)} kept fp16; {time.time() - t0:.0f}s")
