"""E2 (20 Nov 2026): host (CPU torch) reference of the row-sweep packages' mlp<T> entries (block 0's MLP alone), the
same modules build_jsp_adapter_rows.py traces, on the timing script's seed-0 input x ~ N(0, 0.02) (1, 5120, 1, T).
Prints the max |rotated MLP input| / |rotated down input| of that data (the per-tensor scales that fill the INT8 code
range for the TIMING graph) and writes y for MLP=fp16 / w4a8 / w4a8t to OUT.npz.
    env as build_jsp_adapter_rows.py  BUILDPY e2_host_ref.py LUT_DIR OUT.npz T [MLP_ASCALE json]"""
import json, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
lut_dir, outp, T = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
res = {}
x = torch.from_numpy((np.random.default_rng(0).standard_normal((1, 5120, 1, T)) * 0.02).astype(np.float16))
for mode in ("fp16", "w4a8", "w4a8t"):
    os.environ["MLP"] = mode
    import importlib
    import build_jsp_adapter_rows as R
    R = importlib.reload(R)
    B, A = R.B, R.A
    B.KNOWN_LUTS.clear()
    name, i = A.BLOCKS[0], A.ATT_IDX[0]
    W = A.block_W(lut_dir, name, i)
    W[f"{i}/mix.rotation"] = np.array([5000 + i, 6000 + i], np.int64)
    W[f"{i}/mlp.rotation"] = np.array([7000 + i, 8000 + i], np.int64)
    if mode == "fp16":
        m = B.LayerW(W, i).eval().to(torch.float16)
        with torch.no_grad():
            h = B.rms_hidden(x, m.ln2); hr = m.rin(h)
            a = B.silu(m.gate(hr), B.MLP_SILU) * m.up(hr); ar = m.rmid(a)
            y = m.mlp(x)
        res["max_h_rot"], res["max_a_rot"] = float(hr.abs().max()), float(ar.abs().max())
        res["rms_h_rot"], res["rms_a_rot"] = float(hr.float().pow(2).mean().sqrt()), float(ar.float().pow(2).mean().sqrt())
        print(f"timing graph data: max|R h| {res['max_h_rot']:.4f} (rms {res['rms_h_rot']:.4f}), max|R a| {res['max_a_rot']:.4f} (rms {res['rms_a_rot']:.4f})", flush=True)
    else:
        if mode == "w4a8t":
            sc = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 else None
            if sc is None:
                continue
        for mm in ("gate", "up", "down"):
            for f in ("lut", "idx", "scale"):
                W.pop(f"{i}/mlp.{mm}_proj.weight/{f}")
            W[f"{i}/mlp.{mm}_proj.weight/dense"] = np.zeros((1, 1), np.float16)
        R.MLP = mode
        m = R._layer_a8(W, i, lut_dir, name, sc if mode == "w4a8t" else None)
        with torch.no_grad():
            y = m.mlp(x)
    res[f"y_{mode}"] = y.float().numpy()
    print(f"{mode}: y {tuple(y.shape)} finite {bool(torch.isfinite(y).all())}", flush=True)
np.savez(outp, **{k: v for k, v in res.items() if k.startswith("y_")})
outp.with_suffix(".json").write_text(json.dumps({k: v for k, v in res.items() if not k.startswith("y_")}, indent=1))
