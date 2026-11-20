"""C4 (20 Nov 2026): the JetSpec++ adapter at LUT<bits> with a Hadamard rotation and GPTQ, in the ANE build's form
(per-tensor palette + per-output-channel scale, lut_tensor.py), from calibration inputs captured by the replay
(jsp_replay.py JSP_CAPTURE: sampled inputs of every adapter Linear, held-out gsm8k traces).
  rotation  x' = R x, R = blockdiag(H_1024) diag(sign) (random signs, seeded by the tensor name; the target builder's
            Hadamard form): W' = W R^T, H' = R H R^T
  palette   2^bits entries fit by Lloyd on w' / max|w'_row| (lut_tensor.lut_quantize)
  GPTQ      columns in blocks of 128, nearest palette entry x row scale, error fed forward with the Cholesky of H'^-1
            (damp 1% of mean diag); MODE=rtn skips GPTQ (Hadamard + nearest entry)
Writes OUT_DIR/<name>.weight.npz: lut, idx, scale, sign (int8), block. The replay reads W = (lut[idx] scale) R.
    PY quant_adapter_gptq.py CAP_DIR OUT_DIR [bits] [MODE=gptq|rtn]"""
import hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lut_tensor import lut_quantize  # noqa: E402

SRC = "/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/release/adapter/head.safetensors"
BLK = 1024
torch.set_num_threads(8)


def hadamard(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / h.shape[0] ** 0.5


H1024 = hadamard(BLK)


def sign_for(name, n):
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float()


def rot_rows(x, sign):
    """x (..., n) -> R x along the last dim: block Hadamard of (sign * x)"""
    n = x.shape[-1]
    y = (x * sign).reshape(*x.shape[:-1], n // BLK, BLK) @ H1024.T
    return y.reshape(*x.shape[:-1], n)


def gptq(W, H, lut, scale, blocksize=128, damp=0.01):
    """W (o, i) fp32, H (i, i); quantizer: nearest lut entry of w / scale_row. Returns idx (o, i) uint8."""
    o, i = W.shape
    W = W.clone()
    d = torch.diag(H)
    dead = d == 0
    H = H.clone(); H[dead, dead] = 1; W[:, dead] = 0
    H += damp * d.mean() * torch.eye(i)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)
    c = lut.reshape(-1).float()
    mid = 0.5 * (c[1:] + c[:-1])
    s = scale.float()[:, None]
    idx = torch.empty(o, i, dtype=torch.uint8)
    for a in range(0, i, blocksize):
        b = min(a + blocksize, i)
        W1 = W[:, a:b].clone(); Err = torch.zeros_like(W1); U1 = U[a:b, a:b]
        for j in range(b - a):
            w = W1[:, j]
            q = torch.bucketize(w / s[:, 0], mid)
            idx[:, a + j] = q.to(torch.uint8)
            wq = c[q] * s[:, 0]
            e = (w - wq) / U1[j, j]
            W1[:, j:] -= e[:, None] * U1[j, j:][None, :]
            Err[:, j] = e
        W[:, b:] -= Err @ U[a:b, b:]
    return idx


def main():
    cap, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    bits = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    mode = sys.argv[4] if len(sys.argv) > 4 else "gptq"
    f = safe_open(SRC, "pt")
    rep, t0 = {}, time.time()
    for k in f.keys():
        if not (k.endswith("_proj.weight")):
            continue
        name = k[:-len(".weight")]
        W = f.get_tensor(k).float()
        o, i = W.shape
        sign = sign_for(name, i)
        Wr = rot_rows(W, sign)                                      # W R^T: rows of W rotated like x
        lut, idx0, scale = lut_quantize(Wr, bits)
        cf = cap / f"{name}.npy"
        X = torch.from_numpy(np.load(cf)).float() if cf.exists() else None
        Xr = rot_rows(X, sign) if X is not None else None
        if mode == "gptq":
            H = Xr.T @ Xr / X.shape[0]
            idx = gptq(Wr, H, lut, scale)
            del H
        else:
            idx = idx0
        wq = lut.reshape(-1).float()[idx.long()] * scale.float()[:, None]
        rel_w = float((wq - Wr).norm() / Wr.norm())
        rel_o = float("nan")
        if Xr is not None:
            xs = Xr[:512]
            rel_o = float(((wq - Wr) @ xs.T).norm() / (Wr @ xs.T).norm())   # output error on calibration rows
        del X, Xr
        np.savez(out / f"{k}.npz", lut=lut.numpy(), idx=idx.numpy(), scale=scale.numpy(), sign=sign.to(torch.int8).numpy(),
                 block=np.array(BLK))
        rep[k] = {"shape": [o, i], "rel_err_weights": round(rel_w, 5), "rel_err_output_calib": round(rel_o, 5)}
        print(f"{k} {tuple(W.shape)} {mode} LUT{bits}: weight rel err {rel_w:.4f}, output rel err {rel_o:.4f} ({time.time() - t0:.0f}s)", flush=True)
        (out / "report.json").write_text(json.dumps({"bits": bits, "mode": mode, "block": BLK, "tensors": rep}, indent=1))


if __name__ == "__main__":
    main()
