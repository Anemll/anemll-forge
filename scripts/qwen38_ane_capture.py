"""ANE-in-the-loop capture for one-layer chunks of a DBG_MIXER_IN=1 build (qwen38_ane_chunk): each layer is fed the fp32
reference input of that layer (qwen38_divergence.py ref, same TOKENS / NTOK / EXPORT_DIR) and every mixer matrix's
actual ANE input (the dbg outputs: the normed layer input h, and the tensor entering out_proj / o_proj) is saved next
to the reference's input of the same matrix (dflash2_target_ref, out_proj / o_proj replaced by the identity).
Prints the rel-L2 error of each ANE input vs the reference (where the ANE layer departs from the true computation) and
saves layer_XX.npz (fp16: ane_h, ref_h, ane_o, ref_o) for low-rank refits.
    ANE_OUT=~/Models/vq27b/ane7D EXPORT_DIR=~/Models/vq27b/export/mix25in_aw_cal_lr64mix CTX=16384 \
        TOKENS=~/Models/vq27b/tests/div_pi2k.npy NTOK=2048 LAYERS=9,11 python qwen38_ane_capture.py"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import qwen38_divergence as V

LAYERS = [int(x) for x in os.environ.get("LAYERS", "9").split(",")]
SAVE = os.environ.get("SAVE", "1") == "1"


def ref_inputs(tg, R, w, x, kind):
    """Reference (h, o): the normed layer input and the tensor entering out_proj / o_proj, whole sequence."""
    h = R.rms_zc(x, w["input_layernorm.weight"], tg.eps)
    w2 = dict(w)
    key = "linear_attn.out_proj.weight" if kind == "linear_attention" else "self_attn.o_proj.weight"
    w2[key] = torch.eye(w[key].shape[1])
    st = R.SeqState(tg.cfg)
    jobs = [(st, 0, x.shape[0], True)]
    o = tg.gdn(L_CUR[0], w2, h, jobs) if kind == "linear_attention" else tg.attn(L_CUR[0], w2, h, jobs)
    return h.numpy(), o.numpy()


L_CUR = [0]


def main():
    torch.set_grad_enabled(False)
    os.environ.setdefault("MODEL", str(V.MODEL))
    os.environ["EXPORT_DIR"] = str(V.EXPORT)
    os.environ.setdefault("CTX_MAX", str(V.NTOK + 8))
    import dflash2_target_ref as R
    import qwen38_ane_model as M
    d = np.load(V.OUT)
    ids, hl, emb = d["ids"].tolist(), d["hl"], d["emb"]
    n = len(ids)
    m = M.AneQwen3()
    T = m.T
    tg = R.Target()
    kinds = tg.cfg["layer_types"]
    ranges = [tuple(r) for r in m.layer_ranges]
    man = json.loads((M.OUT / f"manifest_ctx{m.ctx}_v4.json").read_text())
    out_dir = V.DIR / f"cap_{M.OUT.parent.name}_{V.SRC}_{V.NTOK}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for L in LAYERS:
        c = ranges.index((L, L))
        dmap = json.loads((M.OUT / man["chunks"][c]["file"]).with_suffix(".dbg.json").read_text())
        mats = sorted({v for v in dmap.values()})
        t0 = time.time()
        m.reset()
        xin = emb if L == 0 else hl[L - 1]
        got = {k: np.zeros((n, 0), np.float32) for k in mats}
        rows = {k: [] for k in mats}
        for p in range(0, n, T):
            blk = ids[p:p + T]
            x = np.zeros((1, hl.shape[2], 1, T), np.float16)
            x[0, :, 0, :len(blk)] = xin[p:p + len(blk)].T
            m.call_chunk(c, blk, x)
            for k in mats:
                rows[k].append(np.asarray(m.last_extra[k])[0, :, 0, :len(blk)].T.astype(np.float32))
            m.accept(len(blk))
        for k in mats:
            got[k] = np.concatenate(rows[k])
        L_CUR[0] = L
        w = tg.W.layer(L)
        h_ref, o_ref = ref_inputs(tg, R, w, torch.from_numpy(xin), kinds[L])
        inv = {v: k for k, v in dmap.items()}
        name_h = dmap[f"{L}:linear_attn.in_proj_qkv"] if kinds[L] == "linear_attention" else dmap[f"{L}:self_attn.q_proj"]
        name_o = dmap[f"{L}:linear_attn.out_proj"] if kinds[L] == "linear_attention" else dmap[f"{L}:self_attn.o_proj"]
        a_h, a_o = got[name_h], got[name_o]

        def rel(a, b):
            return float(np.linalg.norm(a - b) / np.linalg.norm(b)), float(np.median(
                np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-6)))
        eh, eo = rel(a_h, h_ref), rel(a_o, o_ref)
        print(f"layer {L:2d} {'D' if kinds[L] == 'linear_attention' else 'A'}  normed input h: rel {eh[0]:.4f} "
              f"(median token {eh[1]:.4f})   into {inv[name_o].split(':')[1]}: rel {eo[0]:.4f} "
              f"(median token {eo[1]:.4f})   ({time.time() - t0:.0f}s)", flush=True)
        if SAVE:
            np.savez(out_dir / f"layer_{L:02d}.npz", ane_h=a_h.astype(np.float16), ref_h=h_ref.astype(np.float16),
                     ane_o=a_o.astype(np.float16), ref_o=o_ref.astype(np.float16))
    print(f"saved to {out_dir}" if SAVE else "", flush=True)


if __name__ == "__main__":
    main()
