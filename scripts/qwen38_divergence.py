"""Where does the ANE lose ~0.1 nats vs PyTorch (same export)? Divergence of a v4 ANE build against an fp32 CPU
reference of the SAME quantized weights (dequantized LUTs + low-rank factors): weight quantization cancels, what is
left is the ANE path (fp16 activations / accumulation, ANE op approximations, our ANE graph rewrites).
    ref : fp32 CPU reference (dflash2_target_ref streamed layers): hidden state after EVERY layer + final logits
    bf16: the same with the unquantized checkpoint (ground truth; dflash2_target_ref matches the HF modules)
    ane : the ANE build (a) end to end, (b) per chunk ISOLATED - each chunk fed the reference input of its first layer,
          so its own error is separated from inherited error. Per chunk: global rel-L2 / cos, the median per-token
          rel error, and the per-token rel error by position quarter (flat = per-token numerics, rising = state
          drift); plus the NLL / top-1 agreement of the end-to-end run. Works for any chunk plan: a build with one
          layer per chunk (CHUNK_PLAN=0-0,1-1,...) gives every layer's own ANE error (DeltaNet vs attention).
Tokens: an in-domain trace sequence (TRACE, SEQ) or a row of a token matrix (TOKENS=<rows.npy>, ROW), first NTOK.
    TOKENS=~/Models/vq27b/tests/div_pi2k.npy NTOK=2048 EXPORT_DIR=~/Models/vq27b/export/mix25in_aw_cal_lr64mix \
        python qwen38_divergence.py ref
    ANE_OUT=~/Models/vq27b/ane7 CTX=16384 (same TOKENS / NTOK / EXPORT_DIR) python qwen38_divergence.py ane"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

EXPORT = Path(os.path.expanduser(os.environ.get("EXPORT_DIR", "~/Models/vq27b/export/mix25_aw_cal_lr64mix")))
NTOK = int(os.environ.get("NTOK", "256"))
SEQ = int(os.environ.get("SEQ", "0"))
TOKENS = os.path.expanduser(os.environ["TOKENS"]) if os.environ.get("TOKENS") else None
ROW = int(os.environ.get("ROW", "0"))
SRC = f"{Path(TOKENS).stem}_r{ROW}" if TOKENS else f"trace_s{SEQ}"
DIR = Path(os.path.expanduser(os.environ.get("DIV_OUT", "~/Models/vq27b/tests")))
BF16 = len(sys.argv) > 1 and sys.argv[1] == "bf16"  # "bf16": the reference with the unquantized checkpoint
OUT = DIR / f"div_{EXPORT.name}_{SRC}_{NTOK}.npz"
OUT_BF16 = DIR / f"div_bf16_{SRC}_{NTOK}.npz"
TRACE = os.path.expanduser(os.environ.get("TRACE", "~/Models/vq27b/kl/trace.npz"))
MODEL = Path(os.path.expanduser(os.environ.get("MODEL", "~/Models/Qwen3.8-27B")))


def tokens():
    if TOKENS:
        return np.load(TOKENS)[ROW, :NTOK].astype(np.int64)
    d = np.load(TRACE)
    start = int(np.concatenate([[0], np.cumsum(d["lengths"])])[SEQ])
    return d["ids"][start:start + min(NTOK, int(d["lengths"][SEQ]))].astype(np.int64)


def ref():
    os.environ.setdefault("MODEL", str(MODEL))
    if BF16:  # the unquantized checkpoint: ground truth
        os.environ.pop("EXPORT_DIR", None)
    else:
        os.environ["EXPORT_DIR"] = str(EXPORT)
    os.environ.setdefault("CTX_MAX", str(NTOK + 8))
    import dflash2_target_ref as R  # Weights.layer dequantizes the export AND adds its low-rank factors (a @ b)

    ids = tokens()
    tg = R.Target()
    st = R.SeqState(tg.cfg)
    jobs = [(st, 0, len(ids), True)]
    x = tg.embed(ids)
    hl, t0 = [], time.time()
    for i in range(64):
        w = tg.W.layer(i)
        h = R.rms_zc(x, w["input_layernorm.weight"], tg.eps)
        kind = tg.cfg["layer_types"][i]
        x = x + (tg.gdn(i, w, h, jobs) if kind == "linear_attention" else tg.attn(i, w, h, jobs))
        x = x + tg.mlp(w, R.rms_zc(x, w["post_attention_layernorm.weight"], tg.eps))
        hl.append(x.numpy().astype(np.float32))
        if i % 4 == 3:
            print(f"layer {i}: {time.time() - t0:.0f}s", flush=True)
    logits = tg.head(R.rms_zc(x, tg.final_norm, tg.eps)).numpy()
    emb = tg.embed(ids).numpy().astype(np.float32)
    np.savez(OUT_BF16 if BF16 else OUT, ids=ids, emb=emb, hl=np.stack(hl), top=logits.argmax(1),
             lse=np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1)) + logits.max(1),
             tgt_logit=logits[np.arange(len(ids) - 1), ids[1:]])
    print(f"saved {OUT_BF16 if BF16 else OUT} ({len(ids)} tokens, {time.time() - t0:.0f}s)", flush=True)


def per_token(a, b):
    """Per-token rel-L2 error of a vs b, both (N, hid)."""
    return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-6)


def ane():
    import qwen38_ane_model as M
    d = np.load(OUT)
    ids, hl, emb = d["ids"].tolist(), d["hl"], d["emb"]         # hl (64, N, hid): after every layer
    n = len(ids)
    m = M.AneQwen3()
    T = m.T
    kinds = json.loads((MODEL / "config.json").read_text())["text_config"]["layer_types"]
    ranges = [tuple(r) for r in m.layer_ranges]
    q = [slice(k * n // 4, (k + 1) * n // 4) for k in range(4)]

    def glob(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b)), float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))

    # (a) end to end: record every chunk's output rows
    m.reset()
    got = np.zeros((len(m.chunks), n, hl.shape[2]), np.float32)
    nll, agree = [], []
    for p in range(0, n, T):
        blk = ids[p:p + T]
        lg = m.call(blk)
        for c, ch in enumerate(m.chunks):
            got[c, p:p + len(blk)] = ch["y"].to_numpy()[0, :, 0, :len(blk)].T
        for r in range(len(blk)):
            if p + r + 1 < n:
                z = lg[r].astype(np.float64)
                nll.append(float(np.log(np.exp(z - z.max()).sum()) + z.max() - z[ids[p + r + 1]]))
                agree.append(int(z.argmax() == d["top"][p + r]))
        m.accept(len(blk))
    ref_nll = d["lse"][:-1] - d["tgt_logit"]
    nll = np.array(nll)
    print(f"end to end: ANE NLL {nll.mean():.4f} vs fp32 reference {ref_nll.mean():.4f} "
          f"(+{nll.mean() - ref_nll.mean():.4f} nats), top-1 agreement {np.mean(agree):.3f}; excess NLL by position "
          f"quarter: " + " ".join(f"{(nll[s_] - ref_nll[s_]).mean():+.4f}" for s_ in q), flush=True)
    # (b) isolated: chunk c alone, fed the reference input (embeddings for the first layer), own state
    print("chunk layers  types  | end-to-end rel / cos  | isolated rel / cos   median tok | isolated per-token rel "
          "by position quarter", flush=True)
    e2e_pos, iso_pos = np.zeros((len(ranges), n), np.float32), np.zeros((len(ranges), n), np.float32)
    rows = []
    for c, (a, b) in enumerate(ranges):
        m.reset()
        iso = np.zeros((n, hl.shape[2]), np.float32)
        xin_all = emb if a == 0 else hl[a - 1]
        for p in range(0, n, T):
            blk = ids[p:p + T]
            x = np.zeros((1, hl.shape[2], 1, T), np.float16)
            x[0, :, 0, :len(blk)] = xin_all[p:p + len(blk)].T
            y = m.call_chunk(c, blk, x)
            iso[p:p + len(blk)] = y[:len(blk)]
            m.accept(len(blk))
        tgt = hl[b]
        e2e, iso_g = glob(got[c], tgt), glob(iso, tgt)
        e2e_pos[c], iso_pos[c] = per_token(got[c], tgt), per_token(iso, tgt)
        # the layers' own contribution: isolated error relative to the size of this chunk's update (x_out - x_in)
        upd = per_token(iso, tgt) * np.linalg.norm(tgt, axis=1) / np.maximum(
            np.linalg.norm(tgt - xin_all, axis=1), 1e-6)
        types = "".join("A" if kinds[l] == "full_attention" else "D" for l in range(a, b + 1))
        rows.append((c, a, b, types, *e2e, *iso_g, float(np.median(iso_pos[c])), float(np.median(upd))))
        print(f"{c:3d}  {a:2d}-{b:2d}  {types:5s}  | {e2e[0]:.4f} / {e2e[1]:.5f}   | {iso_g[0]:.4f} / {iso_g[1]:.5f}   "
              f"{np.median(iso_pos[c]):.4f}  (of update {np.median(upd):.4f}) | "
              + " ".join(f"{np.median(iso_pos[c][s_]):.4f}" for s_ in q), flush=True)
    res = DIR / f"divane_{M.OUT.parent.name}_{EXPORT.name}_{SRC}_{NTOK}.npz"
    np.savez(res, ranges=np.array(ranges), e2e_pos=e2e_pos, iso_pos=iso_pos, nll=nll, ref_nll=ref_nll,
             rows=np.array([r[4:] for r in rows], np.float64))
    print(f"saved {res}", flush=True)


if __name__ == "__main__":
    {"ref": ref, "bf16": ref, "ane": ane}[sys.argv[1]]()
