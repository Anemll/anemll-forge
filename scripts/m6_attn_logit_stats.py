"""Range of the full-attention scores of Qwen3.8-27B on real text, and the error of 8-bit scores / softmax weights.

Host research (CPU torch, no ANE): the streamed reference of dflash2_target_ref.py (the quantized export via
EXPORT_DIR, or the bf16 checkpoint) runs a teacher-forced prefill over a few real sequences; its attention is
replaced by the same computation split into query-row chunks, and for each of the 16 full-attention layers the
chunk's scores are measured before the forward continues unchanged.

Per layer, over causal-valid entries:
  s   true scores q . k / sqrt(256): min / max, percentiles (fine histogram), row maxima, |s| above 8 / 16 / 24 / 28 / 32
  r   kv8 raw scores q . (codes / 128) / 16 (keys as INT8 codes with a per (KV head, token) scale max|k| / 127,
      as quantize_values in qwen38_kv_cache.py); s_kv8 = r * (scale * 128). Same range statistics plus the
      fraction of r clipped by an INT8 pair with a constant step 1/64 .. 1/4
  simulated outputs o = softmax(.) @ V on a subsample of query rows, against the fp32 reference softmax(s) @ V:
      kv8k        kv8 keys, r unquantized (the key-cache error alone, the baseline of a_r / d)
      a_r<step>   r quantized to INT8 at a constant step 1/64 .. 1/4 (round, clip [-128, 127]) before the scale
                  multiply; a_r1/32z: asymmetric, codes offset by 64 (range [-2, 5.97])
      a_s<step>   s itself quantized to INT8 at a constant step 1/16, 1/8, 1/4 (no kv8 keys); a_s1/8z: asymmetric,
                  range [-8, 23.875]
      b_s         s as FP8 e4m3 with scale 1/16 (clip +-448 / 16 = +-28); b_kv8: the same on s_kv8
      c / c_nofold  softmax weights as UINT8 the pvtu way: per 2048-key tile e = exp(s - m_t) times the value
                  scales over their tile maximum (c_nofold: e alone), codes at step 1/255; denominator exact
      d           a_r at 1/8 combined with c; d_r1/32: a_r at 1/32 with c; d_s1/4: a_s at 1/4 with c
  Values stay fp32 in every variant (only the scores / weights are quantized).

    EXPORT_DIR=~/Models/vq27b/export/mix25in_mixr_lr64mix MODEL=~/Models/Qwen3.8-27B \
        python scripts/m6_attn_logit_stats.py run --out DIR [--pass "kl=4,25" --pass "wiki=4096x1;stride=4"]
    python scripts/m6_attn_logit_stats.py selftest   # chunked attention == Target.attn on layer 3

Env: MODEL, EXPORT_DIR (or DEQ_DIR), THREADS (default 10), TRACE (KL trace dir, default ~/Models/vq27b/kl).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
os.environ.setdefault("MODEL", str(HOME / "Models/Qwen3.8-27B"))
os.environ.setdefault("THREADS", "10")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import dflash2_target_ref as R  # noqa: E402

TRACE = Path(os.environ.get("TRACE", str(HOME / "Models/vq27b/kl")))
WIKI = HOME / "Models/wikitext/wiki2_test.txt"
STEPS = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4)      # constant INT8 steps for r (clip fractions + a_r)
S_STEPS = (1 / 16, 1 / 8, 1 / 4)                     # constant INT8 steps for s (a_s)
ZP = 64  # asymmetric variants (*z): codes - 64 stored, range [-64, 191] steps (scores are mostly positive)
S_THR = (8, 16, 24, 28, 32)
ROW_THR = (16, 24, 32)
TILE = 2048
# fine histograms (exact counts, percentiles at bin resolution)
S_RANGE, S_BINS = 128.0, 32768      # s in [-128, 128], 1/128 per bin
R_RANGE, R_BINS = 32.0, 32768       # r in [-32, 32], 1/512 per bin
PCTS = (0.1, 1, 50, 99, 99.9, 99.99)
F8 = torch.float8_e4m3fn


def step_name(d):
    return f"1/{round(1 / d)}"


class RangeStat:
    """min / max / histogram / threshold counts of one score tensor (valid entries) and its row maxima."""

    def __init__(self, rng, bins, thr):
        self.rng, self.bins, self.thr = rng, bins, thr
        self.hist = torch.zeros(bins, dtype=torch.float64)
        self.n, self.mn, self.mx = 0, float("inf"), -float("inf")
        self.above = {t: 0 for t in thr}
        self.rowmax = []

    def add(self, v, rowmax):
        self.n += v.numel()
        self.mn, self.mx = min(self.mn, float(v.min())), max(self.mx, float(v.max()))
        a = v.abs()
        for t in self.thr:
            self.above[t] += int((a > t).sum())
        self.hist += torch.histc(v.clamp(-self.rng, self.rng), self.bins, -self.rng, self.rng).double()
        self.rowmax.append(rowmax.reshape(-1).float().numpy())

    def merge(self, o):
        self.hist += o.hist
        self.n += o.n
        self.mn, self.mx = min(self.mn, o.mn), max(self.mx, o.mx)
        for t in self.thr:
            self.above[t] += o.above[t]
        self.rowmax += o.rowmax

    def pct(self, hist, ps, signed=True):
        c = hist.cumsum(0).numpy()
        tot = c[-1]
        lo, w = (-self.rng, 2 * self.rng / self.bins) if signed else (0.0, self.rng / (self.bins // 2))
        out = {}
        for p in ps:
            k = tot * p / 100
            j = int(np.searchsorted(c, k))
            prev = c[j - 1] if j else 0.0
            frac = (k - prev) / max(c[j] - prev, 1e-30)
            out[f"p{p:g}"] = float(lo + (j + min(max(frac, 0.0), 1.0)) * w)
        return out

    def summary(self):
        h = self.hist
        half = self.bins // 2
        habs = h[half:] + h[:half].flip(0)  # |x| histogram (bins symmetric about zero)
        rm = np.concatenate(self.rowmax) if self.rowmax else np.zeros(1)
        return {
            "n": int(self.n), "min": self.mn, "max": self.mx, "max_abs": max(abs(self.mn), abs(self.mx)),
            "pct": self.pct(h, PCTS), "abs_pct": self.pct(habs, (50, 99, 99.9, 99.99, 99.999), signed=False),
            "frac_abs_gt": {str(t): self.above[t] / max(self.n, 1) for t in self.thr},
            "rowmax": {"n": int(rm.size), "median": float(np.median(rm)), "p99": float(np.percentile(rm, 99)),
                       "p99.9": float(np.percentile(rm, 99.9)), "max": float(rm.max()), "min": float(rm.min()),
                       **{f"frac_gt_{t}": float((rm > t).mean()) for t in ROW_THR}},
        }


class LayerStats:
    def __init__(self, layer):
        self.layer = layer
        self.s = RangeStat(S_RANGE, S_BINS, S_THR)
        self.r = RangeStat(R_RANGE, R_BINS, (0.5, 1, 2, 4, 8, 16))
        self.s8 = {"max": -float("inf"), "min": float("inf")}           # s_kv8 = r * scale * 128
        self.clip = {d: [0, 0] for d in STEPS}                          # r entries clipped (high, low)
        self.rowclip = {d: 0 for d in STEPS}                            # rows whose max r is clipped
        self.clipz = [0, 0]                                             # r clipped at 1/32 asymmetric
        self.nrows = 0
        self.head_max_s = torch.full((24,), -float("inf"))
        self.head_min_s = torch.full((24,), float("inf"))
        self.head_max_r = torch.full((24,), -float("inf"))
        self.kmax = []                                                  # max|k| per (KV head, token)
        self.err = {}                                                   # variant -> [sum sq err, sum sq ref]
        self.rowerr = {}                                                # variant -> per (row, head) rel errors
        self.t = 0.0

    def add_err(self, name, o, ref):
        e = (o - ref).pow(2).sum(-1)
        rs = ref.pow(2).sum(-1)
        a = self.err.setdefault(name, [0.0, 0.0])
        a[0] += float(e.sum())
        a[1] += float(rs.sum())
        self.rowerr.setdefault(name, []).append((e / rs.clamp_min(1e-30)).sqrt().reshape(-1).numpy())

    def summary(self):
        km = np.concatenate(self.kmax)
        out = {"layer": self.layer, "s": self.s.summary(), "r": self.r.summary(), "s_kv8": self.s8,
               "r_clip": {step_name(d): {"frac": (h + l) / max(self.r.n, 1), "frac_high": h / max(self.r.n, 1),
                                         "frac_low": l / max(self.r.n, 1),
                                         "frac_rows_max_clipped": self.rowclip[d] / max(self.nrows, 1),
                                         "range": [-128 * d, 127 * d]}
                          for d, (h, l) in self.clip.items()},
               "r_clip_1/32z": {"frac_high": self.clipz[0] / max(self.r.n, 1), "frac_low": self.clipz[1] / max(self.r.n, 1),
                                "range": [(-128 + ZP) / 32, (127 + ZP) / 32]},
               "head_max_s": self.head_max_s.tolist(), "head_min_s": self.head_min_s.tolist(),
               "head_max_r": self.head_max_r.tolist(),
               "key_absmax": {"median": float(np.median(km)), "p99": float(np.percentile(km, 99)),
                              "max": float(km.max())},
               "err": {}, "time_s": self.t}
        for k, (e, r) in self.err.items():
            re = np.concatenate(self.rowerr[k])
            out["err"][k] = {"rel_rms": float(np.sqrt(e / r)), "row_median": float(np.median(re)),
                             "row_p99": float(np.percentile(re, 99)), "row_max": float(re.max()),
                             "rows": int(re.size)}
        return out


def kv8(x):
    """quantize_values (qwen38_kv_cache.py) in torch: FP16 values, scale max|x| / 127 per (head, token) stored as
    FP16, codes rint(x / scale) clipped to [-127, 127]. x (nkv, L, hd) -> codes (float), scales (nkv, L)."""
    xh = x.half().float()
    sc = torch.clamp_min(xh.abs().amax(-1) / 127, 1e-6).half().float()
    return torch.clamp(torch.round(xh / sc[..., None]), -127, 127), sc


def out_softmax(sx, valid, V):
    """softmax over valid keys @ V. sx (nkv, grp, R, L), valid (R, L), V (nkv, L, hd) -> (nkv, grp, R, hd)."""
    return torch.softmax(sx.masked_fill(~valid, float("-inf")), -1) @ V[:, None]


def out_pvtu(sx, valid, V, vs128, fold=True):
    """pvtu: per key tile e = exp(s - m_t) (times vs / max(vs) of the tile per head when fold), UINT8 at 1/255;
    the tile corrections exp(m_t - m) and max(vs) multiply the small PV output; the denominator is exact."""
    sm = sx.masked_fill(~valid, float("-inf"))
    m = sm.amax(-1, keepdim=True)
    L = sx.shape[-1]
    num = den = 0
    for a in range(0, L, TILE):
        b = min(a + TILE, L)
        st = sm[..., a:b]
        m_t = st.amax(-1, keepdim=True)
        m_t = torch.where(torch.isfinite(m_t), m_t, m)                  # rows with no valid key in the tile
        e = torch.exp(st - m_t)                                         # masked -> 0
        w_t = torch.exp(m_t - m)
        den = den + e.sum(-1, keepdim=True) * w_t
        if fold:
            vs = vs128[:, a:b][:, None, None]                           # (nkv, 1, 1, Lt)
            vmax = torch.clamp_min(vs.amax(-1, keepdim=True), 1e-4)
            pq = torch.round(e * (vs / vmax) * 255).clamp(0, 255) / 255
            num = num + (pq @ (V[:, a:b] / vs128[:, a:b, None])[:, None]) * (w_t * vmax)
        else:
            pq = torch.round(e * 255).clamp(0, 255) / 255
            num = num + (pq @ V[:, a:b][:, None]) * w_t
    return num / den


def q_int8(x, d):
    return torch.clamp(torch.round(x / d), -128, 127) * d


def q_int8z(x, d, z=ZP):
    return (torch.clamp(torch.round(x / d) - z, -128, 127) + z) * d


def q_fp8(x, scale=1 / 16):
    return torch.clamp(x / scale, -448, 448).to(F8).float() * scale


class StatTarget(R.Target):
    """Target whose full attention runs in query-row chunks (identical math) and records statistics."""
    qchunk, sim_stride, stats = 512, 2, {}

    def attn(self, i, w, h, jobs):
        c = self.cfg
        nh, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        qg_all = (h @ w["self_attn.q_proj.weight"].T).view(-1, nh, 2 * hd)
        k_all = R.rms_zc((h @ w["self_attn.k_proj.weight"].T).view(-1, nkv, hd), w["self_attn.k_norm.weight"], self.eps)
        v_all = (h @ w["self_attn.v_proj.weight"].T).view(-1, nkv, hd)
        q_all = R.rms_zc(qg_all[..., :hd], w["self_attn.q_norm.weight"], self.eps)
        gate_all = qg_all[..., hd:].reshape(qg_all.shape[0], -1)
        outs = []
        t0 = time.time()
        ls = self.stats.setdefault(i, LayerStats(i)) if self.stats is not None else None
        for st, r0, T, _ in jobs:
            p = st.p
            f = torch.arange(p, p + T, dtype=torch.float64)[:, None] * self.inv[None]
            cos, sin = torch.cat([f, f], 1).cos().float()[:, None], torch.cat([f, f], 1).sin().float()[:, None]

            def rope(t):
                r, rest = t[..., :self.rot], t[..., self.rot:]
                half = self.rot // 2
                return torch.cat([r * cos + torch.cat([-r[..., half:], r[..., :half]], -1) * sin, rest], -1)
            q, k = rope(q_all[r0:r0 + T]), rope(k_all[r0:r0 + T])                      # (T, heads, hd)
            st.K[i][:, p:p + T], st.V[i][:, p:p + T] = k.transpose(0, 1), v_all[r0:r0 + T].transpose(0, 1)
            K, V = st.K[i][:, :p + T], st.V[i][:, :p + T]                                # (nkv, L, hd)
            if ls is not None:
                kc, ks = kv8(K)
                _, vsc = kv8(V)
                ks128, vs128 = ks * 128, vsc * 128
                ls.kmax.append((ks * 127).reshape(-1).numpy())
            qh = q.transpose(0, 1).reshape(nkv, nh // nkv, T, hd)                       # head = kv * grp + g
            o = torch.empty(nkv, nh // nkv, T, hd)
            for a in range(0, T, self.qchunk):
                b = min(a + self.qchunk, T)
                Lk = p + b
                sc = (qh[:, :, a:b] @ K[:, None, :Lk].transpose(-1, -2)) / hd ** 0.5      # (nkv, grp, Tc, Lk)
                valid = torch.arange(Lk)[None] <= torch.arange(p + a, p + b)[:, None]
                if ls is not None:
                    self.record(ls, sc, valid, qh[:, :, a:b], kc[:, :Lk], ks128[:, :Lk], vs128[:, :Lk],
                                V[:, :Lk], p + a)
                o[:, :, a:b] = torch.softmax(sc.masked_fill(~valid, float("-inf")), -1) @ V[:, None, :Lk]
                del sc
            o = o.reshape(nh, T, hd).transpose(0, 1).reshape(T, -1)
            outs.append(o * torch.sigmoid(gate_all[r0:r0 + T]))
        if ls is not None:
            ls.t += time.time() - t0
        return torch.cat(outs) @ w["self_attn.o_proj.weight"].T

    def record(self, ls, sc, valid, qc, kc, ks128, vs128, V, pos0):
        hd = qc.shape[-1]
        nkv, grp, Tc, Lk = sc.shape
        neg = float("-inf")
        # ---- ranges over all valid entries
        sm = sc.masked_fill(~valid, neg)
        rm_s = sm.amax(-1)                                                          # (nkv, grp, Tc)
        ls.s.add(sc[:, :, valid], rm_s)
        ls.head_max_s = torch.maximum(ls.head_max_s, rm_s.reshape(nkv * grp, -1).amax(-1))
        ls.head_min_s = torch.minimum(ls.head_min_s, sc.masked_fill(~valid, float("inf")).reshape(nkv * grp, -1).amin(-1))
        del sm
        r = (qc @ (kc / 128)[:, None].transpose(-1, -2)) / hd ** 0.5                 # kv8 raw scores
        rv = r[:, :, valid]
        rm_r = r.masked_fill(~valid, neg).amax(-1)
        ls.r.add(rv, rm_r)
        ls.head_max_r = torch.maximum(ls.head_max_r, rm_r.reshape(nkv * grp, -1).amax(-1))
        ls.clipz[0] += int((rv >= (127 + ZP + 0.5) / 32).sum())
        ls.clipz[1] += int((rv < (-128 + ZP - 0.5) / 32).sum())
        for d in STEPS:
            ls.clip[d][0] += int((rv >= 127.5 * d).sum())
            ls.clip[d][1] += int((rv < -128.5 * d).sum())
            ls.rowclip[d] += int((rm_r >= 127.5 * d).sum())
        ls.nrows += rm_r.numel()
        del rv
        s8 = r * ks128[:, None, None]
        s8v = s8[:, :, valid]
        ls.s8["max"], ls.s8["min"] = max(ls.s8["max"], float(s8v.max())), min(ls.s8["min"], float(s8v.min()))
        del s8v
        # ---- simulated outputs on a subsample of query rows
        sel = torch.arange(Tc)[(torch.arange(Tc) + pos0) % self.sim_stride == 0]
        if not len(sel):
            return
        vs = valid[sel]
        s_, r_, s8_ = sc[:, :, sel], r[:, :, sel], s8[:, :, sel]
        del r, s8
        ref = out_softmax(s_, vs, V)
        ks = ks128[:, None, None]
        ls.add_err("kv8k", out_softmax(s8_, vs, V), ref)
        for d in STEPS:
            ls.add_err(f"a_r{step_name(d)}", out_softmax(q_int8(r_, d) * ks, vs, V), ref)
        ls.add_err("a_r1/32z", out_softmax(q_int8z(r_, 1 / 32) * ks, vs, V), ref)
        for d in S_STEPS:
            ls.add_err(f"a_s{step_name(d)}", out_softmax(q_int8(s_, d), vs, V), ref)
        ls.add_err("a_s1/8z", out_softmax(q_int8z(s_, 1 / 8), vs, V), ref)
        ls.add_err("b_s", out_softmax(q_fp8(s_), vs, V), ref)
        ls.add_err("b_kv8", out_softmax(q_fp8(s8_), vs, V), ref)
        ls.add_err("c", out_pvtu(s_, vs, V, vs128, fold=True), ref)
        ls.add_err("c_nofold", out_pvtu(s_, vs, V, vs128, fold=False), ref)
        ls.add_err("d", out_pvtu(q_int8(r_, 1 / 8) * ks, vs, V, vs128, fold=True), ref)
        ls.add_err("d_r1/32", out_pvtu(q_int8(r_, 1 / 32) * ks, vs, V, vs128, fold=True), ref)
        ls.add_err("d_s1/4", out_pvtu(q_int8(s_, 1 / 4), vs, V, vs128, fold=True), ref)


def make_target(n_layers):
    tgt = StatTarget.__new__(StatTarget)
    tgt.cfg, tgt.W, tgt.n_layers = R.text_cfg(), R.Weights(), n_layers
    c = tgt.cfg
    tgt.eps = c["rms_norm_eps"]
    tgt.rot = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"])
    tgt.inv = 1.0 / c["rope_parameters"]["rope_theta"] ** (torch.arange(0, tgt.rot, 2, dtype=torch.float64) / tgt.rot)
    tgt.stats = {}
    return tgt


DEFAULT_PASSES = (  # KL trace (chat, thinking on: code, math, agentic | science, writing, multilingual, design),
    # then wikitext-2 test at 4K and 16K (one pass streams all 64 layers; passes keep the activations small)
    "kl=1,4,8,10,13,16,24,25,26,27,50,51,52,54,56",
    "kl=29,31,33,35,38,40,43,47,55,58,59,60,61,63",
    "wiki=4096x2",
    "wiki=16384x1;qchunk=256;stride=8",
)


def parse_pass(spec):
    out = {}
    for item in spec.split(";"):
        k, v = item.split("=")
        out[k.strip()] = v.strip()
    return out


class Wiki:
    """Contiguous wikitext-2 test tokens, consumed in order across passes."""

    def __init__(self):
        self.ids, self.pos = None, 0

    def take(self, n):
        if self.ids is None or self.pos + n > len(self.ids):
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(R.MODEL))
            self.ids = tok(WIKI.read_text()[:8 * (self.pos + n) + 100000], add_special_tokens=False)["input_ids"]
        out = self.ids[self.pos:self.pos + n]
        self.pos += n
        return out


def load_sequences(spec, wiki):
    seqs = []
    if spec.get("kl"):
        d = np.load(TRACE / "trace.npz")
        off = np.concatenate([[0], np.cumsum(d["lengths"])])
        for j in (int(v) for v in spec["kl"].split(",")):
            seqs.append((f"kl{j}", d["ids"][off[j]:off[j + 1]].astype(np.int64).tolist()))
    if spec.get("wiki"):
        n, k = (int(v) for v in spec["wiki"].split("x"))
        for _ in range(k):
            a = wiki.pos
            seqs.append((f"wiki@{a}", wiki.take(n)))
    return seqs


def embed_rows(W, ids):
    emb = W.get("model.language_model.embed_tokens.weight")
    x = emb[torch.as_tensor(ids, dtype=torch.long)].float()
    del emb
    return x


def merge_layer(dst, o):
    dst.s.merge(o.s)
    dst.r.merge(o.r)
    dst.s8["max"], dst.s8["min"] = max(dst.s8["max"], o.s8["max"]), min(dst.s8["min"], o.s8["min"])
    for d in STEPS:
        dst.clip[d][0] += o.clip[d][0]
        dst.clip[d][1] += o.clip[d][1]
        dst.rowclip[d] += o.rowclip[d]
    dst.clipz = [dst.clipz[0] + o.clipz[0], dst.clipz[1] + o.clipz[1]]
    dst.nrows += o.nrows
    dst.head_max_s = torch.maximum(dst.head_max_s, o.head_max_s)
    dst.head_min_s = torch.minimum(dst.head_min_s, o.head_min_s)
    dst.head_max_r = torch.maximum(dst.head_max_r, o.head_max_r)
    dst.kmax += o.kmax
    for k, (e, r) in o.err.items():
        a = dst.err.setdefault(k, [0.0, 0.0])
        a[0] += e
        a[1] += r
        dst.rowerr.setdefault(k, []).extend(o.rowerr[k])
    dst.t += o.t


def rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30  # bytes on macOS


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = (f"export {R.EXPORT_DIR}" if R.EXPORT_DIR else f"dequantized {R.DEQ_DIR}" if R.DEQ_DIR else
           f"bf16 {R.MODEL}")
    specs = [parse_pass(p) for p in (args.passes or DEFAULT_PASSES)]
    wiki, done = Wiki(), []   # done: (spec, sequences, {layer: LayerStats}) per finished pass
    t_all = time.time()
    tgt = make_target(args.n_layers)
    print(f"weights: {src}; threads {torch.get_num_threads()}; {len(specs)} passes", flush=True)
    for n_pass, spec in enumerate(specs):
        StatTarget.qchunk = int(spec.get("qchunk", args.qchunk))
        StatTarget.sim_stride = int(spec.get("stride", args.sim_stride))
        seqs = load_sequences(spec, wiki)
        tgt.stats = {}
        cur = (spec, [(n, len(s)) for n, s in seqs], tgt.stats)
        print(f"pass {n_pass} {spec}: sequences {cur[1]} = {sum(len(s) for _, s in seqs)} tokens", flush=True)
        t_pass = time.time()
        x = embed_rows(tgt.W, [t for _, s in seqs for t in s])
        jobs, r0 = [], 0
        for _, s in seqs:
            R.CTX_MAX = len(s)  # K / V state sized per sequence
            jobs.append((R.SeqState(tgt.cfg), r0, len(s), True))
            r0 += len(s)
        del seqs
        for i in range(args.n_layers):
            t = time.time()
            w = tgt.W.layer(i)
            tl = time.time() - t
            h = R.rms_zc(x, w["input_layernorm.weight"], tgt.eps)
            kind = tgt.cfg["layer_types"][i]
            x = x + (tgt.gdn(i, w, h, jobs) if kind == "linear_attention" else tgt.attn(i, w, h, jobs))
            x = x + tgt.mlp(w, R.rms_zc(x, w["post_attention_layernorm.weight"], tgt.eps))
            del w, h
            for st, *_ in jobs:  # single pass: this layer's state is not needed again
                for dct in (st.S, st.conv, st.K, st.V):
                    dct.pop(i, None)
            msg = f"pass {n_pass} layer {i:2d} {kind[:6]}: {time.time() - t:6.1f}s (load {tl:.1f}s)"
            if i in tgt.stats:
                ls = tgt.stats[i]
                msg += (f" stats {ls.t:.1f}s | max|s| {max(abs(ls.s.mn), ls.s.mx):.2f} max|r| "
                        f"{max(abs(ls.r.mn), ls.r.mx):.3f} | "
                        + " ".join(f"{k} {np.sqrt(e / r):.2e}" for k, (e, r) in ls.err.items()
                                   if k in ("kv8k", "a_r1/32", "a_r1/8", "a_s1/4", "b_s", "c", "d")))
                save(done + [cur], src, out / args.json, time.time() - t_all, final=False)
            print(msg + f" | maxrss {rss_gb():.1f} GB | total {time.time() - t_all:.0f}s", flush=True)
        del x, jobs
        done.append(cur)
        print(f"pass {n_pass} done in {time.time() - t_pass:.0f}s", flush=True)
    save(done, src, out / args.json, time.time() - t_all, final=True)
    print(f"done in {time.time() - t_all:.0f}s -> {out / args.json}", flush=True)


def compact(sm):
    return {"max_abs_s": sm["s"]["max_abs"], "p99.99_abs_s": sm["s"]["abs_pct"]["p99.99"],
            "rowmax_p99": sm["s"]["rowmax"]["p99"], "rowmax_max": sm["s"]["rowmax"]["max"],
            "frac_abs_s_gt16": sm["s"]["frac_abs_gt"]["16"], "max_abs_r": sm["r"]["max_abs"],
            "p99.99_abs_r": sm["r"]["abs_pct"]["p99.99"], "err": {k: v["rel_rms"] for k, v in sm["err"].items()}}


def save(passes, src, path, elapsed, final):
    merged = {}
    for _, _, stats in passes:
        for i, ls in stats.items():
            merge_layer(merged.setdefault(i, LayerStats(i)), ls)
    layers = [merged[i].summary() for i in sorted(merged)]
    tot_s, tot_r = RangeStat(S_RANGE, S_BINS, S_THR), RangeStat(R_RANGE, R_BINS, (0.5, 1, 2, 4, 8, 16))
    err = {}
    for i in sorted(merged):
        ls = merged[i]
        tot_s.merge(ls.s)
        tot_r.merge(ls.r)
        for k, (e, r) in ls.err.items():
            a = err.setdefault(k, [0.0, 0.0])
            a[0] += e
            a[1] += r
    overall = {"s": tot_s.summary(), "r": tot_r.summary(),
               "err_rel_rms": {k: float(np.sqrt(e / r)) for k, (e, r) in err.items()}} if layers else {}
    per_pass = [{"spec": spec, "sequences": [{"name": n, "tokens": t} for n, t in seqs],
                 "layers": {str(i): compact(stats[i].summary()) for i in sorted(stats)}}
                for spec, seqs, stats in passes]
    res = {"final": final, "weights": src, "elapsed_s": elapsed, "threads": torch.get_num_threads(),
           "tokens": sum(t for _, seqs, _ in passes for _, t in seqs), "tile": TILE, "zero_point": ZP,
           "layers": layers, "overall": overall, "passes": per_pass}
    path.write_text(json.dumps(res, indent=1))


def selftest(args):
    """Chunked attention with statistics == Target.attn (layer 3, the first KL sequence, inputs = embeddings)."""
    tgt = make_target(64)
    StatTarget.qchunk, StatTarget.sim_stride = 96, 3
    seqs = load_sequences({"kl": "0"}, None)
    ids = seqs[0][1][:300]
    w = tgt.W.layer(3)
    x = embed_rows(tgt.W, ids)
    h = R.rms_zc(x, w["input_layernorm.weight"], tgt.eps)
    R.CTX_MAX = len(ids)
    st1, st2 = R.SeqState(tgt.cfg), R.SeqState(tgt.cfg)
    t = time.time()
    a = tgt.attn(3, w, h, [(st1, 0, len(ids), True)])
    t1 = time.time() - t
    t = time.time()
    b = R.Target.attn(tgt, 3, w, h, [(st2, 0, len(ids), True)])
    t2 = time.time() - t
    err = float((a - b).norm() / b.norm())
    print(f"selftest: rel diff chunked vs Target.attn {err:.2e} (chunked+stats {t1:.1f}s, plain {t2:.1f}s)")
    print(json.dumps(tgt.stats[3].summary()["err"], indent=1))
    print("SELFTEST", "PASS" if err < 1e-5 else "FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "selftest"))
    ap.add_argument("--out", default="/Volumes/SSD4TB/anemll-forge-research/logit_stats")
    ap.add_argument("--json", default="attn_logit_stats.json")
    ap.add_argument("--pass", dest="passes", action="append",
                    help='one pass (repeatable), e.g. "kl=4,25" or "wiki=4096x2;qchunk=256;stride=4"')
    ap.add_argument("--n-layers", type=int, default=64)
    ap.add_argument("--qchunk", type=int, default=512)
    ap.add_argument("--sim-stride", type=int, default=2, help="simulated outputs on every n-th query row")
    a = ap.parse_args()
    {"run": run, "selftest": selftest}[a.cmd](a)
