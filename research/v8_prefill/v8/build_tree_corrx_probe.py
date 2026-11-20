"""D4 (20 Nov 2026): cheap exact tree conv, probe (SCRATCH build, deleted after the test). One target group (layers 0-3:
three DeltaNet layers + the attention layer, the chunked FFN, the served weight constants) at 8 rows / 8K of KV, one
package PER FORM (so a form that fails the ANE compile names itself), each with two entries on the same constants:
  full8_8k   the served chain verify entry (B.Entry)
  tree8_8k   the tree entry (anc + ctap inputs, build_tree_probe.TreeEntry) with that form
Forms (conv correction + cumulative decay):
  corr     the served conv + sum_t w_t (ctap_t @ rows - rows shifted by t)      cum = g @ anc            (A8's build)
  corrc    corr's conv                                                        cum = g @ tri + g @ (anc - tri)
  corrd    the served conv + sum_t w_t (D_t @ rows), D = ctap - the chain shift  (built on the device), exact cum
  corrx    the served conv + [rows w_0 .. rows w_3] (cdim x 44) @ D (44 x 8), channel-major, exact cum
  corrx2   the served conv + sum_t w_t (rows_cm @ D_t^T) (4 matmuls, K = 11), channel-major, exact cum
Every correction is exactly + 0 on a row whose taps / ancestors are the chain's (D's row is zero).
    env as v8/build_dec8_one.sh  BUILDPY build_tree_corrx_probe.py OUT_DIR [FORMS]"""
import gc, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v8_ffn as V  # noqa: E402  (imports qwen38_coreai_build as B with this environment)
import chunkmlp as CM  # noqa: E402
import build_tree_probe as TPB  # noqa: E402  (patches B.tri / B.GDNW.qkv_heads only while a TreeEntry traces)

B, M = V.B, V.M
FORM = {"conv": None, "cum": "anc"}
_qkv0 = B.GDNW.qkv_heads          # build_tree_probe's (the served form when no tree traces)
_verify0 = B.GDNW.verify


def _shift(T, dtype):
    """the chain's taps as a CONSTANT (numpy-built like B.tri; in-place indexing inside forward traces as ops the ANE
    does not run: the first probe's corrd / corrx / corrx2 packages ran off the ANE, ~1,990 ms per call)"""
    s = np.zeros((4, T, T + 3), np.float16)
    for t in range(4):
        s[t, np.arange(T), np.arange(T) + t] = 1
    return torch.from_numpy(s).to(dtype)


def qkv_heads_d4(self, rows, T):
    f = FORM["conv"]
    if f is None or TPB.TREE["ctap"] is None or T != TPB.TREE["T"]:
        return _qkv0(self, rows, T)
    blend = f.endswith("b")
    f = f[:-1] if blend else f
    cdim = rows.shape[1]
    ctap = TPB.TREE["ctap"]
    rcm = rows.transpose(0, 1)                                                          # (cdim, T + 3)
    cv4 = F.conv2d(rows.transpose(0, 1).reshape(1, cdim, 1, T + 3), self.cw.transpose(0, 1).reshape(cdim, 1, 1, 4), groups=cdim)
    conv = cv4.reshape(cdim, T)
    d3 = ctap - _shift(T, rows.dtype)                                                   # (4, T, T + 3): 0 on chain rows
    if f == "corr":
        taps = ctap @ rows
        shifted = torch.stack([rows[t:t + T] for t in range(4)], 0)
        corr = ((taps - shifted) * self.cw.reshape(4, 1, -1)).sum(0).transpose(0, 1)
    elif f == "corrd":
        corr = ((d3 @ rows) * self.cw.reshape(4, 1, -1)).sum(0).transpose(0, 1)
    elif f == "corrx":
        d = d3.permute(0, 2, 1).reshape(4 * (T + 3), T)
        rw = torch.cat([rcm * self.cw[t].reshape(cdim, 1) for t in range(4)], 1)
        corr = rw @ d
    elif f == "corrx2":
        corr = sum((rcm @ d3[t].transpose(0, 1)) * self.cw[t].reshape(cdim, 1) for t in range(4))
    else:
        raise ValueError(f)
    if blend:   # chain rows: the served silu(conv) itself (+ 0 * x); tree rows: silu(conv + correction)
        served = B.silu(cv4).reshape(cdim, T)
        m = torch.clamp(d3.abs().sum(0).sum(1), max=1.0).reshape(1, T)                    # 1 on rows with a non-chain tap
        conv = served + m * (B.silu(conv + corr) - served)
    else:
        conv = B.silu(conv + corr)
    kd, nk, nv, dk, dv = B.kd, B.nk, B.nv, B.dk, B.dv
    qq, kk, vv = conv[:kd], conv[kd:2 * kd], conv[2 * kd:]

    def heads(t):
        t = t.reshape(nk, dk, T).permute(0, 2, 1)
        return t.reshape(nk, 1, T, dk).repeat(1, nv // nk, 1, 1).reshape(nv, T, dk)

    def l2n(t, s_):
        return t * torch.rsqrt((t * t).sum(-1, keepdim=True) + 1e-6) * s_
    vh = vv.reshape(nv, dv, T).permute(0, 2, 1)
    vh = vh * B.GDN_SV if B.GDN_SV != 1 else vh
    return l2n(heads(qq), dk ** -0.5 * B.GDN_SQ), l2n(heads(kk), 1.0), vh


def verify_d4(self, h, conv_rows, conv_sel, rec, pend, commit, commit_last, T):
    if FORM["cum"] != "exact" or TPB.TREE["anc"] is None or T != TPB.TREE["T"]:
        return _verify0(self, h, conv_rows, conv_sel, rec, pend, commit, commit_last, T)
    nv, dv = B.nv, B.dv
    qkv, z, b, a = self.proj(h, T)
    rows = torch.cat([conv_sel @ conv_rows, qkv.transpose(0, 1)], 0)
    qh, kh, vh = self.qkv_heads(rows, T)
    beta, g = torch.sigmoid(b), B.softplus(a + self.dt) * self.neg_a
    s1 = self.commit_pending(rec, pend, commit, commit_last)
    l_inc, l_str = B.tri(T, False), B.tri(T, True)                                      # anc, anc minus its diagonal
    tri_c = TPB._tri0(T, False)
    g3 = g.reshape(nv, 1, T)
    cum = (g3 @ tri_c.T + g3 @ (l_inc - tri_c).T).reshape(nv, T, 1)
    pair = torch.exp(torch.clamp(cum - cum.reshape(nv, 1, T), max=0)) * l_inc
    kb, vb = kh * beta, vh * beta
    n = (kb @ kh.transpose(1, 2)) * (pair * l_str)
    x = B.tri_solve(n, torch.cat([vb, kb * torch.exp(cum)], -1), T)
    u, wk = x[..., :dv], x[..., dv:]
    crow = torch.cat([cum.reshape(nv, 1, T), torch.zeros(nv, 1, dv - T, dtype=cum.dtype)], 2)
    pend_out = torch.cat([kh, u, wk, crow], 1)
    vn = u - wk @ s1
    o = (qh * torch.exp(cum)) @ s1 + ((qh @ kh.transpose(1, 2)) * pair) @ vn
    return self.finish(o, z, T), rows, s1, pend_out


B.GDNW.qkv_heads = qkv_heads_d4
B.GDNW.verify = verify_d4
FORMS = {"corr": ("corr", "anc"), "corrc": ("corr", "exact"), "corrd": ("corrd", "exact"), "corrx": ("corrx", "exact"),
         "corrx2": ("corrx2", "exact"),
         # round 2 (cum = g @ anc: round 1 showed the cum row identical; the 1-ulp differences are in kh, from silu(conv + 0))
         "corrd0": ("corrd", "anc"), "corrx0": ("corrx", "anc"), "corrb": ("corrb", "anc"), "corrdb": ("corrdb", "anc"),
         "corrxb": ("corrxb", "anc")}


class TreeF(TPB.TreeEntry):
    form = None

    def forward(self, *args):
        FORM["conv"], FORM["cum"] = FORMS[self.form]
        try:
            return super().forward(*args)
        finally:
            FORM["conv"], FORM["cum"] = None, "anc"


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    forms = (sys.argv[2] if len(sys.argv) > 2 else "corr,corrc,corrd,corrx,corrx2").split(",")
    assert V.SPLIT, "set V8_SPLIT (e.g. 8192,9216)"
    B.P = 8
    t0 = time.time()
    ck = M.Checkpoint()
    B.KNOWN_LUTS.clear()
    W = {}
    for i in range(4):
        W.update(B.layer_arrays(ck, i))
    mods = nn.ModuleList(B.LayerW(W, i) for i in range(4)).eval().to(torch.float16)
    for m in mods:
        m.cmlp = CM.ChunkMLP(W, m, V.SPLIT).eval().to(torch.float16)
        object.__setattr__(m, "mlp", m.cmlp)
        m.gate = m.up = m.down = m.rmid = None
    del W
    gc.collect()
    mode = B.KV_CACHE_DTYPE
    for f in forms:
        d = out / f; d.mkdir(parents=True, exist_ok=True)
        B.OUT = d
        e8 = B.Entry(mods, 8192, 8, kv_cache_dtype=mode)
        te = TreeF(mods, 8192, 8, kv_cache_dtype=mode); te.form = f
        entries = [("full8_8k", e8, e8.input_names(), e8.output_names()), ("tree8_8k", te, te.input_names(), te.output_names())]
        fname = f"d4_{f}_L00-03.aimodel"
        mb = B.save_program(entries, d / fname)
        man = {"version": "tree_corrx_probe", "kind": "scratch", "form": f,
               "head": {"file": "/Volumes/SSD4TB/anemll-forge-research/coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8/head_T8.aimodel"},
               "chunks": [{"file": fname, "layers": [0, 3], "P": 8, "entries": [x[0] for x in entries], "mb": round(mb)}]}
        (d / "manifest.json").write_text(json.dumps(man, indent=1))
        print(f"{f}: {fname}, {mb:.0f} MB ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
