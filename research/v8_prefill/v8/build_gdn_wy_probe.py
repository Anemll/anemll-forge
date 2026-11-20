"""Can the DeltaNet verify be cheap at 16 / 32 rows? (20 Nov 2026, tree study; SCRATCH build, deleted after the test.)
The decomposition probe (build_tree_decomp.py, results/20nov/decomp) put the 16 / 32-row cost in the DeltaNet mixers
(3 layers: 1.76 / 2.23 / 4.04 ms at 8 / 16 / 32 rows; the FFN 2.98 / 3.04 / 3.10). Their T x T / T^3 part is the UT
transform's (I + N)^-1: inv_unit_lower, log2(T) - 1 rounds of broadcast multiply + reduce over (nv, T, T, T). One
package per TREE_P = 16 | 32 with the 3 DeltaNet mixers of layers 0-2 (build_tree_decomp.Part "gdn"), the solve as:
  gdn<T>_inv   served (GDN_FAST: inv_unit_lower over T rows, one matmul with the right-hand side)
  gdn<T>_none  x = rhs (no solve: the floor of everything else; WRONG output, timing only)
  gdn<T>_blk   block inverse: the 8 x 8 diagonal blocks by inv_unit_lower (batched), the off-diagonal blocks by
               X_ij = -X_ii sum_k N_ik X_kj (broadcast multiply + reduce), then one matmul (exact in exact arithmetic;
               rows in topological order, so a tree's N is block lower triangular too)
  gdn<T>_bfs   block forward substitution on the right-hand side: x_i = X_ii (r_i - sum_{k<i} N_ik x_k)
  gdn<T>_tc    tree form as 19 Nov (anc + ctap inputs; conv taps = ctap @ rows, (T, cdim) layout), served solve
  gdn<T>_tcT   tree form, the conv taps channel-major: [history^T | qkv] (cdim, T + 3) @ ctapT (T + 3, 4T), times the
               conv weights, summed over the 4 taps -> (cdim, T) (no transpose back for the conv)
  gdn<T>_tcTb  tcT with the block inverse (T >= 16)
  gdn<T>_tcC   tree form, the served conv kernel on the rows as they lie + a correction for the rows whose taps are not
               the 3 rows before them: sum_t w_t (ctap_t @ rows - rows shifted by t); the correction is exactly 0 on
               chain-tap rows, so a chain-shaped path computes bit for bit what the chain entry does (if the gather is exact)
TREE_P = 8 builds the inv / none / tc / tcT / tcC entries only.
    env as v8/build_dec8_one.sh  BUILDPY build_gdn_wy_probe.py OUT_DIR   (TREE_P = 16 | 32)"""
import gc, json, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v8_ffn as V  # noqa: E402
import chunkmlp as CM  # noqa: E402
import build_tree_decomp as TD  # noqa: E402

B, M = V.B, V.M
TP = int(os.environ.get("TREE_P", "16"))
SOLVE = {"mode": "inv"}
_tri_solve0 = B.tri_solve


def bmm(a, b):   # (.., m, k) x (.., k, n) as broadcast multiply + reduce (what ANEC takes between computed tensors)
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)


def tri_solve_probe(n, rhs, rows):
    mode = SOLVE["mode"]
    if mode == "inv" or rows <= 8:
        return _tri_solve0(n, rhs, rows)
    if mode == "none":
        return rhs
    nb = rows // 8
    blk = lambda t, i, j: t[..., 8 * i:8 * i + 8, 8 * j:8 * j + 8]  # noqa: E731
    D = [B.inv_unit_lower(blk(n, i, i), 8) for i in range(nb)]
    if mode == "blk":
        X = {}
        for i in range(nb):
            X[i, i] = D[i]
            for j in range(i - 1, -1, -1):
                acc = None
                for k in range(j, i):
                    t = bmm(blk(n, i, k), X[k, j])
                    acc = t if acc is None else acc + t
                X[i, j] = -bmm(D[i], acc)
        z = torch.zeros_like(D[0])
        rowsX = [torch.cat([X[i, j] if j <= i else z for j in range(nb)], -1) for i in range(nb)]
        return torch.cat(rowsX, -2) @ rhs
    # bfs: block forward substitution on the right-hand side
    xs = []
    for i in range(nb):
        r = rhs[..., 8 * i:8 * i + 8, :]
        for k in range(i):
            r = r - blk(n, i, k) @ xs[k]
        xs.append(D[i] @ r)
    return torch.cat(xs, -2)


B.tri_solve = tri_solve_probe
import build_tree_probe as TPB  # noqa: E402  (tri / qkv_heads take anc / ctap while TPB.TREE is set)
import torch.nn.functional as F  # noqa: E402
CM_T = {"on": False, "corr": False}
_qkv_heads_tree = B.GDNW.qkv_heads   # build_tree_probe's (ctap) form while TPB.TREE is set


def qkv_heads_corr(self, rows, T):
    if not CM_T["corr"]:
        return _qkv_heads_tree(self, rows, T)
    cdim = rows.shape[1]
    conv = F.conv2d(rows.transpose(0, 1).reshape(1, cdim, 1, T + 3), self.cw.transpose(0, 1).reshape(cdim, 1, 1, 4), groups=cdim)
    conv = conv.reshape(cdim, T)                                                     # the served conv (chain taps)
    taps = TPB.TREE["ctap"] @ rows                                                   # (4, T, cdim): the tree's taps
    shifted = torch.stack([rows[t:t + T] for t in range(4)], 0)                      # (4, T, cdim): the chain's taps
    corr = ((taps - shifted) * self.cw.reshape(4, 1, -1)).sum(0).transpose(0, 1)    # (cdim, T), 0 on chain-tap rows
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


B.GDNW.qkv_heads = qkv_heads_corr
_verify0 = B.GDNW.verify


def verify_cm(self, h, conv_rows, conv_sel, rec, pend, commit, commit_last, T):
    """GDNW.verify with the tree conv taps channel-major (CM_T on; TPB.TREE holds anc / ctapT (T + 3, 4T))."""
    if not CM_T["on"]:
        return _verify0(self, h, conv_rows, conv_sel, rec, pend, commit, commit_last, T)
    qkv, z, b, a = self.proj(h, T)
    hist = conv_sel @ conv_rows                                                        # (3, cdim)
    rows = torch.cat([hist, qkv.transpose(0, 1)], 0)                                   # (T + 3, cdim): the state output
    rowsT = torch.cat([hist.transpose(0, 1), qkv], 1)                                  # (cdim, T + 3)
    taps = rowsT @ TPB.TREE["ctap"]                                                    # (cdim, 4T): tap t of row i at t * T + i
    cdim = qkv.shape[0]
    conv = (taps.reshape(cdim, 4, T) * self.cw.transpose(0, 1).reshape(cdim, 4, 1)).sum(1)
    conv = B.silu(conv)                                                                # (cdim, T)
    kd, nk, nv, dk, dv = B.kd, B.nk, B.nv, B.dk, B.dv
    qq, kk, vv = conv[:kd], conv[kd:2 * kd], conv[2 * kd:]

    def heads(t):
        t = t.reshape(nk, dk, T).permute(0, 2, 1)
        return t.reshape(nk, 1, T, dk).repeat(1, nv // nk, 1, 1).reshape(nv, T, dk)

    def l2n(t, s_):
        return t * torch.rsqrt((t * t).sum(-1, keepdim=True) + 1e-6) * s_
    vh = vv.reshape(nv, dv, T).permute(0, 2, 1)
    vh = vh * B.GDN_SV if B.GDN_SV != 1 else vh
    qh, kh = l2n(heads(qq), dk ** -0.5 * B.GDN_SQ), l2n(heads(kk), 1.0)
    beta, g = torch.sigmoid(b), B.softplus(a + self.dt) * self.neg_a
    s1 = self.commit_pending(rec, pend, commit, commit_last)
    l_inc, l_str = B.tri(T, False), B.tri(T, True)
    cum = (g.reshape(nv, 1, T) @ l_inc.T).reshape(nv, T, 1)
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


B.GDNW.verify = verify_cm


class GdnPart(TD.Part):
    """solve: inv | none | blk | bfs; tree: None | "tc" (19 Nov ctap form) | "tcT" (channel-major taps)"""
    def __init__(self, base, solve, tree=None):
        super().__init__(base, "gdn")
        self.solve, self.tree = solve, tree

    def input_names(self):
        return self.keep + (["anc", "ctap"] if self.tree else [])

    def example(self):
        ex = super().example()
        if not self.tree:
            return ex
        T, f = self.base.T, torch.float16
        ctap = torch.zeros(4, T, T + 3, dtype=f) if self.tree in ("tc", "tcC") else torch.zeros(T + 3, 4 * T, dtype=f)
        return ex + (torch.tril(torch.ones(T, T, dtype=f)), ctap)

    def forward(self, *args):
        SOLVE["mode"] = self.solve
        if self.tree:
            TPB.TREE.update(anc=args[-2], ctap=args[-1], T=self.base.T)   # tri() -> anc; qkv_heads by the form flags
            CM_T["on"] = self.tree == "tcT"
            CM_T["corr"] = self.tree == "tcC"
            args = args[:-2]
        try:
            return super().forward(*args)
        finally:
            SOLVE["mode"] = "inv"
            TPB.TREE.update(anc=None, ctap=None, T=None)
            CM_T["on"] = False
            CM_T["corr"] = False


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    B.OUT = out
    B.P = TP
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
    e8 = B.Entry(mods, 8192, TP, kv_cache_dtype=B.KV_CACHE_DTYPE)
    entries = []
    forms = [("inv", "inv", None), ("none", "none", None), ("tc", "inv", "tc"), ("tcT", "inv", "tcT"), ("tcC", "inv", "tcC")]
    if TP >= 16:
        forms += [("blk", "blk", None), ("bfs", "bfs", None), ("tcTb", "blk", "tcT")]
    for name, s, tr in forms:
        e = GdnPart(e8, s, tr)
        entries.append((f"gdn{TP}_{name}", e, e.input_names(), e.output_names()))
    fname = f"gdnwy_p{TP}_L00-03.aimodel"
    mb = B.save_program(entries, out / fname)
    mp = out / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {"version": "gdn_wy_probe", "kind": "scratch", "chunks": [],
        "head": {"file": "/Volumes/SSD4TB/anemll-forge-research/coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8/head_T8.aimodel"}}
    man["chunks"] = [c for c in man["chunks"] if c["file"] != fname] + [{"file": fname, "layers": [0, 3], "P": TP,
                                                                          "entries": [x[0] for x in entries], "mb": round(mb)}]
    mp.write_text(json.dumps(man, indent=1))
    print(f"{fname}: {len(entries)} entries, {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
