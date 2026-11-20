"""Single-call tree verify on the ANE, cost probe (queue 13b, 19 Nov 2026; SCRATCH build, deleted after the test unless
it wins): one target group (layers 0-3: three DeltaNet layers + the attention layer, the chunked FFN) as verify entries
at 8K / 32K / 64K of KV history, every entry on the same weight constants:
  chain<T>_<ctx>k   the served verify entry (qwen38_coreai_build.Entry, T = P rows, lazy commit), T = 8
  tree<T>_<ctx>k    the same with two more inputs: anc (T x T fp16, row i's ancestors incl. itself; rows in topological
                    order) and ctap (4 x T x (T + 3) fp16, one-hot: row i's conv taps = its 3 nearest ancestors, reaching
                    into the 3 history rows, then itself). While tracing a tree entry, tri(T, False) -> anc and
                    tri(T, True) -> anc minus its diagonal (the DeltaNet WY / UT block: cum, pair, the strictly lower
                    N of the triangular solve, the output's intra-block term; and the attention's intra-block mask), and
                    the causal conv1d reads its taps through ctap. The lazy-commit tape (pending k / u / wk / cum,
                    commit / commit_last) is unchanged: commit = the accepted path's ancestor mask.
TREE_P = 8 (default): chain8 + tree8 entries; TREE_P = 16 | 32: tree16 | tree32 entries (P = 16 | 32 pending rows; own package).
    env as v8/build_dec8_one.sh  BUILDPY build_tree_probe.py OUT_DIR   (TREE_P = 8 | 16)"""
import gc, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v8_ffn as V  # noqa: E402  (imports qwen38_coreai_build as B with this environment)
import chunkmlp as CM  # noqa: E402

B, M = V.B, V.M
CTXS = [8192, 32768, 65536]
TP = int(os.environ.get("TREE_P", "8"))
TREE = {"anc": None, "ctap": None, "T": None}

_tri0 = B.tri


def tri_tree(n, strict):
    if TREE["anc"] is not None and n == TREE["T"]:
        a = TREE["anc"]
        return a * (1 - torch.eye(n, dtype=a.dtype)) if strict else a
    return _tri0(n, strict)


B.tri = tri_tree
_qkv_heads0 = B.GDNW.qkv_heads


def qkv_heads_tree(self, rows, T):
    if TREE["ctap"] is None or T != TREE["T"]:
        return _qkv_heads0(self, rows, T)
    taps = TREE["ctap"] @ rows                                                 # (4, T, cdim): each row's 4 conv taps
    conv = (taps * self.cw.reshape(4, 1, -1)).sum(0)                            # (T, cdim), cw (4, cdim)
    conv = B.silu(conv).transpose(0, 1)                                         # (cdim, T)
    kd = B.kd
    qq, kk, vv = conv[:kd], conv[kd:2 * kd], conv[2 * kd:]
    nk, nv, dk, dv = B.nk, B.nv, B.dk, B.dv

    def heads(t):
        t = t.reshape(nk, dk, T).permute(0, 2, 1)
        return t.reshape(nk, 1, T, dk).repeat(1, nv // nk, 1, 1).reshape(nv, T, dk)

    def l2n(t, s):
        return t * torch.rsqrt((t * t).sum(-1, keepdim=True) + 1e-6) * s
    vh = vv.reshape(nv, dv, T).permute(0, 2, 1)
    vh = vh * B.GDN_SV if B.GDN_SV != 1 else vh
    return l2n(heads(qq), dk ** -0.5 * B.GDN_SQ), l2n(heads(kk), 1.0), vh


B.GDNW.qkv_heads = qkv_heads_tree


class TreeEntry(B.Entry):
    def input_names(self):
        return super().input_names() + ["anc", "ctap"]

    def example(self):
        T, f = self.T, torch.float16
        return super().example() + (torch.from_numpy(np.tril(np.ones((T, T))).astype(np.float16)), torch.zeros(4, T, T + 3, dtype=f))

    def forward(self, *args):
        TREE.update(anc=args[-2], ctap=args[-1], T=self.T)
        try:
            return super().forward(*args[:-2])
        finally:
            TREE.update(anc=None, ctap=None, T=None)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    assert V.SPLIT, "set V8_SPLIT (e.g. 8192,9216)"
    B.OUT = out
    B.P = TP
    assert B.P == TP
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
    entries, mode = [], B.KV_CACHE_DTYPE
    for c in CTXS:
        if TP == 8:
            e = B.Entry(mods, c, 8, kv_cache_dtype=mode)
            entries.append((f"chain8_{c // 1024}k", e, e.input_names(), e.output_names()))
        e = TreeEntry(mods, c, TP, kv_cache_dtype=mode)
        assert not e.prefill
        entries.append((f"tree{TP}_{c // 1024}k", e, e.input_names(), e.output_names()))
    fname = f"tree_p{TP}_L00-03.aimodel"
    mb = B.save_program(entries, out / fname)
    mp = out / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {"version": "tree_probe", "kind": "scratch", "chunks": [],
                                                           "head": {"file": "/Volumes/SSD4TB/anemll-forge-research/coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8/head_T8.aimodel"}}
    man["chunks"] = [c for c in man["chunks"] if c["file"] != fname] + [{"file": fname, "layers": [0, 3], "P": TP, "entries": [x[0] for x in entries],
                                                                          "mb": round(mb)}]
    mp.write_text(json.dumps(man, indent=1))
    print(f"{fname}: {len(entries)} entries, {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
