"""Where does the ANE's wide-verify cost go? (20 Nov 2026, tree study; SCRATCH build, deleted after the test.) One target
group (layers 0-3: three DeltaNet layers + the attention layer, the chunked FFN, the served weight constants), split
into parts, each as a verify entry of T = P rows (TREE_P = 8 | 16 | 32, one package per P):
  ffn<T>         the 4 FFNs only (x -> x + mlp(x) per layer; no mixer)
  gdn<T>         the 3 DeltaNet mixers only (x -> x + mix(rms(x)); lazy-commit tape of P rows, chain masks)
  att<T>_<ctx>k  the attention mixer only at 8K / 64K of KV history (x -> x + attn(rms(x)))
  full<T>_8k     the whole group (the served Entry form) for reference
  P = 8 only, where tree8's +5% over chain8 (19 Nov) comes from (build_tree_probe.TreeEntry forms, 8K):
  treeA8_8k      ancestor mask input only (the causal conv as served)
  treeC8_8k      conv-tap selection input only (masks as served)
  treeG8_8k      both, the conv taps gathered (ctap @ rows) then ONE depthwise conv2d of stride 4 over the gathered
                 4 x T rows (the served conv2d's kernel; a chain ctap gives the served conv's inputs in order)
  tree8_8k       both (build_tree_probe's tree8)
    env as v8/build_dec8_one.sh  BUILDPY build_tree_decomp.py OUT_DIR   (TREE_P = 8 | 16 | 32)"""
import gc, json, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v8_ffn as V  # noqa: E402  (imports qwen38_coreai_build as B with this environment)
import chunkmlp as CM  # noqa: E402
import build_tree_probe as TPB  # noqa: E402  (patches B.tri / B.GDNW.qkv_heads only while a TreeEntry traces)
import torch.nn.functional as F  # noqa: E402

B, M = V.B, V.M
TP = int(os.environ.get("TREE_P", "8"))


class Part(nn.Module):
    """A part of base (a B.Entry): mode ffn | gdn | att, with exactly the inputs that part reads."""

    def __init__(self, base, mode):
        super().__init__()
        self.base, self.mode = base, mode
        full = base.input_names()
        if mode == "ffn":
            keep = ["x"]
        elif mode == "gdn":
            import re
            keep = ["x", "conv_sel", "commit", "commit_last"] + [n for n in full if re.fullmatch(r"(conv|rec|pend)\d+", n)]
        else:
            keep = ["x", "cos", "sin", "mask"] + [n for j in base.att_j for s in B.KV_INPUTS[base.kv_mode] for n in base.kv_names(s, j)]
        self.full, self.keep = full, keep

    def input_names(self):
        return self.keep

    def output_names(self):
        b = self.base
        if self.mode == "ffn":
            return ["y"]
        if self.mode == "gdn":
            return ["y"] + [f"{s}{j}_out" for j in b.gdn_j for s in ("conv", "rec", "pend")]
        return ["y"] + [f"{s}{j}_new" for j in b.att_j for s in ("k", "v")]

    def example(self):
        ex = self.base.example()
        return tuple(ex[self.full.index(n)] for n in self.keep)

    def forward(self, *args):
        d = dict(zip(self.keep, args))
        b, x, outs = self.base, d["x"], []
        for j, layer in enumerate(b.layers):
            if self.mode == "ffn":
                x = layer.mlp(x)
            elif self.mode == "gdn" and j in b.gdn_j:
                h = B.rms_hidden(x, layer.ln1)
                y, rows, s1, pend_out = layer.mix.verify(h, d[f"conv{j}"], d["conv_sel"], d[f"rec{j}"], d[f"pend{j}"],
                                                         d["commit"], d["commit_last"], b.T)
                outs += [rows, s1, pend_out]
                x = x + y
            elif self.mode == "att" and j in b.att_j:
                h = B.rms_hidden(x, layer.ln1)
                kv = {}
                for s in B.KV_INPUTS[b.kv_mode]:
                    ns = b.kv_names(s, j)
                    kv[s] = B.SegKV([d[n] for n in ns], 2 if s == "k" and B.KV_KEYS_T else 1) if b.nseg else d[ns[0]]
                y, kt, vt = layer.mix(h, d["cos"], d["sin"], d["mask"], kv["k"], kv["v"], b.ctx, b.T, kv.get("vs"),
                                      b.cache_v8, kv.get("ks"), b.cache_k8)
                outs += [kt, vt]
                x = x + y
        return (x, *outs)


class TreeA(TPB.TreeEntry):
    def input_names(self):
        return B.Entry.input_names(self) + ["anc"]

    def example(self):
        return B.Entry.example(self) + (TPB.TreeEntry.example(self)[-2],)

    def forward(self, *args):
        TPB.TREE.update(anc=args[-1], ctap=None, T=self.T)
        try:
            return B.Entry.forward(self, *args[:-1])
        finally:
            TPB.TREE.update(anc=None, ctap=None, T=None)


class TreeC(TPB.TreeEntry):
    def input_names(self):
        return B.Entry.input_names(self) + ["ctap"]

    def example(self):
        return B.Entry.example(self) + (TPB.TreeEntry.example(self)[-1],)

    def forward(self, *args):
        TPB.TREE.update(anc=None, ctap=args[-1], T=self.T)
        try:
            return B.Entry.forward(self, *args[:-1])
        finally:
            TPB.TREE.update(anc=None, ctap=None, T=None)


GATHER = {"on": False}
_qkv_tree = B.GDNW.qkv_heads


def qkv_heads_gather(self, rows, T):
    if not GATHER["on"] or TPB.TREE["ctap"] is None or T != TPB.TREE["T"]:
        return _qkv_tree(self, rows, T)
    taps = TPB.TREE["ctap"] @ rows                                              # (4, T, cdim)
    cdim = rows.shape[1]
    seq = taps.permute(2, 1, 0).reshape(1, cdim, 1, 4 * T)                      # per channel: row 0's 4 taps, row 1's ...
    conv = F.conv2d(seq, self.cw.transpose(0, 1).reshape(cdim, 1, 1, 4), stride=(1, 4), groups=cdim)
    conv = B.silu(conv).reshape(cdim, T)
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


B.GDNW.qkv_heads = qkv_heads_gather


class TreeG(TPB.TreeEntry):
    def forward(self, *args):
        GATHER["on"] = True
        try:
            return super().forward(*args)
        finally:
            GATHER["on"] = False


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    assert V.SPLIT, "set V8_SPLIT (e.g. 8192,9216)"
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
    mode = B.KV_CACHE_DTYPE
    e8 = B.Entry(mods, 8192, TP, kv_cache_dtype=mode)
    e64 = B.Entry(mods, 65536, TP, kv_cache_dtype=mode)
    entries = []
    for name, e in ((f"ffn{TP}", Part(e8, "ffn")), (f"gdn{TP}", Part(e8, "gdn")), (f"att{TP}_8k", Part(e8, "att")),
                    (f"att{TP}_64k", Part(e64, "att")), (f"full{TP}_8k", e8)):
        entries.append((name, e, e.input_names(), e.output_names()))
    if TP == 8:
        for name, cls in (("treeA8_8k", TreeA), ("treeC8_8k", TreeC), ("treeG8_8k", TreeG), ("tree8_8k", TPB.TreeEntry)):
            e = cls(mods, 8192, 8, kv_cache_dtype=mode)
            entries.append((name, e, e.input_names(), e.output_names()))
    fname = f"decomp_p{TP}_L00-03.aimodel"
    mb = B.save_program(entries, out / fname)
    mp = out / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {"version": "tree_decomp", "kind": "scratch", "chunks": []}
    man["chunks"] = [c for c in man["chunks"] if c["file"] != fname] + [{"file": fname, "layers": [0, 3], "P": TP,
                                                                          "entries": [x[0] for x in entries], "mb": round(mb)}]
    mp.write_text(json.dumps(man, indent=1))
    print(f"{fname}: {len(entries)} entries, {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
