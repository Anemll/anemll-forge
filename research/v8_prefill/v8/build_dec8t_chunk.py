"""build_dec8t_chunk.py (20 Nov 2026, tree study): build_dec8_chunk.py's package (every entry unchanged, so the chain
path is the served one) plus single-call tree verify entries on the same weight constants:
  g0_t8_<ctx>k, g1_t8_<ctx>k   each group's v8_<ctx>k verify entry with two more inputs, anc (8 x 8 fp16 ancestor mask,
                               rows in topological order) and ctap (4 x 8 x 11 fp16 one-hot conv taps); the DeltaNet WY /
                               UT block and the attention's intra-block mask take anc, the causal conv its taps through
                               ctap (build_tree_probe.TreeEntry; TREE_FORM=gather: the taps gathered then the served
                               depthwise conv kernel at stride 4, build_tree_decomp.TreeG)
  tdva, tdvba                  dva / dvba (the split verify's ANE parts) with anc / ctap the same way (dvb has no
                               DeltaNet layer and serves both)
TREE_CTXS (default 8192,16384,32768,65536: the entries the server runs without the split) picks the t8 contexts.
Original notes:
8-layer shared-weight package for the decode-verify split (17 Nov 2026): two target groups (layers L0..L0+3 and
L0+4..L0+7, the attention layer last in each) in ONE package, every entry on the same weight constants:
  g0_<entry>, g1_<entry>  each group's target entries exactly as build_shared_chunk makes them (v8_<ctx>k verify,
                          p<rows>_<ctx>k prefill) on the chunked FFN (V8_SPLIT);
  ffn_L<i>, ffn2_L<i>, mix_L<i>, mix2_L<i>, ffnq2_L<i-1>, offn2_L<i>
                          the V8 FFN / mixer entries, exactly as build_shared_chunk (V8_ATTPROJ=1);
  dva                     part A of group 0 (build_dec_split_probe.DecA): its DeltaNet layers + FFNs, then the attention
                          layer's norm, q (with gate) / k / v projections, q / k norm, RoPE;
  dvba                    part B of group 0 (o projection, + residual, FFN) then part A of group 1: ONE ANE call
                          (build_dec_split_probe2.DecBA; y_prev out when group 0's last layer is a drafter tap);
  dvb                     part B of group 1.
The split entries bind no KV and no mask: the same programs serve every context. The attention core between them runs
on the GPU (pipe/engine/dec_kernels.metal). Environment: build_dec8_one.sh (= build_shared2_one.sh's).
    BUILDPY build_dec8_chunk.py OUT_DIR FIRST_LAYER TAPS   (TAPS: the manifest's drafter taps, e.g. 5,19,33,47,61)"""
import gc, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v8_ffn as V  # noqa: E402  (imports qwen38_coreai_build as B with this environment)
import build_v8_mix as MX  # noqa: E402
import build_shared_chunk as SC  # noqa: E402
import chunkmlp as CM  # noqa: E402
from build_dec_split_probe import DecA, DecB  # noqa: E402
from build_dec_split_probe2 import DecBA  # noqa: E402
import os  # noqa: E402
import build_tree_probe as TPB  # noqa: E402  (B.tri / GDNW.qkv_heads take anc / ctap only while a tree entry traces)
TREE_FORM = os.environ.get("TREE_FORM", "ctap")
if TREE_FORM == "gather":
    import build_tree_decomp as TDC  # noqa: E402,F401  (its qkv_heads_gather patch; GATHER["on"] while tracing)
TREE_CTXS = [int(c) for c in os.environ.get("TREE_CTXS", "8192,16384,32768,65536").split(",") if c]


def _tree_on(anc, ctap):
    TPB.TREE.update(anc=anc, ctap=ctap, T=8)
    if TREE_FORM == "gather":
        TDC.GATHER["on"] = True


def _tree_off():
    TPB.TREE.update(anc=None, ctap=None, T=None)
    if TREE_FORM == "gather":
        TDC.GATHER["on"] = False


B, M = V.B, V.M
CTXS = [8192, 16384, 32768, 65536, 98304]


class TreeV(B.Entry):
    """the v8_<ctx>k verify entry + anc / ctap (TreeEntry's form, TREE_FORM conv taps)"""
    def input_names(self):
        return super().input_names() + ["anc", "ctap"]

    def example(self):
        f = torch.float16
        return super().example() + (torch.tril(torch.ones(self.T, self.T, dtype=f)), torch.zeros(4, self.T, self.T + 3, dtype=f))

    def forward(self, *args):
        _tree_on(args[-2], args[-1])
        try:
            return super().forward(*args[:-2])
        finally:
            _tree_off()


class TreeDecA(DecA):
    def input_names(self):
        return super().input_names() + ["anc", "ctap"]

    def example(self):
        f = torch.float16
        return super().example() + (torch.tril(torch.ones(8, 8, dtype=f)), torch.zeros(4, 8, 11, dtype=f))

    def forward(self, *args):
        _tree_on(args[-2], args[-1])
        try:
            return super().forward(*args[:-2])
        finally:
            _tree_off()


# tdvba = DecBA(dvb part of group 0, TreeDecA of group 1): its inputs end with the tree DecA's anc / ctap


def main():
    out, l0, taps_all = Path(sys.argv[1]), int(sys.argv[2]), [int(t) for t in sys.argv[3].split(",")]
    assert V.SPLIT, "set V8_SPLIT (e.g. 8192,9216)"
    assert SC.ATTPROJ, "set V8_ATTPROJ=1 (the served v8_shared2 build has ffnq2 / offn2)"
    out.mkdir(parents=True, exist_ok=True)
    B.OUT = out
    layers = list(range(l0, l0 + 8))
    t0 = time.time()
    ck = M.Checkpoint()
    B.KNOWN_LUTS.clear()
    W = {}
    for i in layers:
        W.update(B.layer_arrays(ck, i))
    mods = nn.ModuleList(B.LayerW(W, i) for i in layers).eval().to(torch.float16)
    for m in mods:   # the shared build's chunked FFN (same weights as v8_shared2)
        m.cmlp = CM.ChunkMLP(W, m, V.SPLIT).eval().to(torch.float16)
        object.__setattr__(m, "mlp", m.cmlp)
        m.gate = m.up = m.down = m.rmid = None
    del W
    gc.collect()
    groups = [nn.ModuleList(mods[:4]), nn.ModuleList(mods[4:])]
    entries, ginfo, mode = [], [], B.KV_CACHE_DTYPE
    for gi, g in enumerate(groups):
        shapes = [(f"v8_{c // 1024}k", c, 8) for c in CTXS] + [(f"p{tp}_{c // 1024}k", c, tp) for c in CTXS for tp in B.prefill_rows(c)]
        names, e0 = {}, None
        for name, c, rows in shapes:
            e = B.Entry(g, c, rows, kv_cache_dtype=mode)
            e0 = e0 or e
            entries.append((f"g{gi}_{name}", e, e.input_names(), e.output_names()))
            names[name] = f"g{gi}_{name}"
        for c in TREE_CTXS:   # tree verify entries (same layers, same weight constants)
            e = TreeV(g, c, 8, kv_cache_dtype=mode)
            entries.append((f"g{gi}_t8_{c // 1024}k", e, e.input_names(), e.output_names()))
            names[f"t8_{c // 1024}k"] = f"g{gi}_t8_{c // 1024}k"
        assert e0.att_j == [3], f"group {gi}: the attention layer must be last ({e0.att_j})"
        ginfo.append({"layers": [g[0].i, g[-1].i], "gdn_j": e0.gdn_j, "att_j": e0.att_j, "taps": e0.taps, "entries": names})
    for m in mods:   # V8 FFN / mixer entries (build_shared_chunk)
        entries.append((f"ffn_L{m.i}", V.Ffn(m).eval(), ["x"], ["y"]))
        entries.append((f"ffn2_L{m.i}", V.Ffn2(m).eval(), ["xa", "xb"], ["ya", "yb"]))
        if m.kind == "linear_attention":
            entries.append((f"mix_L{m.i}", MX.Mix(m).eval(), ["x", "conv", "rec"], ["y", "conv_out", "rec_out"]))
            entries.append((f"mix2_L{m.i}", MX.Mix2(m).eval(), ["xa", "xb", "conv", "rec"], ["ya", "yb", "conv_out", "rec_out"]))
    byi = {m.i: m for m in mods}
    for m in mods:
        if m.kind != "linear_attention" and m.i - 1 in byi:
            entries.append((f"ffnq2_L{m.i - 1}", SC.FfnQ2(byi[m.i - 1], m).eval(), ["xa", "xb"],
                            ["ya", "qa", "ka", "va", "yb", "qb", "kb", "vb"]))
            entries.append((f"offn2_L{m.i}", SC.OFfn2(m).eval(), ["oa", "ob", "xa", "xb"], ["ya", "yb"]))
    g0, g1 = ginfo
    a0 = DecA(groups[0], g0["gdn_j"], g0["att_j"], g0["taps"]).eval()
    b0 = DecB(groups[0][-1]).eval()
    a1 = DecA(groups[1], g1["gdn_j"], g1["att_j"], g1["taps"]).eval()
    b1 = DecB(groups[1][-1]).eval()
    ba = DecBA(b0, a1, groups[0][-1].i in taps_all).eval()
    split = [("dva", a0, a0.input_names(), a0.output_names()), ("dvba", ba, ba.input_names(), ba.output_names()),
             ("dvb", b1, ["o", "x"], ["y"])]
    ta0 = TreeDecA(groups[0], g0["gdn_j"], g0["att_j"], g0["taps"]).eval()
    ta1 = TreeDecA(groups[1], g1["gdn_j"], g1["att_j"], g1["taps"]).eval()
    tba = DecBA(b0, ta1, groups[0][-1].i in taps_all).eval()
    split += [("tdva", ta0, ta0.input_names(), ta0.output_names()), ("tdvba", tba, tba.input_names(), tba.output_names())]
    entries += split
    fname = f"chunk_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
    mb = B.save_program(entries, out / fname)
    info = {"file": fname, "layers": [layers[0], layers[-1]], "entries": [x[0] for x in entries], "groups": ginfo,
            "split": {n: {"in": i, "out": o} for n, _, i, o in split}, "y_prev_tap": groups[0][-1].i in taps_all,
            "mb": round(mb), "split_ffn": V.SPLIT, "tree": {"form": TREE_FORM, "ctxs": TREE_CTXS, "rows": 8}}
    (out / f"{fname}.info.json").write_text(json.dumps(info, indent=1))
    print(f"{fname}: {len(entries)} entries, {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
