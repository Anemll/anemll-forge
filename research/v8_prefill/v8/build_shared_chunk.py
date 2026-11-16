"""Shared-weight chunk (probe): ONE package per target chunk whose entries all run from the same weight constants:
  - the target entries exactly as qwen38_coreai_build.build_chunk makes them (v8_<ctx>k decode, p<rows>_<ctx>k prefill),
    except that each layer's FFN is the chunked form (chunkmlp.ChunkMLP, V8_SPLIT): same weights, split intermediate;
  - V8 FFN entries ffn_L<i> / ffn2_L<i> (build_v8_ffn.Ffn / Ffn2) on those same ChunkMLP modules;
  - V8 DeltaNet mixer entries mix_L<i> / mix2_L<i> (build_v8_mix.Mix / Mix2) on the same GDNW modules.
Core AI stores a constant once per package, so the package should be about the size of the plain target chunk.
Run with the target build's environment (build_seg2_96.sh) plus V8_SPLIT.
    BUILDPY build_shared_chunk.py OUT_DIR FIRST_LAYER LAST_LAYER"""
import gc, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v8_ffn as V  # noqa: E402  (imports qwen38_coreai_build as B with this environment)
import build_v8_mix as MX  # noqa: E402
import chunkmlp as CM  # noqa: E402

B, M = V.B, V.M
import os  # noqa: E402
ATTPROJ = os.environ.get("V8_ATTPROJ") == "1"   # + ffnq2_L<i-1> / offn2_L<i> around every attention layer i


class FfnQ2(nn.Module):
    """FFN pair of layer i-1, then attention layer i's input RMSNorm, mixer rotation and q (with gate) / k / v projections."""
    def __init__(self, lf, la):
        super().__init__(); self.lf, self.la = lf, la

    def example(self):
        return tuple(torch.randn(1, B.hid, 1, 64, dtype=torch.float16) * 0.3 for _ in range(2))

    def forward(self, xa, xb):
        outs = []
        for x in (xa, xb):
            y = self.lf.cmlp(x)
            h = B.rms_hidden(y, self.la.ln1)
            h = self.la.mix.rmix(h) if self.la.mix.rmix is not None else h
            q, k, v = self.la.mix.project(h)
            outs += [y, q, k, v]
        return tuple(outs)


class OFfn2(nn.Module):
    """Attention layer i's o projection (mix_out rotation + o) + residual on each block, then layer i's FFN pair."""
    def __init__(self, la):
        super().__init__(); self.la = la

    def example(self):
        f = torch.float16
        return (torch.randn(1, 6144, 1, 64, dtype=f) * 0.3, torch.randn(1, 6144, 1, 64, dtype=f) * 0.3,
                torch.randn(1, B.hid, 1, 64, dtype=f) * 0.3, torch.randn(1, B.hid, 1, 64, dtype=f) * 0.3)

    def forward(self, oa, ob, xa, xb):
        res = []
        for o, x in ((oa, xa), (ob, xb)):
            o = self.la.mix.rout(o) if self.la.mix.rout is not None else o
            res.append(self.la.cmlp(x + self.la.mix.o(o)))
        return tuple(res)


def main():
    out, l0, l1 = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    assert V.SPLIT, "set V8_SPLIT (e.g. 8192,9216)"
    out.mkdir(parents=True, exist_ok=True)
    B.OUT = out
    layers = list(range(l0, l1 + 1))
    t0 = time.time()
    ck = M.Checkpoint()
    B.KNOWN_LUTS.clear()
    W = {}
    for i in layers:
        W.update(B.layer_arrays(ck, i))
    mods = nn.ModuleList(B.LayerW(W, i) for i in layers).eval().to(torch.float16)
    for m in mods:
        m.cmlp = CM.ChunkMLP(W, m, V.SPLIT).eval().to(torch.float16)
        object.__setattr__(m, "mlp", m.cmlp)        # the target entries call layer.mlp: now the chunked FFN
        m.gate = m.up = m.down = m.rmid = None       # the unchunked FFN weights are not referenced any more
    del W
    gc.collect()
    entries = []
    ctxs =[int(c) for c in sys.argv[4].split(",")] if len(sys.argv) > 4 else [8192, 16384, 32768, 65536, 98304]
    shapes = [(f"v8_{c // 1024}k", c, 8) for c in ctxs] + [(f"p{tp}_{c // 1024}k", c, tp) for c in ctxs for tp in B.prefill_rows(c)]
    mode = B.KV_CACHE_DTYPE
    for name, c, rows in shapes:
        e = B.Entry(mods, c, rows, kv_cache_dtype=mode)
        entries.append((name, e, e.input_names(), e.output_names()))
    e0 = entries[0][1]
    for m in mods:
        entries.append((f"ffn_L{m.i}", V.Ffn(m).eval(), ["x"], ["y"]))
        entries.append((f"ffn2_L{m.i}", V.Ffn2(m).eval(), ["xa", "xb"], ["ya", "yb"]))
        if m.kind == "linear_attention":
            entries.append((f"mix_L{m.i}", MX.Mix(m).eval(), ["x", "conv", "rec"], ["y", "conv_out", "rec_out"]))
            entries.append((f"mix2_L{m.i}", MX.Mix2(m).eval(), ["xa", "xb", "conv", "rec"], ["ya", "yb", "conv_out", "rec_out"]))
    if ATTPROJ:
        byi = {m.i: m for m in mods}
        for m in mods:
            if m.kind != "linear_attention" and m.i - 1 in byi:
                entries.append((f"ffnq2_L{m.i - 1}", FfnQ2(byi[m.i - 1], m).eval(), ["xa", "xb"],
                                ["ya", "qa", "ka", "va", "yb", "qb", "kb", "vb"]))
                entries.append((f"offn2_L{m.i}", OFfn2(m).eval(), ["oa", "ob", "xa", "xb"], ["ya", "yb"]))
    fname = f"chunk_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
    mb = B.save_program(entries, out / fname)
    info = {"file": fname, "layers": [layers[0], layers[-1]], "entries": [x[0] for x in entries], "gdn_j": e0.gdn_j,
            "att_j": e0.att_j, "taps": e0.taps, "mb": round(mb), "split": V.SPLIT}
    (out / f"{fname}.info.json").write_text(json.dumps(info, indent=1))
    print(f"{fname}: {len(entries)} entries, {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
