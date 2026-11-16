"""Probe (timing only): attention-layer projections fused into the neighbouring V8 FFN entries, chunk 0 (layers 2, 3):
  ffn2_L2   baseline FFN pair (chunked FFN, V8_SPLIT)
  ffnq2_L2  FFN pair of layer 2, then layer 3's input RMSNorm, mixer rotation and q (with gate) / k / v projections
  offn2_L3  layer 3's o projection (mix_out rotation + o) + residual on each block, then layer 3's FFN pair
Run with build_shared_probe.sh's environment.   BUILDPY build_attproj_probe.py OUT_DIR"""
import gc, sys, time
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_v8_ffn as V  # noqa: E402
import chunkmlp as CM  # noqa: E402
B, M, T = V.B, V.M, 64


class FfnQ2(nn.Module):
    def __init__(self, lf, la):
        super().__init__(); self.lf, self.la = lf, la

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
    def __init__(self, la):
        super().__init__(); self.la = la

    def forward(self, oa, ob, xa, xb):
        res = []
        for o, x in ((oa, xa), (ob, xb)):
            o = self.la.mix.rout(o) if self.la.mix.rout is not None else o
            res.append(self.la.cmlp(x + self.la.mix.o(o)))
        return tuple(res)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ck = M.Checkpoint(); B.KNOWN_LUTS.clear()
    W = {}
    for i in (2, 3):
        W.update(B.layer_arrays(ck, i))
    mods = nn.ModuleList(B.LayerW(W, i) for i in (2, 3)).eval().to(torch.float16)
    for m in mods:
        m.cmlp = CM.ChunkMLP(W, m, V.SPLIT).eval().to(torch.float16)
        m.gate = m.up = m.down = m.rmid = None
    del W; gc.collect()
    l2, l3 = mods
    entries = [("ffn2_L2", V.Ffn2(l2).eval(), ["xa", "xb"], ["ya", "yb"]),
               ("ffnq2_L2", FfnQ2(l2, l3).eval(), ["xa", "xb"], ["ya", "qa", "ka", "va", "yb", "qb", "kb", "vb"]),
               ("offn2_L3", OFfn2(l3).eval(), ["oa", "ob", "xa", "xb"], ["ya", "yb"])]
    # example inputs: Ffn2 has example(); the new modules get theirs here
    f = torch.float16
    FfnQ2.example = lambda self: tuple(torch.randn(1, B.hid, 1, T, dtype=f) * 0.3 for _ in range(2))
    OFfn2.example = lambda self: (torch.randn(1, 6144, 1, T, dtype=f) * 0.3, torch.randn(1, 6144, 1, T, dtype=f) * 0.3,
                                  torch.randn(1, B.hid, 1, T, dtype=f) * 0.3, torch.randn(1, B.hid, 1, T, dtype=f) * 0.3)
    mb = B.save_program(entries, out / "attproj_L02-03.aimodel")
    print(f"attproj_L02-03.aimodel: {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
