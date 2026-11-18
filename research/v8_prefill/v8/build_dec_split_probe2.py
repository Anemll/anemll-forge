"""Decode-verify split, probe 2 (16 Nov 2026): TWO target chunks (layers L0 .. L0+7) in one package, for timing:
  v8_16k / v8_32k / v8_64k / v8_96k  chunk 0's verify entries exactly as the shared build makes them (the ANE reference
                                     at every context entry of the 96K ladder);
  dva0, dvb0                         chunk 0's split (build_dec_split_probe.DecA / DecB);
  dvba                               FUSED: part B of chunk 0's attention layer (rout, o, + x, FFN) then part A of chunk 1
                                     (its DeltaNet layers + the attention layer's projections): one ANE call per chunk
                                     boundary instead of two;
  dvb1                               part B of chunk 1.
Run with build_dec_split_probe2.sh (the shared build's environment).
    BUILDPY build_dec_split_probe2.py OUT_DIR [FIRST_LAYER]"""
import gc, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v8_ffn as V  # noqa: E402
import chunkmlp as CM  # noqa: E402
from build_dec_split_probe import DecA, DecB  # noqa: E402

B, M = V.B, V.M


class DecBA(nn.Module):
    def __init__(self, b: DecB, a: DecA, tap_prev: bool):
        super().__init__(); self.b, self.a, self.tap_prev = b, a, tap_prev

    def input_names(self):
        return ["o", "xr_prev"] + self.a.input_names()[1:]

    def output_names(self):
        return (["y_prev"] if self.tap_prev else []) + self.a.output_names()

    def example(self):
        return self.b.example() + self.a.example()[1:]

    def forward(self, o, xr_prev, *rest):
        y = self.b(o, xr_prev)
        out = self.a(y, *rest)
        return (y, *out) if self.tap_prev else out


def main():
    out, l0 = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0
    assert V.SPLIT, "set V8_SPLIT (e.g. 8192,9216)"
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
    for m in mods:
        m.cmlp = CM.ChunkMLP(W, m, V.SPLIT).eval().to(torch.float16)
        object.__setattr__(m, "mlp", m.cmlp)
        m.gate = m.up = m.down = m.rmid = None
    del W
    gc.collect()
    c0, c1 = nn.ModuleList(mods[:4]), nn.ModuleList(mods[4:])
    entries = []
    for c in (16384, 32768, 65536, 98304):
        e = B.Entry(c0, c, 8, kv_cache_dtype=B.KV_CACHE_DTYPE)
        entries.append((f"v8_{c // 1024}k", e, e.input_names(), e.output_names()))
    r0 = entries[0][1]
    r1 = B.Entry(c1, 16384, 8, kv_cache_dtype=B.KV_CACHE_DTYPE)   # (not saved: chunk 1's gdn_j / att_j / taps)
    assert r0.att_j == [3] and r1.att_j == [3], (r0.att_j, r1.att_j)
    a0, b0 = DecA(c0, r0.gdn_j, r0.att_j, r0.taps).eval(), DecB(c0[-1]).eval()
    a1, b1 = DecA(c1, r1.gdn_j, r1.att_j, r1.taps).eval(), DecB(c1[-1]).eval()
    ba = DecBA(b0, a1, c0[-1].i in r0.taps).eval()
    entries += [("dva0", a0, a0.input_names(), a0.output_names()), ("dvb0", b0, ["o", "x"], ["y"]),
                ("dvba", ba, ba.input_names(), ba.output_names()), ("dvb1", b1, ["o", "x"], ["y"])]
    fname = f"chunks_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
    mb = B.save_program(entries, out / fname)
    info = {"file": fname, "layers": [layers[0], layers[-1]], "entries": [x[0] for x in entries], "gdn_j": r0.gdn_j,
            "att_j": r0.att_j, "taps0": r0.taps, "taps1": r1.taps, "mb": round(mb), "split": V.SPLIT,
            "io": {n: {"in": i, "out": o} for n, _, i, o in entries}}
    (out / f"{fname}.info.json").write_text(json.dumps(info, indent=1))
    print(f"{fname}: {len(entries)} entries, {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
