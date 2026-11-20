"""WL3A8 FFN-only test build (queue 4, 19 Nov 2026; TEMPORARY, approved as a separate prefill-only test build to be
deleted after the A/B): the 16 packages of v8_ffn2c's layout (four layers each, ffn_L<i> one 64-row block, ffn2_L<i>
two blocks per prediction), every FFN as w8a8/lut3.py's ServedL3 "i8gud": the release export's 3-bit vector-LUT index
constants with the palette dequantized to INT8 codes (constexpr_blockwise_shift_scale), per-token INT8 activations
into gate / up and into down, INT8 x INT8; the served chunked FFN [8192, 9216], scale-free RMSNorm x (1 + w), input
Hadamard, tanh SiLU, mid Hadamard per chunk, residual add. One index constant per matrix chunk (no FP16 copy of the
weights in the package). Compile with forge.py compile (v8/journal_compile.py), one package at a time.
    env as v8/build_dec8_one.sh (V8_SPLIT=8192,9216 ...) BUILDPY build_v8_ffn8x.py OUT_DIR SEG2_BUILD [first_chunk last_chunk]"""
import gc, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/w8a8")
import build_v8_ffn as V  # noqa: E402  (puts the builder on sys.path)
import chunkmlp as CM  # noqa: E402
import lut3  # noqa: E402

B, M, T = V.B, V.M, V.T


class Ffn(nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def example(self):
        return (torch.randn(1, B.hid, 1, T, dtype=torch.float16) * 0.3,)

    def forward(self, x):
        return self.mod(x)


class Ffn2(nn.Module):  # two 64-row blocks in one prediction, as build_v8_ffn.Ffn2
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def example(self):
        return tuple(torch.randn(1, B.hid, 1, T, dtype=torch.float16) * 0.3 for _ in range(2))

    def forward(self, xa, xb):
        return self.mod(xa), self.mod(xb)


def main():
    out, seg2 = Path(sys.argv[1]), Path(sys.argv[2])
    c0, c1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 15)
    assert V.SPLIT == [8192, 9216], V.SPLIT
    assert B.MLP_SILU == "tanh" and B.MLP_DS_TABLE is None and B.MLP_DS == 1.0, "numerics differ from the seg2 build"
    out.mkdir(parents=True, exist_ok=True)
    mp = out / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {
        "version": "v8_ffn8x", "T": T, "kind": "ffn_only", "temporary": True, "export": str(M.EXPORT_DIR), "seg2": str(seg2),
        "numerics": {"MLP_SILU": B.MLP_SILU, "MLP_DS": B.MLP_DS, "MLP_DS_TABLE": None,
                     "ffn": "WL3A8 i8gud (w8a8/lut3.py ServedL3): INT8 palette codes of the 3-bit indices, per-token INT8 "
                            "activations into gate / up and down, INT8 x INT8"},
        "split": V.SPLIT, "chunks": [], "head": {"file": str(seg2 / "head_T8.aimodel")}}
    ck = M.Checkpoint()
    done = {c["file"] for c in man["chunks"]}
    for ci in range(c0, c1 + 1):
        layers = list(range(4 * ci, 4 * ci + 4))
        name = f"ffn8x_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
        if name in done and (out / name).exists():
            continue
        t0 = time.time()
        B.KNOWN_LUTS.clear()
        W = {}
        for i in layers:
            W.update(B.layer_arrays(ck, i))
        lay = [B.LayerW(W, i).eval().to(torch.float16) for i in layers]
        for m in lay:
            m.cmlp = CM.ChunkMLP(W, m, V.SPLIT).eval().to(torch.float16)   # the served mid Hadamards per chunk
            m.mix = None
            m.gate = m.up = m.down = m.rmid = None                             # (as w8a8/qdq_probe.served_layer)
        mods = [lut3.ServedL3(B, m, lut3.layer_lutw(W, m.i, V.SPLIT), "i8gud").eval() for m in lay]
        del W
        gc.collect()
        entries = [(f"ffn_L{m.i}", Ffn(md).eval(), ["x"], ["y"]) for m, md in zip(lay, mods)]
        entries += [(f"ffn2_L{m.i}", Ffn2(md).eval(), ["xa", "xb"], ["ya", "yb"]) for m, md in zip(lay, mods)]
        mb = B.save_program(entries, out / name)
        man["chunks"] = sorted([c for c in man["chunks"] if c["file"] != name] +
                               [{"file": name, "layers": [layers[0], layers[-1]], "entries": [e[0] for e in entries],
                                 "mb": round(mb), "numerics": B.rot_numerics(layers)}], key=lambda c: c["layers"][0])
        mp.write_text(json.dumps(man, indent=1))
        print(f"{name}: {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)
        del lay, mods, entries
        gc.collect()
    print("manifest", mp)


if __name__ == "__main__":
    main()
