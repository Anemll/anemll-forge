"""V8 ANE side: FFN-only packages for every layer, the production FFN exactly as the target build's chunks run it
(qwen38_coreai_build.LayerW.mlp: scale-free RMSNorm x (1+w), input Hadamard, LUT gate / up, tanh SiLU, mid Hadamard,
LUT down, residual add), one 64-row function per layer (ffn_L<i>, x -> y), four layers per package. The manifest lists
the packages for `forge.py compile` and names the seg2 build's head (already compiled) as its head.
    EXPORT_DIR=<release export> BUILDPY build_v8_ffn.py OUT_DIR SEG2_BUILD [first_chunk last_chunk]"""
import gc, json, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn

V02 = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02"
sys.path[:0] = [V02 + "/coreai", V02 + "/scripts"]
import qwen38_coreai_build as B  # noqa: E402
import qwen38_ane_model as M  # noqa: E402

T = 64
SPLIT = [int(v) for v in os.environ["V8_SPLIT"].split(",")] if os.environ.get("V8_SPLIT") else None  # chunked FFN


class Ffn(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer, self.T = layer, T

    def example(self):
        return (torch.randn(1, B.hid, 1, T, dtype=torch.float16) * 0.3,)

    def forward(self, x):
        return self.layer.cmlp(x) if SPLIT else self.layer.mlp(x)


class Ffn2(nn.Module):  # two 64-row blocks in one prediction (one dispatch gap per pair instead of two)
    def __init__(self, layer):
        super().__init__()
        self.layer, self.T = layer, T

    def example(self):
        return tuple(torch.randn(1, B.hid, 1, T, dtype=torch.float16) * 0.3 for _ in range(2))

    def forward(self, xa, xb):
        f = self.layer.cmlp if SPLIT else self.layer.mlp
        return f(xa), f(xb)


def main():
    out, seg2 = Path(sys.argv[1]), Path(sys.argv[2])
    c0, c1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 15)
    assert B.MLP_SILU == "tanh" and B.MLP_DS_TABLE is None and B.MLP_DS == 1.0, "numerics differ from the seg2 build"
    out.mkdir(parents=True, exist_ok=True)
    mp = out / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {
        "version": "v8_ffn1", "T": T, "kind": "ffn_only", "export": str(M.EXPORT_DIR), "seg2": str(seg2),
        "numerics": {"MLP_SILU": B.MLP_SILU, "MLP_DS": B.MLP_DS, "MLP_DS_TABLE": None}, "split": SPLIT, "chunks": [],
        "head": {"file": str(seg2 / "head_T8.aimodel")}}
    ck = M.Checkpoint()
    done = {c["file"] for c in man["chunks"]}
    for ci in range(c0, c1 + 1):
        layers = list(range(4 * ci, 4 * ci + 4))
        name = f"ffn_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
        if name in done and (out / name).exists():
            continue
        t0 = time.time()
        B.KNOWN_LUTS.clear()
        W = {}
        for i in layers:
            W.update(B.layer_arrays(ck, i))
        mods = [B.LayerW(W, i).eval().to(torch.float16) for i in layers]
        if SPLIT:
            import chunkmlp as CM
            for m in mods:
                m.cmlp = CM.ChunkMLP(W, m, SPLIT).eval().to(torch.float16)
        del W
        gc.collect()
        for m in mods:  # keep only what the FFN uses (the mixer weights would not be traced anyway)
            m.mix = None
        entries = [(f"ffn_L{m.i}", Ffn(m).eval(), ["x"], ["y"]) for m in mods]
        if os.environ.get("V8_FFN2", "1") == "1":
            entries += [(f"ffn2_L{m.i}", Ffn2(m).eval(), ["xa", "xb"], ["ya", "yb"]) for m in mods]
        mb = B.save_program(entries, out / name)
        man["chunks"] = sorted([c for c in man["chunks"] if c["file"] != name] +
                               [{"file": name, "layers": [layers[0], layers[-1]], "entries": [e[0] for e in entries],
                                 "mb": round(mb), "numerics": B.rot_numerics(layers)}], key=lambda c: c["layers"][0])
        mp.write_text(json.dumps(man, indent=1))
        print(f"{name}: {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)
        del mods, entries
        gc.collect()
    print("manifest", mp)


if __name__ == "__main__":
    main()
