"""INT8 gate / up twin of the FFN-only build v8_ffn2c (17 Nov 2026): the same 16 packages of four layers with ffn_L<i>
(one 64-row block) and ffn2_L<i> (two blocks per prediction), the chunked FFN [8192, 9216], but gate / up as
w8a8/mlp8.py builds them (INT8 palette, per-token INT8 activations, K split in two). Everything else is the served
FFN. For a same-session prefill A/B in the V8 driver (--ffn): this build against v8_ffn2c, speed and KL.
    env as v8/build_dec8_one.sh (V8_SPLIT=8192,9216 ...) BUILDPY build_v8_ffn8.py OUT_DIR SEG2_BUILD [first_chunk last_chunk]"""
import gc, json, sys, time
from pathlib import Path

import torch

sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/w8a8")
import build_v8_ffn as V  # noqa: E402  (puts the builder on sys.path)
import chunkmlp as CM  # noqa: E402
import mlp8  # noqa: E402

B, M = V.B, V.M


def main():
    out, seg2 = Path(sys.argv[1]), Path(sys.argv[2])
    c0, c1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 15)
    assert V.SPLIT == [8192, 9216], V.SPLIT
    assert B.MLP_SILU == "tanh" and B.MLP_DS_TABLE is None and B.MLP_DS == 1.0, "numerics differ from the seg2 build"
    out.mkdir(parents=True, exist_ok=True)
    mp = out / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {
        "version": "v8_ffn8", "T": V.T, "kind": "ffn_only", "export": str(M.EXPORT_DIR), "seg2": str(seg2),
        "numerics": {"MLP_SILU": B.MLP_SILU, "MLP_DS": B.MLP_DS, "MLP_DS_TABLE": None,
                     "gate_up": "INT8 palette (absmax/127 per matrix) x per-token INT8 activations, K split 2 x 2560"},
        "split": V.SPLIT, "chunks": [], "head": {"file": str(seg2 / "head_T8.aimodel")}}
    ck = M.Checkpoint()
    done = {c["file"] for c in man["chunks"]}
    for ci in range(c0, c1 + 1):
        layers = list(range(4 * ci, 4 * ci + 4))
        name = f"ffn8_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
        if name in done and (out / name).exists():
            continue
        t0 = time.time()
        B.KNOWN_LUTS.clear()
        W = {}
        for i in layers:
            W.update(B.layer_arrays(ck, i))
        mods = [B.LayerW(W, i).eval().to(torch.float16) for i in layers]
        for m in mods:
            m.cmlp = CM.ChunkMLP(W, m, V.SPLIT).eval().to(torch.float16)
            m.cmlp = mlp8.make_mlp8(B, W, m, V.SPLIT)     # Ffn / Ffn2 call layer.cmlp (V.SPLIT is set)
            m.mix = None
        del W
        gc.collect()
        entries = [(f"ffn_L{m.i}", V.Ffn(m).eval(), ["x"], ["y"]) for m in mods]
        entries += [(f"ffn2_L{m.i}", V.Ffn2(m).eval(), ["xa", "xb"], ["ya", "yb"]) for m in mods]
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
