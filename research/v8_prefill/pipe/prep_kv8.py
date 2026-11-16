"""Attention k / v projections as the export's own int8 weights (15 Nov 2026): writes OUT/<i>_{k,v}.i8 (int8 [1024][5120])
and <i>_{k,v}.sc (fp32 per-row scales) for every attention layer and a manifest copy whose dense k / v entries carry
"w8" / "sc" (the engine's I8KV path reads these instead of the fp16 copy; same weights: fp16 = int8 x scale).
    EXPORT_DIR=<release export> BUILDPY prep_kv8.py BASE_MANIFEST OUT_MANIFEST"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_engine as P  # noqa: E402


def main():
    base, outm = Path(sys.argv[1]), Path(sys.argv[2])
    man = json.loads(base.read_text()); out = Path(man["dir"])
    ck = P.M.Checkpoint()
    for L in man["layers"]:
        if L["kind"] == "gdn":
            continue
        i = L["i"]; W = P.B.layer_arrays(ck, i)
        for n in ("k", "v"):
            q = f"{i}/self_attn.{n}_proj.weight"
            w8, sc = np.ascontiguousarray(W[q + "/int8"], np.int8), np.ascontiguousarray(W[q + "/scale"], np.float32).reshape(-1)
            assert w8.shape == (1024, 5120) and sc.shape == (1024,), (w8.shape, sc.shape)
            w8.tofile(out / f"{i}_{n}.i8"); sc.tofile(out / f"{i}_{n}.sc")
            ref = np.fromfile(out / L["dense"][n]["w"], np.float16).reshape(1024, 5120).astype(np.float32)
            err = np.abs(ref - w8.astype(np.float32) * sc[:, None]).max()
            L["dense"][n].update({"w8": f"{i}_{n}.i8", "sc": f"{i}_{n}.sc"})
            print(f"layer {i} {n}: int8 {w8.shape}, max |fp16 copy - int8 x scale| {err:.3g}", flush=True)
    outm.write_text(json.dumps(man, indent=1)); print("manifest", outm)


if __name__ == "__main__":
    main()
