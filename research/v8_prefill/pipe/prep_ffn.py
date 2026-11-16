"""GPU FFN tables for chosen layers of a shared-weight V8 build (15 Nov 2026): the build stores each FFN chunked
(info.json "split", e.g. [8192, 9216]: gate / up = index-row slices of one palette, down = index-column slices with
partial sums; chunkmlp.py). Each chunk's matrix is located in the compiled HWX with prep_engine.table() on its slice
of the export's index rows, exactly as the mixer matrices; no weight is copied (zero-copy from the same HWX).
Writes OUT_DIR/<i>_{gate,up,down}_c<j>.tab and a copy of the base manifest with mats gate_c<j> / up_c<j> (absolute
rows: row0 offset by the chunk start) and down_c<j> (xoff = the chunk's first intermediate channel) added.
    EXPORT_DIR=<release export> BUILDPY prep_ffn.py BUILD_DIR BASE_MANIFEST OUT_MANIFEST LAYERS(comma)"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_engine as P  # noqa: E402


def main():
    build, base, outm = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    layers = [int(x) for x in sys.argv[4].split(",") if x]
    man = json.loads(base.read_text())
    out = Path(man["dir"])
    bman = json.loads((build / "manifest.json").read_text())
    ck = P.M.Checkpoint()
    byi = {L["i"]: L for L in man["layers"]}
    hwx_of = {}
    for i in layers:
        ch = next(c for c in bman["chunks"] if c["layers"][0] <= i <= c["layers"][1])
        info = json.loads((build / (ch["file"] + ".info.json")).read_text())
        sizes = info["split"]
        L = byi[i]
        hi = L["mats"][next(iter(L["mats"]))]["hwx"]   # the layer's chunk HWX (same package as its mixer)
        if hi not in hwx_of:
            h = man["hwx"][hi]
            f = open(h["path"], "rb"); f.seek(h["off"]); hwx_of[hi] = f.read(h["size"])
        K = hwx_of[hi]
        W = P.B.layer_arrays(ck, i)
        b0 = 0
        for j, s in enumerate(sizes):
            b1 = b0 + s
            for m in ("gate", "up", "down"):
                p = f"{i}/mlp.{m}_proj.weight"
                assert p + "/lr_a" not in W, (i, m, "low-rank FFN not supported")
                lut, idx, sc = W[p + "/lut"], W[p + "/idx"], W[p + "/scale"]
                cd = lut.shape[1]
                if m == "down":
                    isub, ssub, roff, xoff = np.ascontiguousarray(idx[:, b0:b1]), sc, 0, b0
                else:
                    isub = np.ascontiguousarray(idx[b0 // cd: b1 // cd])
                    ssub, roff, xoff = np.ascontiguousarray(sc.reshape(-1)[b0:b1]), b0 // cd, 0
                tab, G, fold, cd2 = P.table(K, lut, isub, ssub)
                tab = tab.copy(); tab[:, 2] += roff
                name = f"{m}_c{j}"
                tab.astype(np.uint32).tofile(out / f"{i}_{name}.tab")
                L["mats"][name] = {"hwx": hi, "tab": f"{i}_{name}.tab", "bits": int(np.log2(lut.shape[0])), "G": G,
                                   "cin": int(isub.shape[1]), "cout": int(idx.shape[0] * cd) if m == "down" else s,
                                   "cd": cd2, "fmul": 1.0 / fold, "lr_a": None, "lr_b": None, "xoff": xoff}
                print(f"layer {i} {name}: {L['mats'][name]['bits']}-bit cd{cd2} G{G} {L['mats'][name]['cout']}x{isub.shape[1]} "
                      f"rows +{roff} xoff {xoff} fold {fold:g}", flush=True)
            b0 = b1
        assert b0 == 17408
    outm.write_text(json.dumps(man, indent=1))
    print("manifest", outm)


if __name__ == "__main__":
    main()
