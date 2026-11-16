"""A8 FFN prep (16 Nov 2026): what the GPU needs to expand every layer's FFN (the shared build's chunked LUT matrices,
read zero-copy from the HWX) into the runtime-INT8 weights of the ANE's A8 FFN program (v8_work/w8a8/a8_ffn_real_probe.py
form: bg / bu [17][5120][1024], bd [17][1024][5120], codes per output row).
  OUT/<i>_{gate,up,down}_c<j>.tab   HWX subgroup tables (prep_engine.table, as prep_ffn.py), every layer
  OUT/pal8.bin                      int8 [64 layers][3 (gate, up, down)][64][2]: round(127 lut / max|lut[:, d]|)
  OUT/<i>_a8s.f16                   fp16 sg [17408], su [17408] (x seed_mid signs), sd [5120]: row scale x max|lut[:, d]|
                                    / 127 x 128 (the ANE program dequantizes codes by 1/128)
  OUT_MANIFEST                      BASE_MANIFEST + mats gate_c<j> / up_c<j> / down_c<j> (absolute table paths) per layer
Codes depend only on the palette index (one int8 palette per matrix), so the expander is a table lookup. Checks one
layer's reconstruction against the export's dense weights.
    EXPORT_DIR=<release export> MODEL=<hf model> BUILDPY prep_a8.py BUILD_DIR BASE_MANIFEST OUT_DIR OUT_MANIFEST"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_engine as P  # noqa: E402

HID, INTER = 5120, 17408


def main():
    build, base, out, outm = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads(base.read_text())
    bman = json.loads((build / "manifest.json").read_text())
    ck = P.M.Checkpoint()
    byi = {L["i"]: L for L in man["layers"]}
    pal8 = np.zeros((64, 3, 64, 2), np.int8)
    cur_hi, K = None, None
    for i in range(64):
        ch = next(c for c in bman["chunks"] if c["layers"][0] <= i <= c["layers"][1])
        sizes = json.loads((build / (ch["file"] + ".info.json")).read_text())["split"]
        L = byi[i]
        hi = L["mats"][next(m for m in L["mats"] if not m.startswith(("gate_c", "up_c", "down_c")))]["hwx"]
        if hi != cur_hi:   # one chunk HWX in memory at a time
            h = man["hwx"][hi]
            with open(h["path"], "rb") as f:
                f.seek(h["off"]); K = f.read(h["size"])
            cur_hi = hi
        W = P.B.layer_arrays(ck, i)
        mid = np.fromfile(Path(man["dir"]) / L["rot"]["mlp_mid"], np.float32)
        assert mid.shape == (INTER,) and set(np.unique(mid)) <= {-1.0, 1.0}
        sc_out = {}
        for mi, m in enumerate(("gate", "up", "down")):
            p = f"{i}/mlp.{m}_proj.weight"
            lut, idx, sc = (np.asarray(W[p + x]) for x in ("/lut", "/idx", "/scale"))
            assert lut.shape == (64, 2), lut.shape
            md = np.abs(lut.astype(np.float32)).max(0)                       # per component
            pal8[i, mi] = np.clip(np.rint(127 * lut.astype(np.float32) / md), -127, 127).astype(np.int8)
            rs = sc.reshape(-1).astype(np.float32) * md[np.arange(sc.size) % 2] / 127 * 128
            sc_out[m] = rs * mid if m == "up" else rs
            cd = lut.shape[1]
            b0 = 0
            for j, s in enumerate(sizes):
                b1 = b0 + s
                if m == "down":
                    isub, ssub, roff, xoff = np.ascontiguousarray(idx[:, b0:b1]), sc, 0, b0
                else:
                    isub = np.ascontiguousarray(idx[b0 // cd: b1 // cd])
                    ssub, roff, xoff = np.ascontiguousarray(sc.reshape(-1)[b0:b1]), b0 // cd, 0
                tab, G, fold, cd2 = P.table(K, lut, isub, ssub)
                tab = tab.copy(); tab[:, 2] += roff
                name = f"{m}_c{j}"
                tp = out / f"{i}_{name}.tab"
                tab.astype(np.uint32).tofile(tp)
                L["mats"][name] = {"hwx": hi, "tab": str(tp), "bits": int(np.log2(lut.shape[0])), "G": G,
                                   "cin": int(isub.shape[1]), "cout": int(idx.shape[0] * cd) if m == "down" else s,
                                   "cd": cd2, "fmul": 1.0 / fold, "lr_a": None, "lr_b": None, "xoff": xoff}
                b0 = b1
            assert b0 == INTER
        np.concatenate([sc_out["gate"], sc_out["up"], sc_out["down"]]).astype(np.float16).tofile(out / f"{i}_a8s.f16")
        assert all(abs(float(v)) < 6e4 for v in (np.abs(sc_out["gate"]).max(), np.abs(sc_out["up"]).max(), np.abs(sc_out["down"]).max()))
        if i == 36:   # reconstruction check: codes x scales vs the export's dense weights
            for mi, m in enumerate(("gate", "up", "down")):
                p = f"{i}/mlp.{m}_proj.weight"
                lut, idx, sc = (np.asarray(W[p + x]) for x in ("/lut", "/idx", "/scale"))
                dense = np.stack([lut[:, d][idx] for d in range(2)], 1).reshape(-1, idx.shape[1]).astype(np.float32) \
                    * sc.reshape(-1, 1).astype(np.float32)
                codes = np.stack([pal8[i, mi][:, d][idx] for d in range(2)], 1).reshape(-1, idx.shape[1]).astype(np.float32)
                rs = sc_out[m] * (mid if m == "up" else 1) / 128
                rel = np.linalg.norm(codes * rs[:, None] - dense) / np.linalg.norm(dense)
                print(f"layer {i} {m}: int8 palette reconstruction rel err {rel:.2e}", flush=True)
        print(f"layer {i}: tables {len(sizes)} chunks x 3, G {G}", flush=True)
    pal8.tofile(out / "pal8.bin")
    outm.write_text(json.dumps(man, indent=1))
    print("manifest", outm)


if __name__ == "__main__":
    main()
