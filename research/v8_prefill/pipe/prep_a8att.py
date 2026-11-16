"""A8 attention-projection prep (cd 2 G 16 and cd 1 G 32 LUTs: 32 output rows per HWX subgroup) (16 Nov 2026): what the ANE's runtime-INT8 attention projection programs (v8_work/w8a8/
a8_att_build.py: AQKV, AO) need per attention layer, beside the GPU's expansion of the q / o LUT matrices (the GPU
manifest's own q / o tables, zero-copy from the HWX):
  OUT/att_pal8.bin        int8 [64 layers][2 (q, o)][64][2]: round(127 lut / max|lut[:, d]|) (16-entry palettes, padded)
  OUT/<i>_att.npz         sq [12288], so [5120] fp16 (row scale x max|lut[:, d]| / 127 x 128); kv [2][5120][1024] int8
                          (the export's own k / v int8, transposed), skv [2][1024] fp16 (scale x 128); lrb [5120][192]
                          (B^T of q | k | v), lraq [64][12288] (A_q^T), lrakv [2][64][1024], lrbo [6144][64] (B_o^T),
                          lrao [64][5120] (A_o^T), all fp16 (W = lut part + A B: y = x W^T + (x B^T) A^T)
Checks one layer's q / o reconstruction against the export.
    EXPORT_DIR=<release export> MODEL=<hf model> BUILDPY prep_a8att.py BASE_MANIFEST OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_engine as P  # noqa: E402


def main():
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    man = json.loads(base.read_text())
    ck = P.M.Checkpoint()
    pal8 = np.zeros((64, 2, 64, 2), np.int8)
    for L in man["layers"]:
        if L["kind"] == "gdn":
            continue
        i = L["i"]; W = P.B.layer_arrays(ck, i)
        a = lambda n, x: np.asarray(W[f"{i}/self_attn.{n}_proj.weight/{x}"])  # noqa: E731
        res = {}
        for mi, (n, key) in enumerate((("q", "sq"), ("o", "so"))):
            lut = a(n, "lut").astype(np.float32); sc = a(n, "scale").astype(np.float32).reshape(-1)
            cd = lut.shape[1]
            assert lut.shape[0] == 16 and cd in (1, 2), lut.shape
            md = np.abs(lut).max(0)
            pal8[i, mi, :16, :cd] = np.clip(np.rint(127 * lut / md), -127, 127).astype(np.int8)
            res[key] = (sc * md[np.arange(sc.size) % cd] / 127 * 128).astype(np.float16)
            if i in (3, 27):
                idx = a(n, "idx")
                dense = np.stack([lut[:, d][idx] for d in range(cd)], 1).reshape(-1, idx.shape[1]) * sc[:, None]
                codes = np.stack([pal8[i, mi][:, d][idx] for d in range(cd)], 1).reshape(-1, idx.shape[1]).astype(np.float32)
                rel = np.linalg.norm(codes * (res[key].astype(np.float32)[:, None] / 128) - dense) / np.linalg.norm(dense)
                print(f"layer {i} {n}: int8 palette reconstruction rel err {rel:.2e}", flush=True)
        res["kv"] = np.ascontiguousarray(np.stack([a("k", "int8").T, a("v", "int8").T]))
        res["skv"] = (np.stack([a("k", "scale"), a("v", "scale")]).astype(np.float32) * 128).astype(np.float16)
        res["lrb"] = np.ascontiguousarray(np.concatenate([a(n, "lr_b") for n in ("q", "k", "v")]).T.astype(np.float16))
        res["lraq"] = np.ascontiguousarray(a("q", "lr_a").T.astype(np.float16))
        res["lrakv"] = np.ascontiguousarray(np.stack([a("k", "lr_a").T, a("v", "lr_a").T]).astype(np.float16))
        res["lrbo"] = np.ascontiguousarray(a("o", "lr_b").T.astype(np.float16))
        res["lrao"] = np.ascontiguousarray(a("o", "lr_a").T.astype(np.float16))
        assert res["kv"].shape == (2, 5120, 1024) and res["lrb"].shape == (5120, 192) and res["lraq"].shape == (64, 12288)
        assert res["lrbo"].shape == (6144, 64) and res["lrao"].shape == (64, 5120)
        for n in ("q", "o"):
            mt = L["mats"][n]
            assert mt["bits"] == 4 and mt["cd"] * mt["G"] == 32 and mt["fmul"] == 1.0, (i, n, mt)
        np.savez(out / f"{i}_att.npz", **res)
        print(f"layer {i}: attention projections prepared", flush=True)
    pal8.tofile(out / "att_pal8.bin")
    print("done")


if __name__ == "__main__":
    main()
