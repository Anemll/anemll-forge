"""A8 output-projection prep (16 Nov 2026): every layer's mixer output projection (DeltaNet linear_attn.out_proj,
attention self_attn.o_proj; 6144 -> 5120, 4-bit LUT cd 2 G 16 or cd 1 G 32 + rank-64 A B) for the ANE's AO program
(v8_work/w8a8/a8_att_build.py), beside the GPU's expansion of the LUT part (the GPU manifest's out / o tables):
  OUT/out_pal8.bin   int8 [64 layers][64][2]: round(127 lut / max|lut[:, d]|) (16-entry palettes, padded)
  OUT/<i>_out.npz    so [5120] fp16 (row scale x max|lut[:, d]| / 127 x 128), lrbo [6144][64] (B^T), lrao [64][5120] (A^T)
    EXPORT_DIR=<release export> MODEL=<hf model> BUILDPY prep_a8out.py BASE_MANIFEST OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_engine as P  # noqa: E402


def main():
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    man = json.loads(base.read_text())
    ck = P.M.Checkpoint()
    pal8 = np.zeros((64, 64, 2), np.int8)
    for L in man["layers"]:
        i = L["i"]; W = P.B.layer_arrays(ck, i)
        key, mk = ("linear_attn.out_proj", "out") if L["kind"] == "gdn" else ("self_attn.o_proj", "o")
        a = lambda x: np.asarray(W[f"{i}/{key}.weight/{x}"])  # noqa: E731
        lut = a("lut").astype(np.float32); sc = a("scale").astype(np.float32).reshape(-1)
        cd = lut.shape[1]
        assert lut.shape[0] == 16 and cd in (1, 2) and sc.size == 5120, (i, lut.shape, sc.shape)
        mt = L["mats"][mk]
        assert mt["bits"] == 4 and mt["cd"] * mt["G"] == 32 and mt["fmul"] == 1.0 and mt["cin"] == 6144, (i, mt)
        md = np.abs(lut).max(0)
        pal8[i, :16, :cd] = np.clip(np.rint(127 * lut / md), -127, 127).astype(np.int8)
        so = (sc * md[np.arange(sc.size) % cd] / 127 * 128).astype(np.float16)
        np.savez(out / f"{i}_out.npz", so=so, lrbo=np.ascontiguousarray(a("lr_b").T.astype(np.float16)),
                 lrao=np.ascontiguousarray(a("lr_a").T.astype(np.float16)))
        if i in (4, 36):
            idx = a("idx")
            dense = np.stack([lut[:, d][idx] for d in range(cd)], 1).reshape(-1, idx.shape[1]) * sc[:, None]
            codes = np.stack([pal8[i][:, d][idx] for d in range(cd)], 1).reshape(-1, idx.shape[1]).astype(np.float32)
            rel = np.linalg.norm(codes * (so.astype(np.float32)[:, None] / 128) - dense) / np.linalg.norm(dense)
            print(f"layer {i} {mk} (cd {cd}): int8 palette reconstruction rel err {rel:.2e}", flush=True)
    pal8.tofile(out / "out_pal8.bin")
    print("done")


if __name__ == "__main__":
    main()
