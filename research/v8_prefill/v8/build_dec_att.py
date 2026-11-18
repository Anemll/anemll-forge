"""Partial decode attention on the ANE (17 Nov 2026): one small weight-free package whose entries attend the 8 verify
rows' queries over the FIRST N keys of the KV history (whole segments, so no mask: always full once the context passes
them), with exactly the served verify's history math (qwen38_coreai_build.AttnW, ATT_INT8MM s8 / s8b / sm8 / pvf8,
two-pass softmax over ATT_BLOCK tiles, INT8 keys stored dim-major, INT8 values, per-token scales). The GPU attends the
rest of the keys at the same time (engine/dec_kernels.metal dec_combine_part) and the two partials are merged.
No weights, no context ladder: the same programs serve every attention layer and every context past N.
  att8k   segment 0           (keys [0, 8K))
  att16k  segments 0-1        ([0, 16K))
  att32k  segments 0-2        ([0, 32K))
  att64k  segments 0-3        ([0, 64K))
Inputs q [24][8][256] (part A's output), k_<s> [4][256][L], ks_<s> [4][L], v_<s> [4][L][256], vs_<s> [4][L]; outputs
o [24][8][256] (normalized over these keys), m, den [24][8] (row max and softmax sum: the merge weight).
Environment: build_dec_att.sh (the dec8 build's).
    BUILDPY build_dec_att.py OUT_DIR"""
import json, sys, time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v8_ffn as V  # noqa: E402  (imports qwen38_coreai_build as B with this environment)

B = V.B
SEGS = [8192, 8192, 16384, 32768]


class AttPart(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        import coreai_torch._compression.custom_layers  # noqa: F401  (native quantize / dequantize)
        assert B.KV_CACHE_DTYPE == "kv8" and B.KV_KEYS_T and B.ATT_INT8MM == "s8,s8b,sm8,pvf8", (B.KV_CACHE_DTYPE, B.ATT_INT8MM)
        self.sizes, self.blk = list(sizes), B.ATT_BLOCK
        assert all(L % self.blk == 0 for L in self.sizes)
        f16 = torch.float16
        self.register_buffer("v8_unit", torch.tensor(1 / 128, dtype=f16))
        self.register_buffer("v8_zero", torch.tensor(0, dtype=torch.int8))
        self.register_buffer("s8_unit", torch.tensor(B.ATT_S8_UNIT, dtype=f16))
        self.register_buffer("s8b_unit", torch.tensor(B.ATT_S8B_UNIT, dtype=f16))
        import os
        self.register_buffer("pf8_unit", torch.tensor(float(os.environ.get("ATT_PF8_UNIT", 1 / 64)), dtype=f16))

    def input_names(self):
        return ["q"] + [f"{n}_{s}" for s in range(len(self.sizes)) for n in ("k", "ks", "v", "vs")]

    def output_names(self):
        return ["o", "m", "den"]

    def example(self):
        ex = [torch.randn(B.nh, 8, B.hd, dtype=torch.float16)]
        for L in self.sizes:
            ex += [torch.randint(-127, 128, (B.nkv, B.hd, L), dtype=torch.int8), torch.full((B.nkv, L), 0.02, dtype=torch.float16),
                   torch.randint(-127, 128, (B.nkv, L, B.hd), dtype=torch.int8), torch.full((B.nkv, L), 0.01, dtype=torch.float16)]
        return tuple(ex)

    def forward(self, q, *kv):
        nkv, nh, hd, T = B.nkv, B.nh, B.hd, 8
        grp, blk = nh // nkv, self.blk
        q8, d8 = B.quant8, B.dequant8
        qg4 = q.permute(1, 0, 2).reshape(T, nkv, grp, hd).permute(1, 2, 0, 3).reshape(nkv, grp * T, hd)   # rows (g, t)
        it, tiles = iter(kv), []
        for L in self.sizes:
            k8, ks, v8, vs = next(it), next(it), next(it), next(it)
            kd = d8(k8, self.v8_unit, self.v8_zero)                       # codes / 128 (the verify's k_st)
            for a in range(0, L, blk):
                s_ = (qg4 @ kd[:, :, a:a + blk]) * hd ** -0.5
                s_ = d8(q8(s_, self.s8_unit, self.v8_zero), self.s8_unit, self.v8_zero)                   # s8
                s_ = s_ * (ks[:, a:a + blk].reshape(nkv, 1, blk) * 128)
                s_ = d8(q8(s_, self.s8b_unit, self.v8_zero), self.s8b_unit, self.v8_zero)                 # s8b
                tiles.append((s_, v8[:, a:a + blk], vs[:, a:a + blk]))
        mts = [t_[0].amax(-1, keepdim=True) for t_ in tiles]
        m = mts[0]
        for m_t in mts[1:]:
            m = torch.maximum(m, m_t)
        den = num = None
        for (s_, vt8, vst), m_t in zip(tiles, mts):                      # the verify's two-pass pvf8 / sm8 tile
            e_ = torch.exp(s_ - m_t)
            e_ = d8(q8(e_, self.pf8_unit, None, torch.float8_e4m3fn), self.pf8_unit, None)              # sm8
            w_t = torch.exp(m_t - m)
            d_ = e_.sum(-1, keepdim=True) * w_t
            vs_t = vst.reshape(nkv, 1, blk) * 128
            vmax = torch.clamp_min(vs_t.amax(-1, keepdim=True), 1e-4)
            pn = e_ * (vs_t * (1 / vmax))
            ed = d8(q8(pn, self.pf8_unit, None, torch.float8_e4m3fn), self.pf8_unit, None)              # pvf8
            vd = d8(vt8, self.v8_unit, self.v8_zero)
            vd = d8(q8(vd, self.v8_unit, self.v8_zero), self.v8_unit, self.v8_zero)
            n_ = (ed @ vd) * (w_t * vmax)
            den = d_ if den is None else den + d_
            num = n_ if num is None else num + n_
        o = (num / den).reshape(nh, T, hd)
        return o, m.reshape(nh, T), den.reshape(nh, T)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    B.OUT = out
    t0 = time.time()
    entries = []
    for n in range(1, len(SEGS) + 1):
        mod = AttPart(SEGS[:n]).eval()
        entries.append((f"att{sum(SEGS[:n]) // 1024}k", mod, mod.input_names(), mod.output_names()))
    fname = "dec_att.aimodel"
    mb = B.save_program(entries, out / fname)
    info = {"file": fname, "entries": [e[0] for e in entries], "segs": SEGS, "block": B.ATT_BLOCK, "mb": round(mb, 2),
            "io": {n: {"in": i, "out": o} for n, _, i, o in entries}}
    (out / f"{fname}.info.json").write_text(json.dumps(info, indent=1))
    print(f"{fname}: {len(entries)} entries, {mb:.2f} MB in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
