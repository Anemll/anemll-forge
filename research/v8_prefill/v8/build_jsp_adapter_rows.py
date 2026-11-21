"""E1 / E2 (20 Nov 2026): row sweep of the JetSpec++ adapter on the ANE (a SCRATCH latency build, deleted after).
One package of COUNT adapter blocks (default 3: blocks 0-2, the D2 b3 layout) built exactly as
build_jsp_adapter_ane.py builds them (the target builder's attention + MLP layer on each block's LUT tensors, INT8 KV
history), with one entry per row count: blk<T>_<ctx>k for T in ROWS (default 1,4,8,16,32,64). The builder's verify /
prefill switch (B.P, `T > P`) is set to T while each entry is constructed and traced, so every row count takes the
same verify-form attention (ATT_BLOCK tiles) the served 16-row entry takes.
E2 A/B (MLP=w4a8q / w4a16q, Anemll's form): INT4 weight codes as a constexpr constant (no palette) x INT8 activations
through quantize / dequantize (w4a8q) or FP16 activations (w4a16q).
E2 A/B (MLP=w8a8t): the same INT8 weight values stored DENSE (1 byte per weight, no palette), per-tensor activations.
E2 (MLP=w4a8 / w4a8t): the adapter MLPs as INT8 x INT8 on the LUT4 indices (w8a8/lut3.py's form, one index tensor,
the 16-entry palette as INT8 codes through constexpr_blockwise_shift_scale): gate / up on the rotated RMSNorm output
quantized to INT8 per token (w4a8: dynamic per-token peak, lut3.AQd) or per tensor (w4a8t: a constant scale per
block input from MLP_ASCALE, calibrated on the GPTQ inputs), down on the rotated SwiGLU output the same way.
Attention projections stay FP16-activation LUT4.
    env as v8/build_dec8_one.sh  BUILDPY build_jsp_adapter_rows.py LUT_DIR OUT_DIR [FIRST] [COUNT] [CTX] [ROWS]
    env: JSP_ROT=1 (rotated LUT4 forms), MLP=fp16|w4a8|w4a8t, MLP_ASCALE=json (w4a8t)"""
import gc, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_jsp_adapter_ane as A  # noqa: E402  (imports build_v8_ffn -> qwen38_coreai_build as B)

B = A.B
MLP = os.environ.get("MLP", "fp16")
SPLIT = [x for x in os.environ.get("SPLIT_OC", "").split(",") if x]   # E1b: output-channel splits (q,k,v,down,...)


class StackT(A.Stack):
    """A.Stack with the builder's P set to T while it is constructed and traced (verify form at every row count)."""

    def __init__(self, layers, ctx, T):
        old = B.P
        B.P = T
        try:
            super().__init__(layers, ctx, T)
        finally:
            B.P = old
        self.T = T

    def forward(self, *args):
        old = B.P
        B.P = self.T
        try:
            return super().forward(*args)
        finally:
            B.P = old


class SplitConv(nn.Module):
    """E1b: a QConv as N output-channel parts with separate constants (row slices of idx / scale), outputs concatenated:
    exact per channel; gives the bonded compiler independent convs it can place on both ANE instances."""

    def __init__(self, W, key, n=2):
        super().__init__()
        o = W[f"{key}/idx"].shape[0]
        assert o % n == 0, (key, o, n)
        step, parts = o // n, []
        for j in range(n):
            Wj = {f"{key}.p{j}/lut": W[f"{key}/lut"], f"{key}.p{j}/idx": W[f"{key}/idx"][j * step:(j + 1) * step],
                  f"{key}.p{j}/scale": W[f"{key}/scale"][j * step:(j + 1) * step]}
            parts.append(B.QConv(Wj, f"{key}.p{j}"))
        self.parts = nn.ModuleList(parts)

    def forward(self, x):
        return torch.cat([m(x) for m in self.parts], 1)


def split_oc(m, W, i, which, n=2):
    """replace the named projections of LayerW m by SplitConv (q / k / v / o of the attention, gate / up / down; with
    an INT8 MLP the MLP matrices are split inside MlpA8 instead)"""
    for w in which:
        if MLP != "fp16" and w in ("gate", "up", "down"):
            continue
        if w in ("q", "k", "v", "o"):
            sc = SplitConv(W, f"{i}/self_attn.{w}_proj.weight", n).to(torch.float16)
            setattr(m.mix, w, sc)
        else:
            sc = SplitConv(W, f"{i}/mlp.{w}_proj.weight", n).to(torch.float16)
            setattr(m, w, sc)


class LutW4(nn.Module):
    """One LUT4 matrix (o, i): idx 0..15, the per-tensor 16-entry palette, the per-row scale. FP16 path w16(); INT8 path
    w8(): palette codes round(lut / s), s = max|lut| / 127 (constexpr_blockwise_shift_scale on the expanded INT8 codes:
    the ANE expands the indices online, nothing materializes). Buffer names indices / lut8 / lut16 as in lut3.LutW."""

    def __init__(self, z, dense=False, int4=False):
        super().__init__()
        lut = z["lut"].astype(np.float32).reshape(-1)
        idx = z["idx"].astype(np.uint8)
        o, i = idx.shape
        s = float(np.abs(lut).max()) / 127.0
        l8 = np.clip(np.rint(lut / s), -127, 127).astype(np.int8)
        self.dense, self.int4 = dense, int4
        if int4:   # E2 W4A8 as Anemll specifies (no palette): INT4 codes as a compile-time constant (constexpr_blockwise_shift_scale,
            # 4-bit packed: values in [-8, 7], input_dtype int4) with ONE shared scale, the per-row scale a multiply after
            # the conv (the shared-scale form reaches the INT8 ceiling on 3 Oct). TIMING codes: the LUT4 index - 8 (a
            # uniform grid, not the palette's values); accuracy of a real uniform INT4 quantization is the replay's job.
            self.register_buffer("q4", torch.from_numpy(np.ascontiguousarray(idx.astype(np.int8) - 8)).view(o, i, 1, 1))
            self.register_buffer("s4", torch.tensor(float(np.abs(lut).max()) / 8.0, dtype=torch.float16).view(1, 1, 1, 1))
        elif dense:   # E2 A/B (w8a8t): the SAME INT8 values stored dense (l8[idx], 1 byte per weight), no palette
            self.register_buffer("w8dense", torch.from_numpy(np.ascontiguousarray(l8[idx])).view(o, i, 1, 1))
        else:
            self.register_buffer("indices", torch.from_numpy(np.ascontiguousarray(idx)).view(o, i, 1, 1))
            self.register_buffer("lut16", torch.from_numpy(lut.astype(np.float16)).view(1, 1, 1, 1, 16, 1))
            self.register_buffer("lut8", torch.from_numpy(l8).view(1, 1, 1, 1, 16, 1))
        self.register_buffer("s", torch.tensor(s, dtype=torch.float16).view(1, 1, 1, 1))
        self.register_buffer("rowscale", torch.from_numpy(z["scale"].astype(np.float16).reshape(-1)).view(1, -1, 1, 1))
        self.s_f, self.l8_np = s, l8

    def w8(self):
        if self.int4:
            return torch.ops.coreai.constexpr_blockwise_shift_scale(self.q4, self.s4, None, None, torch.int4)
        if self.dense:
            return torch.ops.coreai.constexpr_blockwise_shift_scale(self.w8dense, self.s, None, None, torch.int8)
        return torch.ops.coreai.constexpr_blockwise_shift_scale(torch.ops.coreai.lut_to_dense(self.indices, self.lut8, 0),
                                                                self.s, None, None, torch.int8)


class AQt(nn.Module):
    """Per-tensor INT8 activations with a CONSTANT scale (calibrated): q = quantize(x / a) -> dequantize(unit)."""

    def __init__(self, a):
        super().__init__()
        self.register_buffer("inv", torch.tensor(127.0 / a, dtype=torch.float16))
        self.register_buffer("one", torch.tensor(1.0, dtype=torch.float16))
        self.register_buffer("unit", torch.tensor(1 / 128, dtype=torch.float16))
        self.register_buffer("z", torch.tensor(0, dtype=torch.int8))
        self.fac = a * 128.0 / 127.0

    def forward(self, x):
        q = torch.ops.coreai.quantize(x * self.inv, self.one, torch.int8, zero_point=self.z)
        return torch.ops.coreai.dequantize(q, self.unit, zero_point=self.z, output_dtype=torch.float16), self.fac


class MlpA8(nn.Module):
    """The builder's LayerW.mlp (scale-free RMSNorm x (1 + w), input Hadamard rin, gate / up, SiLU, mid Hadamard rmid,
    down, residual) with INT8 x INT8 gate / up / down on the LUT4 indices."""

    def __init__(self, layer, zs, mode, ascale=None):
        super().__init__()
        self.__dict__["L"] = layer
        import lut3  # noqa: E402  (w8a8/)
        dn, q4 = mode == "w8a8t", mode in ("w4a8q", "w4a16q")
        self.g, self.u = LutW4(zs["gate"], dn, q4), LutW4(zs["up"], dn, q4)
        zd = zs["down"]
        if "down" in SPLIT:   # E2: down as two output-channel halves (separate INT8 constants), outputs concatenated
            o2 = zd["idx"].shape[0] // 2
            self.dparts = nn.ModuleList(LutW4({"lut": zd["lut"], "idx": zd["idx"][j * o2:(j + 1) * o2], "scale": zd["scale"][j * o2:(j + 1) * o2]}, dn, q4)
                                        for j in range(2))
            self.d = None
        else:
            self.d, self.dparts = LutW4(zd, dn, q4), None
        if mode == "w4a16q":
            self.ah = self.ad = None
        elif mode == "w4a8":
            self.ah, self.ad = lut3.AQd(), lut3.AQd()
        else:
            self.ah, self.ad = AQt(ascale["h"]), AQt(ascale["a"])

    def forward(self, x):
        L = self.L
        h = B.rms_hidden(x, L.ln2)
        h = L.rin(h) if L.rin is not None else h
        hq, fh = self.ah(h) if self.ah is not None else (h, 1.0)
        gg = F.conv2d(hq, self.g.w8()) * fh * self.g.rowscale
        uu = F.conv2d(hq, self.u.w8()) * fh * self.u.rowscale
        a = B.silu(gg, B.MLP_SILU) * uu
        a = L.rmid(a) if L.rmid is not None else a
        aq, fa = self.ad(a) if self.ad is not None else (a, 1.0)
        if self.dparts is not None:
            return x + torch.cat([F.conv2d(aq, d.w8()) * fa * d.rowscale for d in self.dparts], 1)
        return x + F.conv2d(aq, self.d.w8()) * fa * self.d.rowscale


def main():
    lut_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    ctx = int(sys.argv[5]) if len(sys.argv) > 5 else 8192
    rows = [int(r) for r in (sys.argv[6] if len(sys.argv) > 6 else "1,4,8,16,32,64").split(",")]
    out.mkdir(parents=True, exist_ok=True)
    B.OUT = out
    t0 = time.time()
    B.KNOWN_LUTS.clear()
    names = A.BLOCKS[first:first + count]
    rot = os.environ.get("JSP_ROT") == "1"
    asc = json.loads(Path(os.environ["MLP_ASCALE"]).read_text()) if MLP in ("w4a8t", "w8a8t", "w4a8q", "w4a16q") else None
    layers = nn.ModuleList()
    for k, name in enumerate(names):
        i = A.ATT_IDX[first + k]
        W = A.block_W(lut_dir, name, i)
        if rot:
            W[f"{i}/mix.rotation"] = np.array([5000 + i, 6000 + i], np.int64)
            W[f"{i}/mlp.rotation"] = np.array([7000 + i, 8000 + i], np.int64)
        if MLP != "fp16":   # the MLP matrices go through LutW4, not the builder's palettizer
            for m in ("gate", "up", "down"):
                for f in ("lut", "idx", "scale"):
                    W.pop(f"{i}/mlp.{m}_proj.weight/{f}")
            for m in ("gate", "up", "down"):   # placeholders (dense 1 x 1; dropped from the layer below)
                W[f"{i}/mlp.{m}_proj.weight/dense"] = np.zeros((1, 1), np.float16)
        m = B.LayerW(W, i).eval().to(torch.float16) if MLP == "fp16" else _layer_a8(W, i, lut_dir, name, asc)
        if SPLIT:
            split_oc(m, W, i, SPLIT, int(os.environ.get("SPLIT_N", "2")))
        if rot:
            assert m.rin is not None and m.rmid is not None and m.mix.rmix is not None and m.mix.rout is not None, (name, i)
        layers.append(m)
        del W
        gc.collect()
    entries = []
    for T in rows:
        e = StackT(layers, ctx, T)
        entries.append((f"blk{T}_{ctx // 1024}k", e, e.input_names(), e.output_names()))
    for T in [int(r) for r in os.environ.get("ROWS_MLP", "").split(",") if r]:   # the first block's MLP alone: a pure weight stream
        entries.append((f"mlp{T}", A.MlpOnly(layers[0], T).eval(), ["x"], ["y"]))
    tag = {"fp16": "", "w4a8": "_w4a8", "w4a8t": "_w4a8t", "w8a8t": "_w8a8t", "w4a8q": "_w4a8q", "w4a16q": "_w4a16q"}[MLP]
    tag += ("_s" + "".join(x[0] for x in SPLIT) + os.environ.get("SPLIT_N", "2")) if SPLIT else ""
    fname = f"jsp_rows_b{first:02d}-{first + count - 1:02d}{'_rot' if rot else ''}{tag}.aimodel"
    mb = B.save_program(entries, out / fname)
    mp = out / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {"version": "jsp_adapter_rows", "kind": "scratch", "chunks": [],
        "head": {"file": "/Volumes/SSD4TB/anemll-forge-research/coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8/head_T8.aimodel"}}
    man["chunks"] = [c for c in man["chunks"] if c["file"] != fname] + [{"file": fname, "layers": [first, first + count - 1],
                                                                          "blocks": names, "P": 16, "rows": rows, "ctx": ctx,
                                                                          "mlp": MLP, "lut_dir": str(lut_dir), "rot": rot, "split_oc": SPLIT,
                                                                          "entries": [x[0] for x in entries], "mb": round(mb)}]
    mp.write_text(json.dumps(man, indent=1))
    print(f"{fname}: {names} as target attention layers {A.ATT_IDX[first:first + count]}, MLP {MLP}, rows {rows}, "
          f"{len(entries)} entries, {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)


def _layer_a8(W, i, lut_dir, name, asc):
    """LayerW with the MLP replaced by MlpA8 (the builder's rin / rmid / ln2 kept: they come from the mlp.rotation and
    post_attention_layernorm entries; the placeholder MLP matrices are dropped)."""
    m = B.LayerW(W, i).eval().to(torch.float16)
    zs = {k: np.load(lut_dir / f"{name}.mlp.{k}_proj.weight.npz") for k in ("gate", "up", "down")}
    a8 = MlpA8(m, zs, MLP, asc[name] if asc is not None else None).eval()
    m.a8mlp = a8.to(torch.float16)          # registered (its buffers trace as constants), then shadows LayerW.mlp
    object.__setattr__(m, "mlp", m.a8mlp)
    m.gate = m.up = m.down = None
    return m


if __name__ == "__main__":
    main()
