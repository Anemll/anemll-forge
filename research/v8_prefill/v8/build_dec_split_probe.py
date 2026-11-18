"""Decode-verify split around the attention layer, probe build (16 Nov 2026): ONE target chunk (layers L0 .. L0+3, the
attention layer last) with three entries on the same weight constants:
  v8_16k  the target's T=8 verify entry exactly as the shared build makes it (the reference);
  dva     part A: the chunk's DeltaNet layers in verify form (states in / out, the release commit logic) and their FFNs,
          then the attention layer up to its attention core: input RMSNorm, mixer rotation, q (with gate) / k / v
          projections, q / k RMSNorm and RoPE, exactly as AttnW.forward computes them. No KV inputs.
          outputs: xr (the residual stream entering the attention layer), q / g (query, gate) [24][8][256], kt / vt
          (the new key / value rows) [4][8][256], the taps, the DeltaNet state outputs;
  dvb     part B: o (the gated attention output, channel-major [1][6144][1][8]) -> mix_out rotation -> o projection,
          + xr, then the attention layer's FFN -> y.
The attention core in between runs on the GPU (pipe/unit/decode_step_gpu.swift). Run with the shared build's environment
(v8/build_shared2_one.sh) plus OUT:
    BUILDPY build_dec_split_probe.py OUT_DIR [FIRST_LAYER]"""
import gc, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v8_ffn as V  # noqa: E402  (imports qwen38_coreai_build as B with this environment)
import chunkmlp as CM  # noqa: E402

B, M = V.B, V.M


class DecA(nn.Module):
    def __init__(self, layers, gdn_j, att_j, taps):
        super().__init__()
        self.layers, self.gdn_j, self.att_j, self.taps = layers, gdn_j, att_j, taps

    def input_names(self):
        return ["x", "cos", "sin", "conv_sel", "commit", "commit_last"] + [f"{s}{j}" for j in self.gdn_j for s in ("conv", "rec", "pend")]

    def output_names(self):
        return (["xr", "q", "g", "kt", "vt"] + [f"tap{l}" for l in self.taps]
                + [f"{s}{j}_out" for j in self.gdn_j for s in ("conv", "rec", "pend")])

    def forward(self, x, cos, sin, conv_sel, commit, commit_last, *rest):
        T, nh, nkv, hd, rot = 8, B.nh, B.nkv, B.hd, B.rot
        it = iter(rest)
        gdn_in = {j: (next(it), next(it), next(it)) for j in self.gdn_j}
        taps, gdn_out = [], []
        for j, layer in enumerate(self.layers):
            h = B.rms_hidden(x, layer.ln1)
            if j in gdn_in:
                cr, rc, pd = gdn_in[j]
                y, rows, s1, pend_out = layer.mix.verify(h, cr, conv_sel, rc, pd, commit, commit_last, T)
                gdn_out += [rows, s1, pend_out]
                x = layer.mlp(x + y)
                if layer.i in self.taps:
                    taps.append(x)
                continue
            a = layer.mix                                   # the attention layer (the chunk's last): up to the core
            hq = a.rmix(h) if a.rmix is not None else h
            q_o, k_o, v_o = a.project(hq)

            def tmajor(t_, c):
                return t_.reshape(c, T).transpose(0, 1)
            qg = tmajor(q_o, 2 * nh * hd).reshape(T, nh, 2 * hd)
            qh, gate = B.rms_last(qg[:, :, :hd], a.qn), qg[:, :, hd:]
            kh = B.rms_last(tmajor(k_o, nkv * hd).reshape(T, nkv, hd), a.kn)
            vh = tmajor(v_o, nkv * hd).reshape(T, nkv, hd)
            c3, s3 = cos.reshape(T, 1, rot), sin.reshape(T, 1, rot)

            def rope(t_):
                r, rest_ = t_[..., :rot], t_[..., rot:]
                return torch.cat([r * c3 + torch.cat([-r[..., rot // 2:], r[..., :rot // 2]], -1) * s3, rest_], -1)
            q_out = rope(qh).permute(1, 0, 2)               # (nh, T, hd)
            g_out = gate.permute(1, 0, 2)                    # (nh, T, hd)
            kt, vt = rope(kh).permute(1, 0, 2), vh.permute(1, 0, 2)
            return (x, q_out, g_out, kt, vt, *taps, *gdn_out)
        raise AssertionError("no attention layer")

    def example(self):
        f = torch.float16
        ex = [torch.randn(1, B.hid, 1, 8, dtype=f) * 0.02, torch.ones(8, B.rot, dtype=f), torch.zeros(8, B.rot, dtype=f),
              torch.zeros(3, B.P + 3, dtype=f), torch.zeros(1, B.P, 1, dtype=f), torch.zeros(1, B.P, 1, dtype=f)]
        for _ in self.gdn_j:
            ex += [torch.zeros(B.P + 3, B.cdim, dtype=f), torch.zeros(B.nv, B.dk, B.dv, dtype=f), torch.zeros(B.nv, 3 * B.P + 1, B.dv, dtype=f)]
        return tuple(ex)


class DecB(nn.Module):
    def __init__(self, la):
        super().__init__(); self.la = la

    def example(self):
        f = torch.float16
        return (torch.randn(1, B.nh * B.hd, 1, 8, dtype=f) * 0.3, torch.randn(1, B.hid, 1, 8, dtype=f) * 0.3)

    def forward(self, o, x):
        a = self.la.mix
        o = a.rout(o) if a.rout is not None else o
        return self.la.mlp(x + a.o(o))


def main():
    out, l0 = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0
    assert V.SPLIT, "set V8_SPLIT (e.g. 8192,9216)"
    out.mkdir(parents=True, exist_ok=True)
    B.OUT = out
    layers = list(range(l0, l0 + 4))
    t0 = time.time()
    ck = M.Checkpoint()
    B.KNOWN_LUTS.clear()
    W = {}
    for i in layers:
        W.update(B.layer_arrays(ck, i))
    mods = nn.ModuleList(B.LayerW(W, i) for i in layers).eval().to(torch.float16)
    for m in mods:   # the shared build's chunked FFN (same weights as v8_shared2)
        m.cmlp = CM.ChunkMLP(W, m, V.SPLIT).eval().to(torch.float16)
        object.__setattr__(m, "mlp", m.cmlp)
        m.gate = m.up = m.down = m.rmid = None
    del W
    gc.collect()
    ref = B.Entry(mods, 16384, 8, kv_cache_dtype=B.KV_CACHE_DTYPE)
    assert ref.att_j == [len(layers) - 1], f"the attention layer must be the chunk's last: att_j {ref.att_j}"
    a = DecA(mods, ref.gdn_j, ref.att_j, ref.taps).eval()
    b = DecB(mods[-1]).eval()
    entries = [("v8_16k", ref, ref.input_names(), ref.output_names()),
               ("dva", a, a.input_names(), a.output_names()),
               ("dvb", b, ["o", "x"], ["y"])]
    fname = f"chunk_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
    mb = B.save_program(entries, out / fname)
    info = {"file": fname, "layers": [layers[0], layers[-1]], "entries": [x[0] for x in entries], "gdn_j": ref.gdn_j,
            "att_j": ref.att_j, "taps": ref.taps, "mb": round(mb), "split": V.SPLIT,
            "io": {n: {"in": i, "out": o} for n, _, i, o in entries}}
    (out / f"{fname}.info.json").write_text(json.dumps(info, indent=1))
    print(f"{fname}: {len(entries)} entries, {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
