"""V8 ANE side, DeltaNet mixers: for every linear-attention layer, the production mixer exactly as the target build's
chunks run it in prefill (qwen38_coreai_build GDNW.prefill: RMSNorm x (1+w), rotation, LUT qkv / z, dense a / b, causal
conv1d, chunked gated delta rule with C_SUB sub-chunks, gated RMSNorm, rotation, LUT out) plus the residual add, with
an explicit state: conv rows (3, cdim) = the last three conv inputs, rec (nv, dk, dv) in the build's GDN_SV-scaled
layout (= the bridge's conv<j>[:3] / rec<j>). No pending rows (nothing uncommitted in a prefill). Entries per layer:
mix_L<i> (x, conv, rec) -> (y, conv_out, rec_out) and mix2_L<i> (two consecutive 64-row blocks, the state carried
from the first to the second inside one prediction). Three layers per package (the GDN layers of one target chunk).
    EXPORT_DIR=<release export> BUILDPY build_v8_mix.py OUT_DIR SEG2_BUILD [first_chunk last_chunk]"""
import gc, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn

V02 = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02"
sys.path[:0] = [V02 + "/coreai", V02 + "/scripts"]
import qwen38_coreai_build as B  # noqa: E402
import qwen38_ane_model as M  # noqa: E402

T = 64
import os as _os
if _os.environ.get("MIX_CSUB"):   # probe: DeltaNet sub-chunk rows inside the mixer (target build: 8)
    B.C_SUB = int(_os.environ["MIX_CSUB"])


def mix_block(layer, x, conv3, rec):
    """One 64-row prefill block of a DeltaNet layer's mixer: GDNW.prefill with no pending rows and every row valid."""
    mix = layer.mix
    h = B.rms_hidden(x, layer.ln1)
    qkv, z, b, a = mix.proj(h, T)
    rows = torch.cat([conv3, qkv.transpose(0, 1)], 0)                                      # (T + 3, cdim)
    qh, kh, vh = mix.qkv_heads(rows, T)
    beta, g = torch.sigmoid(b), B.softplus(a + mix.dt) * mix.neg_a
    s = rec
    nv, dk, dv = B.nv, B.dk, B.dv
    NB, C = T // B.C_SUB, B.C_SUB
    l_inc, l_str = B.tri(C, False), B.tri(C, True)
    q4, k4, v4 = qh.reshape(nv, NB, C, dk), kh.reshape(nv, NB, C, dk), vh.reshape(nv, NB, C, dv)
    b4, g4 = beta.reshape(nv, NB, C, 1), g.reshape(nv, NB, C, 1)
    cum = (g4.reshape(nv, NB, 1, C) @ l_inc.T).reshape(nv, NB, C, 1)
    pair = torch.exp(torch.clamp(cum - cum.reshape(nv, NB, 1, C), max=0)) * l_inc
    kb, vb = k4 * b4, v4 * b4
    n = (kb @ k4.transpose(-1, -2)) * (pair * l_str)
    xs = B.tri_solve(n, torch.cat([vb, kb * torch.exp(cum)], -1), C)
    u, wk = xs[..., :dv], xs[..., dv:]
    total = cum[:, :, C - 1:C, :]
    qd, kdc = q4 * torch.exp(cum), k4 * torch.exp(total - cum)
    intra = (q4 @ k4.transpose(-1, -2)) * pair
    outs = []
    wq = torch.cat([wk, qd], 2) if B.GDN_FAST else None
    for bi in range(NB):
        if B.GDN_FAST:
            ws = wq[:, bi] @ s
            vn = u[:, bi] - ws[:, :C]
            outs.append(ws[:, C:] + intra[:, bi] @ vn)
        else:
            vn = u[:, bi] - wk[:, bi] @ s
            outs.append(qd[:, bi] @ s + intra[:, bi] @ vn)
        s = s * torch.exp(total[:, bi]) + kdc[:, bi].transpose(1, 2) @ vn
    y = mix.finish(torch.cat(outs, 1), z, T)
    return x + y, rows[T:T + 3], s


class Mix(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def example(self):
        f = torch.float16
        return (torch.randn(1, B.hid, 1, T, dtype=f) * 0.3, torch.zeros(3, B.cdim, dtype=f), torch.zeros(B.nv, B.dk, B.dv, dtype=f))

    def forward(self, x, conv, rec):
        return mix_block(self.layer, x, conv, rec)


class Mix2(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def example(self):
        f = torch.float16
        return (torch.randn(1, B.hid, 1, T, dtype=f) * 0.3, torch.randn(1, B.hid, 1, T, dtype=f) * 0.3,
                torch.zeros(3, B.cdim, dtype=f), torch.zeros(B.nv, B.dk, B.dv, dtype=f))

    def forward(self, xa, xb, conv, rec):
        ya, c1, s1 = mix_block(self.layer, xa, conv, rec)
        yb, c2, s2 = mix_block(self.layer, xb, c1, s1)
        return ya, yb, c2, s2


def main():
    out, seg2 = Path(sys.argv[1]), Path(sys.argv[2])
    c0, c1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 15)
    out.mkdir(parents=True, exist_ok=True)
    mp = out / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {
        "version": "v8_mix1", "T": T, "kind": "gdn_mixer_only", "export": str(M.EXPORT_DIR), "seg2": str(seg2),
        "numerics": {"GDN_SV": B.GDN_SV, "GDN_SQ": B.GDN_SQ, "C_SUB": B.C_SUB, "GDN_FAST": bool(B.GDN_FAST), "SILU": B.SILU},
        "chunks": [], "head": {"file": str(seg2 / "head_T8.aimodel")}}
    ck = M.Checkpoint()
    done = {c["file"] for c in man["chunks"]}
    for ci in range(c0, c1 + 1):
        layers = [i for i in range(4 * ci, 4 * ci + 4) if B.CFG["layer_types"][i] == "linear_attention"]
        name = f"mix_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
        if name in done and (out / name).exists():
            continue
        t0 = time.time()
        B.KNOWN_LUTS.clear()
        W = {}
        for i in layers:
            W.update(B.layer_arrays(ck, i))
        mods = [B.LayerW(W, i).eval().to(torch.float16) for i in layers]
        del W
        gc.collect()
        for m in mods:  # keep only what the mixer uses
            m.gate = m.up = m.down = m.rin = m.rmid = None
        entries = [(f"mix_L{m.i}", Mix(m).eval(), ["x", "conv", "rec"], ["y", "conv_out", "rec_out"]) for m in mods]
        entries += [(f"mix2_L{m.i}", Mix2(m).eval(), ["xa", "xb", "conv", "rec"], ["ya", "yb", "conv_out", "rec_out"]) for m in mods]
        mb = B.save_program(entries, out / name)
        man["chunks"] = sorted([c for c in man["chunks"] if c["file"] != name] +
                               [{"file": name, "layers": layers, "entries": [e[0] for e in entries], "mb": round(mb),
                                 "numerics": B.rot_numerics(layers)}], key=lambda c: c["layers"][0])
        mp.write_text(json.dumps(man, indent=1))
        print(f"{name}: {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)
        del mods, entries
        gc.collect()
    print("manifest", mp)


if __name__ == "__main__":
    main()
