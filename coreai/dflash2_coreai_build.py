"""Core AI build of the DFlash2 drafter (5 layers, dynamic gated convs, sliding-window attention over a W-slot ring):
the Core ML graph of scripts/dflash2_ane_drafter.py (same fp16 range tricks: max-normalized RMSNorm, residual carried
/ RESID_SCALE from layer 0's attention add, LUT refold LUT_F, host feature scale FEAT_SCALE) as PyTorch modules for
coreai-torch, with two changes for Core AI:
- The ring caches are host-owned inputs kc<i> / vc<i> (8 kv heads, W, 128), not states (in-graph state writes put the
  graph on the GPU). The context rows committed since the last call go in as target features; the entry returns their
  K / V (k_new<i> / v_new<i>) and the host writes them into slot = position % W after the call. The query block sees
  them in the same call through extra attention columns: scores over [ring (W) | new context rows (R) | block (T)],
  additive mask (T, W + R + T) from the host (ring slots about to be overwritten hold positions >= W back: already
  outside the 2048 window).
- SiLU in the tanh form (the ANE's native silu has ~1e-3 absolute error near 0).
Entries (one package, shared weights): draft (R = 8 context rows + the 8-row block [anchor, mask x 7] -> K / V rows,
logits of rows 1..7, selector projection hp, final hidden) and ctx64 (64 context rows -> K / V rows).
    .venv/bin/python dflash2_coreai_build.py            # -> $OUT/dflash2_lut4_gptq.aimodel + .json
env: DRAFTER (checkpoint dir), DRAFT_EXPORT (dflash2_quant.py export), HEAD_EXPORT (lm_head LUT), MODEL (target, for
the mask-token embedding), OUT, NO_HEAD=1 (body only), SPLIT=1 (one package per layer + fc / head packages: fallback
if MPSGraph rejects several attention layers in one program)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen38_coreai_build as B  # noqa: E402  (QConv, save_program, silu: exact LUT palettization, tanh SiLU)

E = os.path.expanduser
DRAFTER = Path(E(os.environ.get("DRAFTER", "~/Models/dflash2/checkpoint")))
DRAFT_EXPORT = Path(E(os.environ.get("DRAFT_EXPORT", "~/Models/dflash2/export/drafter_lut4_gptq_q7_cal")))
# The drafter drafts with the TARGET's lm_head: default to the head of the target export being served (TARGET_EXPORT);
# an explicit HEAD_EXPORT overrides. (The old default, full_mix25_mixer4_head4, is a much coarser LUT4 than the
# current exports' heads and silently mismatched the target on 2026-09-28.)
TARGET_EXPORT = Path(E(os.environ.get("TARGET_EXPORT", "~/Models/vq27b/export/mix25in_mixr_lr64mix")))
HEAD_EXPORT = Path(E(os.environ.get("HEAD_EXPORT", str(TARGET_EXPORT / "lm_head.safetensors"))))
MODEL = Path(E(os.environ.get("MODEL", "~/Models/Qwen3.8-27B")))
OUT = Path(E(os.environ.get("OUT", "~/Models/dflash2/coreai")))
NO_HEAD = os.environ.get("NO_HEAD") == "1"


def rot1_fix_b_check():
    """rot1 fix B (next/rot1/RUNBOOK.md R0.8): the drafter keeps its unrotated weights, so it needs an unrotated LM head
    (no folded final-norm gain, no R) and the original checkpoint's mask-token row. A head export or MODEL carrying a
    rot1 basis.json would silently mismatch the drafter's hidden state; refuse it."""
    for what, d in (("HEAD_EXPORT", HEAD_EXPORT.parent), ("MODEL", MODEL)):
        f = Path(d) / "basis.json"
        if f.exists():
            b = json.loads(f.read_text())
            raise SystemExit(f"rot1 fix B: {what} {d} belongs to rot1 basis {b.get('name')} "
                             f"({str(b.get('basis_id'))[:12]}); the drafter needs the base export's unrotated head "
                             f"and the original small checkpoint")
W, T, R, RP = 2048, 8, 8, 64
RESID_SCALE = 256.0
LUT_F = {"down": 1 / 64, "default": 1 / 16}
FEAT_SCALE = 0.125     # applied by the host to the target features (fc -> RMSNorm: scale invariant)
# layer 0's input RMSNorm output of the 7 mask-token rows is scaled by MASK_SCALE: +10% accepted drafts on the traces
# (fp32 reference sweep 2026-09-28, both halves; 0.65-0.85 all close). 1.0 = the checkpoint's math.
MASK_SCALE = float(os.environ.get("MASK_SCALE", "0.7"))
HEAD_PARTS = 8
DBG = os.environ.get("DBG") == "1"   # debug: extra outputs d_* (layer 0 step by step, every layer's residual)
TAPS: list = []
L0_TAPS = ["d_n0", "d_dyn0", "d_a0", "d_q0", "d_kb0", "d_o0", "d_op0", "d_xa0", "d_m0", "d_y0"]
f16 = torch.float16
torch.set_grad_enabled(False)


def load_weights(cfg):
    """{name: np.ndarray}: export tensors (lut / idx / scale, int8 / scale) with the Core ML build's foldings, plus the
    checkpoint's small tensors (norms, conv base kernels, selector projection)."""
    w = {}
    with safe_open(DRAFT_EXPORT / "drafter_quant.safetensors", "np") as f:
        t = {k: f.get_tensor(k) for k in f.keys()}
    for base in sorted({k.rsplit(".", 1)[0] for k in t}):
        name = base[:-len(".weight")] if base.endswith(".weight") else base
        if f"{base}.int8" in t:
            w[f"{name}/int8"], w[f"{name}/scale"] = t[f"{base}.int8"], t[f"{base}.scale"].astype(np.float32)
            continue
        lut, idx, sc = t[f"{base}.lut"].astype(np.float32), t[f"{base}.idx"], t[f"{base}.scale"].astype(np.float32)
        if name.endswith(("o_proj", "down_proj")):  # branch outputs carry 1 / RESID_SCALE
            sc = sc / RESID_SCALE
        fold = LUT_F["down" if name.endswith("down_proj") else "default"]
        w[f"{name}/lut"], w[f"{name}/idx"], w[f"{name}/scale"] = (lut * fold).astype(np.float16), idx, (sc / fold)
    # the drafter checkpoint's small tensors: the full upstream model.safetensors, or small.safetensors holding only
    # them (scripts/qwen38_small_checkpoint.py --drafter; the quantized-export repository's drafter/)
    small = DRAFTER / "model.safetensors"
    small = small if small.exists() else DRAFTER / "small.safetensors"
    with safe_open(small, "pt") as f:
        for k in f.keys():
            if k.endswith(("norm.weight", "base_kernel")) or k == "candidate_selector.hidden_projection.weight":
                w[k] = f.get_tensor(k).float().numpy()
    for k in list(w):
        if k.endswith("/scale"):
            w[k] = w[k].astype(np.float16)
    return w


MASK_ROW = "dflash.mask_token_embedding"  # the target embedding row of the mask token (a small checkpoint's copy)


def mask_embedding(cfg) -> np.ndarray:
    tok = cfg["dflash_config"]["mask_token_id"]
    name = "model.language_model.embed_tokens.weight"
    wmap = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    if name not in wmap and MASK_ROW in wmap:
        with safe_open(MODEL / wmap[MASK_ROW], "pt") as f:
            return f.get_tensor(MASK_ROW)[0].float().numpy()
    with safe_open(MODEL / wmap[name], "pt") as f:
        return f.get_slice(name)[tok:tok + 1][0].float().numpy()


def qconv(w, name, dense=None):
    if dense is not None:
        return B.QConv({"x/dense": dense.astype(np.float16)}, "x")
    sub = {k.replace(name + "/", "x/"): v for k, v in w.items() if k.startswith(name + "/")}
    return B.QConv(sub, "x")


def rms_robust(x, w, dim, eps=1e-6):
    """x * rsqrt(mean(x^2) + eps) * w computed on xs = x / m, m = max|x| clamped to >= 1e-3, eps as eps / m^2 (the
    target builder's rms_hidden form): no fp16 overflow for massive activations, eps exact for tiny rows (the mask-token
    embedding, rms ~3e-3), zero rows stay zero. The Core ML form (x * 1 / (max + 1e-4), eps as inv * inv * eps + 1e-7)
    normalized those tiny rows ~35% too small on the ANE under Core AI (as if eps were ~1.5e-5)."""
    m = x.abs().amax(dim, keepdim=True).clamp_min(1e-3)
    xs = x / m
    return xs * torch.rsqrt((xs * xs).mean(dim, keepdim=True) + eps / (m * m)) * w


def rope(t, cos, sin):
    """t (heads, rows, 128), cos / sin (rows, 128): NeoX half split."""
    h = t.shape[-1] // 2
    return t * cos + torch.cat([-t[..., h:], t[..., :h]], -1) * sin


def heads(x, n, rows, hd=128):
    """(1, n * hd, 1, rows) -> (n, rows, hd)."""
    return x.reshape(n, hd, rows).permute(0, 2, 1)


def gconv(u, coef, base, rows):
    """Dynamic 2-tap grouped conv within the block: u (1, 5120, 1, rows), coef (2 taps, 320, 1, rows), base (2, 5120).
    out[t] = (base0 + coef0[t]) u[t] + (base1 + coef1[t]) u[t - 1], u[-1] = 0 (16 channels per group)."""
    u3 = u.reshape(320, 16, rows)
    b = base.reshape(2, 320, 16, 1)
    shifted = torch.cat([torch.zeros(320, 16, 1, dtype=u.dtype), u3[..., :rows - 1]], 2)
    out = (coef[0] + b[0]) * u3 + (coef[1] + b[1]) * shifted
    return out.reshape(1, 5120, 1, rows)


class Layer(nn.Module):
    def __init__(self, cfg, w, i):
        super().__init__()
        p = f"layers.{i}."
        self.i = i
        self.nh, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, n, qconv(w, p + "self_attn." + n))
        for n in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, n, qconv(w, p + "mlp." + n))
        self.akp, self.mkp = qconv(w, p + "attention_conv.kernel_projection"), qconv(w, p + "mlp_conv.kernel_projection")
        t = lambda k, *s: torch.from_numpy(w[p + k].astype(np.float16)).reshape(*s)  # noqa: E731
        self.register_buffer("in_ln", t("input_layernorm.weight", 1, -1, 1, 1))
        self.register_buffer("post_ln", t("post_attention_layernorm.weight", 1, -1, 1, 1))
        self.register_buffer("q_norm", t("self_attn.q_norm.weight", -1))
        self.register_buffer("k_norm", t("self_attn.k_norm.weight", -1))
        self.register_buffer("abase", t("attention_conv.base_kernel", 2, 2, 5120))
        self.register_buffer("mbase", t("mlp_conv.base_kernel", 2, 2, 5120))
        rs = torch.full((1, 1, 1, T), MASK_SCALE, dtype=torch.float16)
        rs[..., 0] = 1.0                                        # row 0 = the anchor token
        self.register_buffer("rowscale", rs)

    def ctx_kv(self, fused, cos, sin, rows):
        k = rope(rms_robust(heads(self.k_proj(fused), self.nkv, rows), self.k_norm, -1), cos, sin)
        return k, heads(self.v_proj(fused), self.nkv, rows)

    def forward(self, x, kc, vc, kx, vx, q_cos, q_sin, mask):
        nh, nkv, hd = self.nh, self.nkv, self.hd
        grp = nh // nkv
        n = rms_robust(x, self.in_ln, 1)
        if self.i == 0 and MASK_SCALE != 1.0:
            n = n * self.rowscale
        dyn = self.akp(n).reshape(2, 2, 320, 1, T)
        a = gconv(n, dyn[0], self.abase[0], T)
        q = rope(rms_robust(heads(self.q_proj(a), nh, T), self.q_norm, -1), q_cos, q_sin)
        kb = rope(rms_robust(heads(self.k_proj(a), nkv, T), self.k_norm, -1), q_cos, q_sin)
        vb = heads(self.v_proj(a), nkv, T)
        if DBG and self.i == 0:
            TAPS.extend([n, dyn.reshape(1, -1, 1, T), a, q, kb])
        qg = q.reshape(nkv, grp * T, hd)
        s = torch.cat([qg @ kc.transpose(1, 2), qg @ kx.transpose(1, 2), qg @ kb.transpose(1, 2)], 2) * hd ** -0.5
        nr = kx.shape[1]
        s = s.reshape(nkv, grp, T, W + nr + T) + mask
        pr = torch.softmax(s, -1).reshape(nkv, grp * T, W + nr + T)
        o = pr[..., :W] @ vc + pr[..., W:W + nr] @ vx + pr[..., W + nr:] @ vb
        o = o.reshape(nh, T, hd).permute(0, 2, 1).reshape(1, nh * hd, 1, T)
        if DBG and self.i == 0:
            TAPS.append(o)
        o = self.o_proj(o)                                     # scaled by 1 / RESID_SCALE (folded)
        if self.i == 0:
            x = x * (1 / RESID_SCALE)
        x = x + gconv(o, dyn[1], self.abase[1], T)
        if DBG and self.i == 0:
            TAPS.extend([o, x])
        n = rms_robust(x, self.post_ln, 1)
        dyn = self.mkp(n).reshape(2, 2, 320, 1, T)
        m = gconv(n, dyn[0], self.mbase[0], T)
        y = self.down_proj(B.silu(self.gate_proj(m), "tanh") * self.up_proj(m))  # scaled by 1 / RESID_SCALE
        if DBG and self.i == 0:
            TAPS.extend([m, y])
        x = x + gconv(y, dyn[1], self.mbase[1], T)
        if DBG:
            TAPS.append(x)
        return x


class Core(nn.Module):
    """All drafter weights; the entries below share this module (one weight copy in the package)."""

    def __init__(self, cfg, w, mask_emb):
        super().__init__()
        self.L = cfg["num_hidden_layers"]
        self.fc = qconv(w, "fc")
        self.register_buffer("hidden_norm", torch.from_numpy(w["hidden_norm.weight"].astype(np.float16)).view(1, -1, 1, 1))
        self.layers = nn.ModuleList(Layer(cfg, w, i) for i in range(self.L))
        self.register_buffer("norm", torch.from_numpy(w["norm.weight"].astype(np.float16)).view(1, -1, 1, 1))
        self.sel = qconv(w, None, dense=w["candidate_selector.hidden_projection.weight"])
        self.register_buffer("mask_emb", torch.from_numpy(mask_emb.astype(np.float16)).view(1, -1, 1, 1))
        self.head = None
        if not NO_HEAD:
            with safe_open(HEAD_EXPORT, "np") as f:
                lut, idx, sc = f.get_tensor("lm_head.lut"), f.get_tensor("lm_head.idx"), f.get_tensor("lm_head.scale")
            v = idx.shape[0]
            step = -(-v // HEAD_PARTS)
            self.head = nn.ModuleList(B.QConv({"h/lut": lut.astype(np.float16), "h/idx": idx[a:min(a + step, v)],
                                               "h/scale": sc[a:min(a + step, v)].astype(np.float16)}, "h")
                                      for a in range(0, v, step))

    def context(self, feat, cos, sin, rows):
        fused = rms_robust(self.fc(feat), self.hidden_norm, 1)
        return [l.ctx_kv(fused, cos, sin, rows) for l in self.layers]


class DraftEntry(nn.Module):
    """draft: R new context rows + the 8-row block. Inputs feat (1, 25600, 1, R), ctx_cos / ctx_sin (R, 128), anchor
    (1, 5120, 1, 1), q_cos / q_sin (T, 128), mask (T, W + R + T), kc<i> / vc<i> (nkv, W, 128). Outputs k_new<i> /
    v_new<i> (nkv, R, 128), [logits (7, V)], hp (1, 256, 1, T), hidden (1, 5120, 1, T)."""

    def __init__(self, core):
        super().__init__()
        self.core = core

    def names(self):
        L = self.core.L
        ins = ["feat", "ctx_cos", "ctx_sin", "anchor", "q_cos", "q_sin", "mask"] + \
              [f"{s}{i}" for i in range(L) for s in ("kc", "vc")]
        outs = [f"{s}_new{i}" for i in range(L) for s in ("k", "v")] + ([] if NO_HEAD else ["logits"]) + ["hp", "hidden"]
        if DBG:
            outs += L0_TAPS + [f"d_xm{i}" for i in range(L)]
        return ins, outs

    def example(self):
        L = self.core.L
        z = lambda *s: torch.zeros(*s, dtype=f16)  # noqa: E731
        return (z(1, 25600, 1, R), z(R, 128), z(R, 128), z(1, 5120, 1, 1), z(T, 128), z(T, 128), z(T, W + R + T),
                *[z(8, W, 128) for _ in range(2 * L)])

    def forward(self, feat, ctx_cos, ctx_sin, anchor, q_cos, q_sin, mask, *ring):
        c = self.core
        TAPS.clear()
        kv = c.context(feat, ctx_cos, ctx_sin, R)
        x = torch.cat([anchor, c.mask_emb.expand(1, 5120, 1, T - 1)], 3)
        for i, l in enumerate(c.layers):
            x = l(x, ring[2 * i], ring[2 * i + 1], kv[i][0], kv[i][1], q_cos, q_sin, mask)
        h = rms_robust(x, c.norm, 1)
        hp = c.sel(h)
        outs = [t for k, v in kv for t in (k, v)]
        if c.head is not None:
            h7 = h[..., 1:]
            outs.append(torch.cat([p(h7).reshape(-1, T - 1).transpose(0, 1) for p in c.head], 1))
        return (*outs, hp, h, *TAPS)


class CtxEntry(nn.Module):
    """ctx64: feat (1, 25600, 1, RP), ctx_cos / ctx_sin (RP, 128) -> k_new<i> / v_new<i> (nkv, RP, 128)."""

    def __init__(self, core):
        super().__init__()
        self.core = core

    def names(self):
        L = self.core.L
        return ["feat", "ctx_cos", "ctx_sin"], [f"{s}_new{i}" for i in range(L) for s in ("k", "v")]

    def example(self):
        z = lambda *s: torch.zeros(*s, dtype=f16)  # noqa: E731
        return z(1, 25600, 1, RP), z(RP, 128), z(RP, 128)

    def forward(self, feat, ctx_cos, ctx_sin):
        return tuple(t for k, v in self.core.context(feat, ctx_cos, ctx_sin, RP) for t in (k, v))


def main():
    rot1_fix_b_check()
    cfg = json.loads((DRAFTER / "config.json").read_text())
    assert cfg["sliding_window"] == W and cfg["dflash_config"]["block_size"] == T
    t0 = time.time()
    w = load_weights(cfg)
    core = Core(cfg, w, mask_embedding(cfg)).eval().to(f16)
    del w
    print(f"modules built ({time.time() - t0:.0f}s)", flush=True)
    tag = "dflash2_lut4_gptq" + ("_nohead" if NO_HEAD else "") + ("_dbg" if DBG else "")
    entries = []
    for name, e in (("draft", DraftEntry(core)), ("ctx64", CtxEntry(core))):
        ins, outs = e.names()
        entries.append((name, e, ins, outs))
    out = OUT / f"{tag}.aimodel"
    mb = B.save_program(entries, out)
    (OUT / f"{tag}.json").write_text(json.dumps({
        "W": W, "T": T, "R": R, "RP": RP, "resid_scale": RESID_SCALE, "lut_f": LUT_F, "feat_scale": FEAT_SCALE,
        "export": str(DRAFT_EXPORT), "head_export": None if NO_HEAD else str(HEAD_EXPORT), "mask_scale": MASK_SCALE,
        "target_export": str(TARGET_EXPORT), "mb": round(mb),
        "entries": {n: {"inputs": i, "outputs": o} for n, _, i, o in entries}}, indent=1))
    print(f"built {out} ({mb:.0f} MB, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
