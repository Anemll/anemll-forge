"""Gated DeltaNet core on the M6 ANE: the release formulation against exact rewrites, timed through the Swift bridge.

The program holds L GDN layer cores (real conv taps, A_log, dt_bias and norm weights of layers 0..L-1) between their
projections: inputs are the projection outputs in the chunk's (1, C, 1, T) layout, outputs the out_proj input, the
conv rows, the committed state and the pending rows, as in `qwen38_coreai_build.GDNW`. `ref` is that class's own
`verify` / `prefill`; every other variant must match it on the host in FP32 before it is timed, and its ANE output is
compared with the FP32 reference.

    <coreai venv>/bin/python scripts/m6_gdn_bench.py check  --variants ref,fast         # host FP32 equivalence
    <coreai venv>/bin/python scripts/m6_gdn_bench.py build  --variants ref,fast --out DIR
    <coreai venv>/bin/python scripts/m6_gdn_bench.py time   --out DIR                   # ANE ms + error vs FP32
    <coremltools venv>/bin/python scripts/m6_gdn_bench.py coreml --out DIR              # .mlpackage for anemll-profile
Env: MODEL (checkpoint with the GDN small tensors), MPSGRAPH_ANE_BONDED_COMPILE_MODE (default: the SoC policy, 2 on M6, 1 on M5)."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "scripts"))
import ane_compile_mode  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)  # the SoC's bonded compile mode (M6: 2, M5: 1) unless set explicitly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))
sys.path.insert(0, str(ROOT / "coreai" / "swift_bridge"))
sys.path.insert(0, str(ROOT / "scripts"))
import coreai_bridge as B  # noqa: E402
import inspect_coreai_cache as IC  # noqa: E402
import m6_entry_sweep as S  # noqa: E402
import qwen38_coreai_build as Bld  # noqa: E402

nv, nk, dk, dv, P = Bld.nv, Bld.nk, Bld.dk, Bld.dv, Bld.P
kd, vd, cdim = Bld.kd, Bld.vd, Bld.cdim
f16 = torch.float16


# ---- variants ------------------------------------------------------------------------------------------------------
# A variant may replace GDNW.verify / GDNW.prefill on the instance and override builder globals (C_SUB, fwd_sub)
# while the program is traced or run on the host. timing_only variants drop math on purpose (cost attribution).
def _heads16(conv, T, gdn_sq):
    """q, k at the 16 key heads (l2-normalized there, as the norm commutes with the head repeat), v at 48."""
    qq, kk, vv = conv[:kd], conv[kd:2 * kd], conv[2 * kd:]

    def l2n(t, s):
        t = t.reshape(nk, dk, T).permute(0, 2, 1)
        return t * torch.rsqrt((t * t).sum(-1, keepdim=True) + 1e-6) * s
    vh = vv.reshape(nv, dv, T).permute(0, 2, 1)
    vh = vh * Bld.GDN_SV if Bld.GDN_SV != 1 else vh
    return l2n(qq, dk ** -0.5 * gdn_sq), l2n(kk, 1.0), vh


def _rep(t, T):
    return t.reshape(nk, 1, T, t.shape[-1]).repeat(1, nv // nk, 1, 1).reshape(nv, T, t.shape[-1])


def verify_fast(self, h, conv_rows, conv_sel, rec, pend, commit, commit_last, T):
    """Same math as GDNW.verify: l2 norms at 16 heads before the repeat, wk @ S and q @ S as one matmul."""
    qkv, z, b, a = self.proj(h, T)
    rows = torch.cat([conv_sel @ conv_rows, qkv.transpose(0, 1)], 0)
    conv = rows[0:T] * self.cw[0:1] + rows[1:T + 1] * self.cw[1:2] + rows[2:T + 2] * self.cw[2:3] + rows[3:T + 3] * self.cw[3:4]
    conv = Bld.silu(conv).transpose(0, 1)
    q16, k16, vh = _heads16(conv, T, Bld.GDN_SQ)
    qh, kh = _rep(q16, T), _rep(k16, T)
    beta, g = torch.sigmoid(b), Bld.softplus(a + self.dt) * self.neg_a
    s1 = self.commit_pending(rec, pend, commit, commit_last)
    l_inc, l_str = Bld.tri(T, False), Bld.tri(T, True)
    cum = (g.reshape(nv, 1, T) @ l_inc.T).reshape(nv, T, 1)
    pair = torch.exp(torch.clamp(cum - cum.reshape(nv, 1, T), max=0)) * l_inc
    kb, vb = kh * beta, vh * beta
    n = (kb @ kh.transpose(1, 2)) * (pair * l_str)
    x = Bld.fwd_sub(n, torch.cat([vb, kb * torch.exp(cum)], -1), T)
    u, wk = x[..., :dv], x[..., dv:]
    crow = torch.cat([cum.reshape(nv, 1, T), torch.zeros(nv, 1, dv - T, dtype=cum.dtype)], 2)
    pend_out = torch.cat([kh, u, wk, crow], 1)
    ws = torch.cat([wk, qh * torch.exp(cum)], 1) @ s1                                   # (nv, 2T, dv)
    vn = u - ws[:, :T]
    o = ws[:, T:] + ((qh @ kh.transpose(1, 2)) * pair) @ vn
    return self.finish(o, z, T), rows, s1, pend_out


def prefill_noloop(self, h, conv_rows, conv_sel, conv_sel_out, rec, pend, commit, commit_last, valid, T):
    """Timing only: GDNW.prefill with every sub-chunk reading the entry state (no sequential state chain)."""
    qkv, z, b, a = self.proj(h, T)
    rows = torch.cat([conv_sel @ conv_rows, qkv.transpose(0, 1)], 0)
    qh, kh, vh = self.qkv_heads(rows, T)
    v1 = valid.reshape(1, T, 1)
    beta, g = torch.sigmoid(b) * v1, Bld.softplus(a + self.dt) * self.neg_a * v1
    s = self.commit_pending(rec, pend, commit, commit_last)
    NB, C = T // Bld.C_SUB, Bld.C_SUB
    l_inc, l_str = Bld.tri(C, False), Bld.tri(C, True)
    q4, k4, v4 = qh.reshape(nv, NB, C, dk), kh.reshape(nv, NB, C, dk), vh.reshape(nv, NB, C, dv)
    b4, g4 = beta.reshape(nv, NB, C, 1), g.reshape(nv, NB, C, 1)
    cum = (g4.reshape(nv, NB, 1, C) @ l_inc.T).reshape(nv, NB, C, 1)
    pair = torch.exp(torch.clamp(cum - cum.reshape(nv, NB, 1, C), max=0)) * l_inc
    kb, vb = k4 * b4, v4 * b4
    n = (kb @ k4.transpose(-1, -2)) * (pair * l_str)
    x = Bld.tri_solve(n, torch.cat([vb, kb * torch.exp(cum)], -1), C)
    u, wk = x[..., :dv], x[..., dv:]
    total = cum[:, :, C - 1:C, :]
    qd, kdc = q4 * torch.exp(cum), k4 * torch.exp(total - cum)
    intra = (q4 @ k4.transpose(-1, -2)) * pair
    s4 = s.reshape(nv, 1, dk, dv)
    vn = u - wk @ s4
    o = (qd @ s4 + intra @ vn).reshape(nv, T, dv)
    s = s * torch.exp(total[:, -1]) + kdc[:, -1].transpose(1, 2) @ vn[:, -1]
    conv_out = torch.cat([conv_sel_out @ rows, torch.zeros(P, cdim, dtype=rows.dtype)], 0)
    return self.finish(o, z, T), conv_out, s, pend * 0


def _eye_like(n):
    rows = n.shape[-1]
    return torch.eye(rows, dtype=n.dtype).expand(n.shape)


def fwd_sub_neumann(n, rhs, rows):
    """(I + N) X = rhs with N strictly lower (nilpotent): X = (I - N)(I + N^2)(I + N^4)... rhs, applied factor by
    factor (matmuls on rhs) - log2(rows) steps instead of rows - 1 dependent row updates."""
    x = rhs - n @ rhs
    p, k = n, 2
    while k < rows:
        p = p @ p
        x = x + p @ x
        k *= 2
    return x


def fwd_sub_inverse(n, rhs, rows):
    """Same identity, the inverse formed first: T = (I - N)(I + N^2)(I + N^4)..., then one T @ rhs."""
    eye = _eye_like(n)
    t, p, k = eye - n, n, 2
    while k < rows:
        p = p @ p
        t = t @ (eye + p)
        k *= 2
    return t @ rhs


def _bmm_small(a, b):
    """a @ b for small square blocks as broadcast multiply + reduce (no matmul between computed tensors)."""
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)


def fwd_sub_inverse_ew(n, rhs, rows):
    """fwd_sub_inverse with the rows x rows products as broadcast multiply-reduce; one matmul with rhs."""
    eye = _eye_like(n)
    t, p, k = eye - n, n, 2
    while k < rows:
        p = _bmm_small(p, p)
        t = _bmm_small(t, eye + p)
        k *= 2
    return t @ rhs


def _qkv_heads_nt(self, rows, T):
    """GDNW.qkv_heads with its layout flips replaced by reshapes (timing only)."""
    conv = rows[0:T] * self.cw[0:1] + rows[1:T + 1] * self.cw[1:2] + rows[2:T + 2] * self.cw[2:3] + rows[3:T + 3] * self.cw[3:4]
    conv = Bld.silu(conv).reshape(cdim, T)
    qq, kk, vv = conv[:kd], conv[kd:2 * kd], conv[2 * kd:]

    def heads(t):
        return t.reshape(nk, 1, T, dk).repeat(1, nv // nk, 1, 1).reshape(nv, T, dk)

    def l2n(t, s_):
        return t * torch.rsqrt((t * t).sum(-1, keepdim=True) + 1e-6) * s_
    vh = vv.reshape(nv, T, dv) * Bld.GDN_SV
    return l2n(heads(qq), dk ** -0.5 * Bld.GDN_SQ), l2n(heads(kk), 1.0), vh


def _finish_nt(self, o, z, T):
    o = Bld.rms_last(o, self.normw, Bld.EPS * (Bld.GDN_SQ * Bld.GDN_SV) ** 2) * Bld.silu(z)
    return self.out(o.reshape(1, vd, 1, T))


def verify_nt(self, h, conv_rows, conv_sel, rec, pend, commit, commit_last, T):
    """Timing only: GDNW.verify with the q/k/v/z/conv/output layout flips replaced by reshapes."""
    qkv, z, b, a = h
    z, b, a = z.reshape(nv, T, dv), b.reshape(nv, T, 1), a.reshape(nv, T, 1)
    rows = torch.cat([conv_sel @ conv_rows, qkv.reshape(T, cdim)], 0)
    qh, kh, vh = _qkv_heads_nt(self, rows, T)
    beta, g = torch.sigmoid(b), Bld.softplus(a + self.dt) * self.neg_a
    s1 = self.commit_pending(rec, pend, commit, commit_last)
    l_inc, l_str = Bld.tri(T, False), Bld.tri(T, True)
    cum = (g.reshape(nv, 1, T) @ l_inc.T).reshape(nv, T, 1)
    pair = torch.exp(torch.clamp(cum - cum.reshape(nv, 1, T), max=0)) * l_inc
    kb, vb = kh * beta, vh * beta
    n = (kb @ kh.transpose(1, 2)) * (pair * l_str)
    x = Bld.tri_solve(n, torch.cat([vb, kb * torch.exp(cum)], -1), T)
    u, wk = x[..., :dv], x[..., dv:]
    crow = torch.cat([cum.reshape(nv, 1, T), torch.zeros(nv, 1, dv - T, dtype=cum.dtype)], 2)
    pend_out = torch.cat([kh, u, wk, crow], 1)
    vn = u - wk @ s1
    o = (qh * torch.exp(cum)) @ s1 + ((qh @ kh.transpose(1, 2)) * pair) @ vn
    return _finish_nt(self, o, z, T), rows, s1, pend_out


def prefill_nt(self, h, conv_rows, conv_sel, conv_sel_out, rec, pend, commit, commit_last, valid, T):
    """Timing only: GDNW.prefill with the layout flips replaced by reshapes."""
    qkv, z, b, a = h
    z, b, a = z.reshape(nv, T, dv), b.reshape(nv, T, 1), a.reshape(nv, T, 1)
    rows = torch.cat([conv_sel @ conv_rows, qkv.reshape(T, cdim)], 0)
    qh, kh, vh = _qkv_heads_nt(self, rows, T)
    v1 = valid.reshape(1, T, 1)
    beta, g = torch.sigmoid(b) * v1, Bld.softplus(a + self.dt) * self.neg_a * v1
    s = self.commit_pending(rec, pend, commit, commit_last)
    NB, C = T // Bld.C_SUB, Bld.C_SUB
    l_inc, l_str = Bld.tri(C, False), Bld.tri(C, True)
    q4, k4, v4 = qh.reshape(nv, NB, C, dk), kh.reshape(nv, NB, C, dk), vh.reshape(nv, NB, C, dv)
    b4, g4 = beta.reshape(nv, NB, C, 1), g.reshape(nv, NB, C, 1)
    cum = (g4.reshape(nv, NB, 1, C) @ l_inc.T).reshape(nv, NB, C, 1)
    pair = torch.exp(torch.clamp(cum - cum.reshape(nv, NB, 1, C), max=0)) * l_inc
    kb, vb = k4 * b4, v4 * b4
    n = (kb @ k4.transpose(-1, -2)) * (pair * l_str)
    x = Bld.tri_solve(n, torch.cat([vb, kb * torch.exp(cum)], -1), C)
    u, wk = x[..., :dv], x[..., dv:]
    total = cum[:, :, C - 1:C, :]
    qd, kdc = q4 * torch.exp(cum), k4 * torch.exp(total - cum)
    intra = (q4 @ k4.transpose(-1, -2)) * pair
    outs = []
    for bi in range(NB):
        vn = u[:, bi] - wk[:, bi] @ s
        outs.append(qd[:, bi] @ s + intra[:, bi] @ vn)
        s = s * torch.exp(total[:, bi]) + kdc[:, bi].transpose(1, 2) @ vn
    conv_out = torch.cat([conv_sel_out @ rows, torch.zeros(P, cdim, dtype=rows.dtype)], 0)
    return _finish_nt(self, torch.cat(outs, 1), z, T), conv_out, s, pend * 0


VARIANTS = {
    "ref": {},
    "neumann": {"globals": {"fwd_sub": fwd_sub_neumann}},
    "inv": {"globals": {"fwd_sub": fwd_sub_inverse}},
    "inv_ew": {"globals": {"fwd_sub": fwd_sub_inverse_ew}},
    "ew16": {"globals": {"fwd_sub": fwd_sub_inverse_ew, "C_SUB": 16}},
    "ew32": {"globals": {"fwd_sub": fwd_sub_inverse_ew, "C_SUB": 32}},
    "ew64": {"globals": {"fwd_sub": fwd_sub_inverse_ew, "C_SUB": 64}},
    "fast_b": {"globals": {"GDN_FAST": True}},     # the builder's GDN_FAST graph
    "nt": {"verify": verify_nt, "prefill": prefill_nt, "globals": {"fwd_sub": fwd_sub_inverse_ew}, "timing_only": True},
    "fast": {"verify": verify_fast},
    "c16": {"globals": {"C_SUB": 16}},
    "c32": {"globals": {"C_SUB": 32}},
    "noloop": {"prefill": prefill_noloop, "timing_only": True},
    "nofwd": {"globals": {"fwd_sub": lambda n, rhs, rows: rhs}, "timing_only": True},
}


def _patch_methods(**methods):
    """Variant hook: replace GDNW methods on the instance (timing-only ablations)."""
    def apply(mix):
        for name, fn in methods.items():
            setattr(mix, name, types.MethodType(fn, mix))
    return apply


def _qkv_heads_noconv(self, rows, T):
    conv = rows[3:T + 3].transpose(0, 1)
    qq, kk, vv = conv[:kd], conv[kd:2 * kd], conv[2 * kd:]

    def heads(t):
        t = t.reshape(nk, dk, T).permute(0, 2, 1)
        return t.reshape(nk, 1, T, dk).repeat(1, nv // nk, 1, 1).reshape(nv, T, dk)

    def l2n(t, s_):
        return t * torch.rsqrt((t * t).sum(-1, keepdim=True) + 1e-6) * s_
    vh = vv.reshape(nv, dv, T).permute(0, 2, 1) * Bld.GDN_SV
    return l2n(heads(qq), dk ** -0.5 * Bld.GDN_SQ), l2n(heads(kk), 1.0), vh


def _qkv_heads_dw(self, rows, T):
    """GDNW.qkv_heads with the causal conv1d as one native depthwise conv over a channel-major (1, C, 1, T + 3) view."""
    w = self.cw.transpose(0, 1).reshape(cdim, 1, 1, 4)
    conv = torch.nn.functional.conv2d(rows.transpose(0, 1).reshape(1, cdim, 1, T + 3), w, groups=cdim)
    conv = Bld.silu(conv).reshape(cdim, T)
    qq, kk, vv = conv[:kd], conv[kd:2 * kd], conv[2 * kd:]

    def heads(t):
        t = t.reshape(nk, dk, T).permute(0, 2, 1)
        return t.reshape(nk, 1, T, dk).repeat(1, nv // nk, 1, 1).reshape(nv, T, dk)

    def l2n(t, s_):
        return t * torch.rsqrt((t * t).sum(-1, keepdim=True) + 1e-6) * s_
    vh = vv.reshape(nv, dv, T).permute(0, 2, 1)
    vh = vh * Bld.GDN_SV if Bld.GDN_SV != 1 else vh
    return l2n(heads(qq), dk ** -0.5 * Bld.GDN_SQ), l2n(heads(kk), 1.0), vh


def prefill_merged(self, h, conv_rows, conv_sel, conv_sel_out, rec, pend, commit, commit_last, valid, T):
    """GDNW.prefill with wk @ S and (q * exp(cum)) @ S as one matmul per sub-chunk (same math)."""
    qkv, z, b, a = self.proj(h, T)
    rows = torch.cat([conv_sel @ conv_rows, qkv.transpose(0, 1)], 0)
    qh, kh, vh = self.qkv_heads(rows, T)
    v1 = valid.reshape(1, T, 1)
    beta, g = torch.sigmoid(b) * v1, Bld.softplus(a + self.dt) * self.neg_a * v1
    s = self.commit_pending(rec, pend, commit, commit_last)
    NB, C = T // Bld.C_SUB, Bld.C_SUB
    l_inc, l_str = Bld.tri(C, False), Bld.tri(C, True)
    q4, k4, v4 = qh.reshape(nv, NB, C, dk), kh.reshape(nv, NB, C, dk), vh.reshape(nv, NB, C, dv)
    b4, g4 = beta.reshape(nv, NB, C, 1), g.reshape(nv, NB, C, 1)
    cum = (g4.reshape(nv, NB, 1, C) @ l_inc.T).reshape(nv, NB, C, 1)
    pair = torch.exp(torch.clamp(cum - cum.reshape(nv, NB, 1, C), max=0)) * l_inc
    kb, vb = k4 * b4, v4 * b4
    n = (kb @ k4.transpose(-1, -2)) * (pair * l_str)
    x = Bld.tri_solve(n, torch.cat([vb, kb * torch.exp(cum)], -1), C)
    u, wk = x[..., :dv], x[..., dv:]
    total = cum[:, :, C - 1:C, :]
    qd, kdc = q4 * torch.exp(cum), k4 * torch.exp(total - cum)
    intra = (q4 @ k4.transpose(-1, -2)) * pair
    wq = torch.cat([wk, qd], 2)                                                         # (nv, NB, 2C, dk)
    outs = []
    for bi in range(NB):
        ws = wq[:, bi] @ s
        vn = u[:, bi] - ws[:, :C]
        outs.append(ws[:, C:] + intra[:, bi] @ vn)
        s = s * torch.exp(total[:, bi]) + kdc[:, bi].transpose(1, 2) @ vn
    conv_out = torch.cat([conv_sel_out @ rows, torch.zeros(P, cdim, dtype=rows.dtype)], 0)
    return self.finish(torch.cat(outs, 1), z, T), conv_out, s, pend * 0


VARIANTS.update({
    "inv_dw_m": {"globals": {"fwd_sub": fwd_sub_inverse_ew}, "prefill": prefill_merged,
                 "hook": _patch_methods(qkv_heads=_qkv_heads_dw)},
    "inv_dw": {"globals": {"fwd_sub": fwd_sub_inverse_ew}, "hook": _patch_methods(qkv_heads=_qkv_heads_dw)},
    "inv_noloop": {"prefill": prefill_noloop, "globals": {"fwd_sub": fwd_sub_inverse_ew}, "timing_only": True},
    "inv_nocommit": {"globals": {"fwd_sub": fwd_sub_inverse_ew}, "timing_only": True,
                     "hook": _patch_methods(commit_pending=lambda self, rec, pend, commit, commit_last: rec * 0.5)},
    "inv_noconv": {"globals": {"fwd_sub": fwd_sub_inverse_ew}, "timing_only": True,
                   "hook": _patch_methods(qkv_heads=_qkv_heads_noconv)},
    "inv_nofinish": {"globals": {"fwd_sub": fwd_sub_inverse_ew}, "timing_only": True,
                     "hook": _patch_methods(finish=lambda self, o, z, T: self.out(o.permute(0, 2, 1).reshape(1, vd, 1, T)))},
})


@contextlib.contextmanager
def patched(variant: str):
    """Builder globals of a variant, active while its program is traced or run on the host."""
    over = VARIANTS[variant].get("globals", {})
    old = {k: getattr(Bld, k) for k in over}
    for k, v in over.items():
        setattr(Bld, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(Bld, k, v)


# ---- program -------------------------------------------------------------------------------------------------------
def small_tensors(layer: int) -> dict:
    wmap = json.loads((Bld.M.MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    pre = f"model.language_model.layers.{layer}.linear_attn."
    out = {}
    for k in ("A_log", "conv1d.weight", "dt_bias", "norm.weight"):
        with safe_open(Bld.M.MODEL / wmap[pre + k], framework="pt") as f:
            out[k] = f.get_tensor(pre + k).float().numpy()
    return out


def make_mix(layer: int, variant: str) -> nn.Module:
    sm = small_tensors(layer)
    p = f"{layer}/linear_attn."
    z = np.zeros
    W = {p + "in_proj_qkv.weight/dense": z((cdim, 8)), p + "in_proj_z.weight/dense": z((vd, 8)),
         p + "out_proj.weight/dense": z((8, vd)), p + "in_proj_a.weight": z((nv, 8)), p + "in_proj_b.weight": z((nv, 8)),
         p + "conv1d.weight": sm["conv1d.weight"], p + "A_log": sm["A_log"], p + "dt_bias": sm["dt_bias"],
         p + "norm.weight": sm["norm.weight"]}
    mix = Bld.GDNW(W, layer)

    def proj(self, h, T):  # the GDNW.proj reshapes, from projection outputs given as inputs
        qkv, zz, bb, aa = h
        return qkv.reshape(cdim, T), zz.reshape(nv, dv, T).permute(0, 2, 1), bb.reshape(nv, T, 1), aa.reshape(nv, T, 1)
    mix.proj = types.MethodType(proj, mix)
    mix.out = nn.Identity()
    spec = VARIANTS[variant]
    if "verify" in spec:
        mix.verify = types.MethodType(spec["verify"], mix)
    if "prefill" in spec:
        mix.prefill = types.MethodType(spec["prefill"], mix)
    if "hook" in spec:
        spec["hook"](mix)
    return mix


class Cores(nn.Module):
    def __init__(self, mixes, T: int):
        super().__init__()
        self.mixes, self.T, self.prefill = nn.ModuleList(mixes), T, T > P

    def input_names(self):
        return (["qkv", "z", "b", "a", "conv_rows", "conv_sel", "commit", "commit_last", "rec", "pend"]
                + (["conv_sel_out", "valid"] if self.prefill else []))

    def output_names(self):
        return [f"{s}{j}" for j in range(len(self.mixes)) for s in ("o", "rows", "s", "pend")]

    def forward(self, qkv, z, b, a, conv_rows, conv_sel, commit, commit_last, rec, pend, *extra):
        out = []
        for mix in self.mixes:
            h = (qkv, z, b, a)
            if self.prefill:
                r = mix.prefill(h, conv_rows, conv_sel, extra[0], rec, pend, commit, commit_last, extra[1], self.T)
            else:
                r = mix.verify(h, conv_rows, conv_sel, rec, pend, commit, commit_last, self.T)
            out += list(r)
        return tuple(out)

    def example(self):
        return tuple(torch.from_numpy(x) for x in example_inputs(self.T, np.random.default_rng(0)))


def example_inputs(T: int, rng, accepted: int = 5) -> list[np.ndarray]:
    """Live-like inputs: projection outputs ~N(0, 0.5), a previous call's pending rows (`accepted` committed)."""
    def n(*shape, s=0.5):
        return (rng.standard_normal(shape) * s).astype(np.float16)
    qkv, z, b, a = n(1, cdim, 1, T), n(1, vd, 1, T), n(1, nv, 1, T, s=1.0), n(1, nv, 1, T, s=1.0)
    conv_rows = n(P + 3, cdim)
    k = accepted
    conv_sel = np.zeros((3, P + 3), np.float16)
    conv_sel[np.arange(3), k + np.arange(3)] = 1
    commit = np.zeros((1, P, 1), np.float16)
    commit[0, :k] = 1
    commit_last = np.zeros((1, P, 1), np.float16)
    commit_last[0, k - 1] = 1
    rec = n(nv, dk, dv, s=2.0)
    kp = rng.standard_normal((nv, P, dk))
    kp /= np.linalg.norm(kp, axis=-1, keepdims=True)
    pend = np.zeros((nv, 3 * P + 1, dv), np.float16)
    pend[:, 0:P, :dk] = kp
    pend[:, P:2 * P] = n(nv, P, dv, s=8.0)
    pend[:, 2 * P:3 * P, :dk] = n(nv, P, dk, s=0.3)
    pend[:, 3 * P, :P] = -np.cumsum(rng.uniform(0.0, 0.3, (nv, P)), axis=-1)
    ins = [qkv, z, b, a, conv_rows, conv_sel, commit, commit_last, rec, pend]
    if T > P:
        so = np.zeros((3, T + 3), np.float16)
        so[np.arange(3), T - 3 + np.arange(3)] = 1
        ins += [so, np.ones((1, T, 1), np.float16)]
    return ins


def build_cores(variant: str, layers: int, T: int) -> Cores:
    return Cores([make_mix(i, variant) for i in range(layers)], T).eval().to(f16)


_TRI = Bld.tri


def host_outputs(variant: str, layers: int, T: int, dtype=torch.float32) -> list[np.ndarray]:
    mod = build_cores(variant, layers, T).to(dtype)
    ins = [torch.from_numpy(x).to(dtype) for x in example_inputs(T, np.random.default_rng(0))]
    Bld.tri = lambda n, strict: _TRI(n, strict).to(dtype)  # the builder's masks are fp16 constants
    try:
        with torch.no_grad(), patched(variant):
            return [o.float().numpy() for o in mod(*ins)]
    finally:
        Bld.tri = _TRI


def rel(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-30))


# ---- commands ------------------------------------------------------------------------------------------------------
def cmd_check(a):
    for T in a.rows:
        ref = host_outputs("ref", a.layers, T)
        for v in a.variants:
            if v == "ref":
                continue
            out = host_outputs(v, a.layers, T)
            errs = [rel(x, y) for x, y in zip(out, ref)]
            tag = " (timing only)" if VARIANTS[v].get("timing_only") else ""
            print(f"T={T:3d} {v:10s} max rel diff vs ref (fp32 host): {max(errs):.2e}{tag}", flush=True)


def cmd_build(a):
    a.out.mkdir(parents=True, exist_ok=True)
    for v in a.variants:
        dst = a.out / f"{v}.aimodel"
        if dst.exists() and not a.force:
            print(f"{v}: exists")
            continue
        entries = []
        for T in a.rows:
            mod = build_cores(v, a.layers, T)
            entries.append((f"t{T}", mod, mod.input_names(), mod.output_names()))
        t = time.time()
        with patched(v):
            Bld.save_program(entries, dst)
        print(f"{v}: built in {time.time() - t:.0f}s", flush=True)


def placement(pkg: Path, entries: list[str]) -> str:
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    r = IC.inspect_package(pkg, entries, Path.home() / "Library/Caches/coreai-cache", build, sys.executable)
    return r["status"]


def cmd_time(a):
    pkgs = sorted(a.out.glob("*.aimodel"))
    if a.variants:
        pkgs = [p for p in pkgs if p.stem in a.variants]
    refs, res, runs = {}, {}, {}
    for p in pkgs:
        m = B.Model(p, compute="ane")
        for name in m.function_names:
            T = int(name[1:])
            fn = m.function(name)
            data = example_inputs(T, np.random.default_rng(0))
            ins = {}
            for n, x in zip(fn.input_names, data):
                b = fn.buffer("input", n)
                b.np[...] = x
                ins[n] = b
            outs = {n: fn.buffer("output", n) for n in fn.output_names}
            plan = B.Plan([fn.bind(ins, outs)])
            plan.run()
            if T not in refs:
                refs[T] = host_outputs("ref", a.layers, T)
            errs = [rel(outs[n].np.astype(np.float32), r) for n, r in zip(fn.output_names, refs[T])]
            runs[(p.stem, name)] = plan
            res.setdefault(p.stem, {})[name] = {"err_max": max(errs), "err_o": max(errs[0::4]),
                                                "err_s": max(errs[2::4]), "keep": (m, fn, ins, outs)}
        res[p.stem]["placement"] = placement(p, m.function_names)
    acc = {k: [] for k in runs}
    for _ in range(a.rounds):
        for k, plan in runs.items():
            acc[k].append(S.timed(plan, a.n, 2)["median_ms"])
    report = {}
    for (v, name), xs in acc.items():
        r = res[v][name]
        report.setdefault(v, {"placement": res[v]["placement"]})[name] = {
            "median_ms": float(np.median(xs)), "min_ms": float(np.min(xs)), "err_max": r["err_max"],
            "err_o": r["err_o"], "err_s": r["err_s"]}
    for v, r in report.items():
        cells = "  ".join(f"{n}: {x['median_ms']:6.3f} ms (o err {x['err_o']:.1e}, S err {x['err_s']:.1e})"
                          for n, x in r.items() if n != "placement")
        print(f"{v:10s} [{r['placement']}] {cells}", flush=True)
    (a.out / "timing.json").write_text(json.dumps(report, indent=1))


def cmd_coreml(a):
    """Same cores as a Core ML mlprogram (FP16, CPU+ANE) for per-op cost estimates, e.g. `anemll-profile <pkg>`.
    Core ML and Core AI lower to the same ANE compiler; the op mix, not the runtime, is what this inspects."""
    import coremltools as ct
    a.out.mkdir(parents=True, exist_ok=True)
    for v in a.variants:
        for T in a.rows:
            mod = build_cores(v, a.layers, T).float()
            ins = [torch.from_numpy(x).float() for x in example_inputs(T, np.random.default_rng(0))]
            Bld.tri = lambda n, strict: _TRI(n, strict).float()
            try:
                with torch.no_grad(), patched(v):
                    traced = torch.jit.trace(mod, tuple(ins), check_trace=False)
            finally:
                Bld.tri = _TRI
            ml = ct.convert(traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                            compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE,
                            inputs=[ct.TensorType(name=n, shape=x.shape, dtype=np.float16)
                                    for n, x in zip(mod.input_names(), ins)],
                            outputs=[ct.TensorType(name=n, dtype=np.float16) for n in mod.output_names()])
            dst = a.out / f"{v}_t{T}.mlpackage"
            ml.save(str(dst))
            print(f"{v} t{T}: {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("check", "build", "time", "coreml"))
    ap.add_argument("--variants", default="ref,fast")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--rows", default="8,64")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=5)
    a = ap.parse_args()
    a.variants = [v for v in a.variants.split(",") if v]
    a.rows = [int(x) for x in a.rows.split(",")]
    {"check": cmd_check, "build": cmd_build, "time": cmd_time, "coreml": cmd_coreml}[a.cmd](a)


if __name__ == "__main__":
    main()
