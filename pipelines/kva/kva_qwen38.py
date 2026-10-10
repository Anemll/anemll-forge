#!/usr/bin/env python3
"""Late-Layer KV Approximation (LLKVApprox / CED-style) prototype for Qwen3.8-27B.

Background and attribution
--------------------------
This is a spinoff research prototype of the "project the late-half KV/mixer state from a
mid-stack residual" idea. It adapts, rather than reinvents, published work:

  * DeepSeek-V4.1-Flash CED: decoder global KV projected from an encoder mid-state H_{L/2}
    (https://arxiv.org/abs/2609.19969).
  * @kis LLKVApprox on Qwen3-8B (~1/2 prefill, frozen base + a trained projector):
    https://x.com/kis/status/2098185646749909306 and the Q3-8B-KVA-Projector artifact.
  * alimpfard's hybrid Qwen3.8-27B KV-approximation, which is the recipe this file follows:
    https://huggingface.co/alimpfard/qwen3.8-27b-kv-approximation
      - prefill layers 0..31 exact; a projector maps the layer-31 residual to the late
        mixer inputs for layers 32..63,
      - late Gated-DeltaNet (GDN): predict in_proj_qkv / in_proj_a / in_proj_b, then run the
        exact cheap conv + delta scan,
      - late GQA: predict k_proj / v_proj (pre-norm, pre-RoPE), then the exact K/V write,
      - recommend an exact tail of the last 1-4k tokens for agentic fidelity.

Why NumPy and why a prototype
-----------------------------
The anemll-forge release path is Core AI / Core ML on Apple Silicon (M3U -> M6 ANE). Those
toolchains, the Qwen3.8-27B checkpoint and Apple GPU/ANE are not present in every environment
(e.g. CI / Linux). So this file is a dependency-light, hardware-independent reference that:

  * mirrors the exact Qwen3.8 decoder math from scripts/qwen38_decode_ref.py (RMSNorm,
    GDN conv + delta recurrence, GQA with partial RoPE and the per-head output gate),
  * implements the KVA on/off prefill paths and the exact-tail flag so the plumbing and the
    projector I/O contract can be validated and timed on any machine,
  * reads the real text_config from a checkpoint when --model is given, and otherwise uses the
    repo-derived reference config (qwen3_5_27b_config.reference.json).

It is NOT a trained model: by default the base weights and the projector are random, so KVA-on
output does not match exact. The structural prefill speedup (skipping the late-half MLP + mixer
residual for the approximated region) and the I/O shapes are real and measurable; the *quality*
of the approximation depends on a trained projector (see --fit-projector for a synthetic
least-squares demonstration, and the README for using a real published projector).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE_CONFIG = HERE / "qwen3_5_27b_config.reference.json"


# ----------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------
@dataclass
class Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    full_attention_interval: int
    rope_theta: float
    partial_rotary_factor: float
    layer_types: list[str] = field(default_factory=list)

    @property
    def conv_dim(self) -> int:
        return 2 * self.linear_num_key_heads * self.linear_key_head_dim \
            + self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def rot_dim(self) -> int:
        r = int(self.head_dim * self.partial_rotary_factor)
        return r - (r % 2)                 # rotary half-split requires an even dim

    def finalize(self) -> "Config":
        if not self.layer_types:
            n, k = self.num_hidden_layers, self.full_attention_interval
            self.layer_types = [
                "full_attention" if (i % k) == (k - 1) else "linear_attention"
                for i in range(n)
            ]
        return self

    @property
    def gdn_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]

    @property
    def attn_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]


def _config_from_text(tc: dict) -> Config:
    rope = tc.get("rope_parameters", tc.get("rope_scaling", {})) or {}
    cfg = Config(
        hidden_size=tc["hidden_size"],
        intermediate_size=tc.get("intermediate_size", 4 * tc["hidden_size"]),
        num_hidden_layers=tc["num_hidden_layers"],
        vocab_size=tc.get("vocab_size", 0),
        rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
        num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"],
        linear_num_key_heads=tc["linear_num_key_heads"],
        linear_num_value_heads=tc["linear_num_value_heads"],
        linear_key_head_dim=tc["linear_key_head_dim"],
        linear_value_head_dim=tc["linear_value_head_dim"],
        linear_conv_kernel_dim=tc["linear_conv_kernel_dim"],
        full_attention_interval=tc.get("full_attention_interval", 4),
        rope_theta=rope.get("rope_theta", 1e7),
        partial_rotary_factor=rope.get("partial_rotary_factor", 0.25),
        layer_types=list(tc.get("layer_types", [])),
    )
    return cfg.finalize()


def load_reference_config() -> Config:
    tc = json.loads(REFERENCE_CONFIG.read_text())["text_config"]
    return _config_from_text(tc)


def load_checkpoint_config(model_dir: Path) -> Config:
    cfg_path = Path(model_dir) / "config.json"
    tc = json.loads(cfg_path.read_text()).get("text_config")
    if tc is None:
        raise ValueError(f"No text_config in {cfg_path}")
    return _config_from_text(tc)


def scaled_config(base: Config, hidden: int, layers: int) -> Config:
    """A small config that preserves the hybrid structure (interval-4 attention, head ratios),
    for fast on/off timing and end-to-end checks on CPU."""
    assert hidden % 64 == 0 and layers % base.full_attention_interval == 0
    # keep head_dim-ish ratios but shrink counts to keep matrices small
    nkv = max(1, base.num_key_value_heads // 2)
    nq = nkv * (base.num_attention_heads // base.num_key_value_heads)
    hd = hidden // nq
    lk_heads = max(2, base.linear_num_key_heads // 4)
    lv_heads = max(2, base.linear_num_value_heads // 4)
    lhd = max(16, base.linear_key_head_dim // 4)
    cfg = Config(
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        vocab_size=512,
        rms_norm_eps=base.rms_norm_eps,
        num_attention_heads=nq,
        num_key_value_heads=nkv,
        head_dim=hd,
        linear_num_key_heads=lk_heads,
        linear_num_value_heads=lv_heads,
        linear_key_head_dim=lhd,
        linear_value_head_dim=lhd,
        linear_conv_kernel_dim=base.linear_conv_kernel_dim,
        full_attention_interval=base.full_attention_interval,
        rope_theta=base.rope_theta,
        partial_rotary_factor=base.partial_rotary_factor,
    )
    return cfg.finalize()


# ----------------------------------------------------------------------------------------------
# Elementwise helpers (match scripts/qwen38_decode_ref.py)
# ----------------------------------------------------------------------------------------------
def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def rms_zc(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """Qwen3_5RMSNorm: zero-centered weight (x normalized, scaled by 1 + w). Works on (..., d)."""
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * (1.0 + w)


# ----------------------------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------------------------
def init_weights(cfg: Config, rng: np.random.Generator, dtype=np.float32) -> dict:
    """Random frozen 'base' weights with checkpoint-relative names (one dict per layer index)."""
    h = cfg.hidden_size
    w: dict[int, dict] = {}

    def small(*shape, s=0.02):
        return (rng.standard_normal(shape) * s).astype(dtype)

    for i, kind in enumerate(cfg.layer_types):
        lw: dict[str, np.ndarray] = {
            "input_layernorm.weight": small(h, s=0.01),
            "post_attention_layernorm.weight": small(h, s=0.01),
            "mlp.gate_proj.weight": small(cfg.intermediate_size, h),
            "mlp.up_proj.weight": small(cfg.intermediate_size, h),
            "mlp.down_proj.weight": small(h, cfg.intermediate_size),
        }
        if kind == "linear_attention":
            nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
            dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
            lw.update({
                "linear_attn.in_proj_qkv.weight": small(cfg.conv_dim, h),
                "linear_attn.in_proj_z.weight": small(nv * dv, h),
                "linear_attn.in_proj_a.weight": small(nv, h),
                "linear_attn.in_proj_b.weight": small(nv, h),
                "linear_attn.conv1d.weight": small(cfg.conv_dim, cfg.linear_conv_kernel_dim, s=0.3),
                "linear_attn.A_log": small(nv, s=0.1),
                "linear_attn.dt_bias": small(nv, s=0.1),
                "linear_attn.norm.weight": small(dv, s=0.01),
                "linear_attn.out_proj.weight": small(h, nv * dv),
            })
        else:
            nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
            lw.update({
                "self_attn.q_proj.weight": small(nh * 2 * hd, h),
                "self_attn.k_proj.weight": small(nkv * hd, h),
                "self_attn.v_proj.weight": small(nkv * hd, h),
                "self_attn.q_norm.weight": small(hd, s=0.01),
                "self_attn.k_norm.weight": small(hd, s=0.01),
                "self_attn.o_proj.weight": small(h, nh * hd),
            })
        w[i] = lw
    w["embed_tokens"] = small(cfg.vocab_size, h, s=0.02) if cfg.vocab_size else None
    w["final_norm.weight"] = small(h, s=0.01)
    w["lm_head.weight"] = small(cfg.vocab_size, h) if cfg.vocab_size else None
    return w


# ----------------------------------------------------------------------------------------------
# Projector: H_{split} residual -> late mixer inputs (per late layer)
# ----------------------------------------------------------------------------------------------
def projector_output_spec(cfg: Config, late_layers: list[int]) -> dict[int, dict[str, int]]:
    """Per-late-layer map of {target name -> output dim} predicted from the split residual."""
    nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    nkv, hd = cfg.num_key_value_heads, cfg.head_dim
    spec: dict[int, dict[str, int]] = {}
    for i in late_layers:
        if cfg.layer_types[i] == "linear_attention":
            # recipe: predict in_proj_qkv, in_proj_a, in_proj_b (scan-critical). z is the output
            # gate; we predict it too so the late mixer output can be formed in the exact tail
            # without the per-layer hidden state.
            spec[i] = {"qkv": cfg.conv_dim, "z": nv * dv, "a": nv, "b": nv}
        else:
            spec[i] = {"k": nkv * hd, "v": nkv * hd}
    return spec


def init_projector(cfg: Config, late_layers: list[int], rng: np.random.Generator,
                   dtype=np.float32) -> dict:
    """A thin per-layer linear projector: out = W_head @ h_split + bias, one head per target.
    This mirrors a frozen-base + trained-projector setup; here the heads are random unless fit."""
    h = cfg.hidden_size
    spec = projector_output_spec(cfg, late_layers)
    proj: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for i, heads in spec.items():
        proj[i] = {name: ((rng.standard_normal((dim, h)) * (1.0 / np.sqrt(h))).astype(dtype),
                          np.zeros(dim, dtype))
                   for name, dim in heads.items()}
    return proj


def project(proj_layer: dict[str, tuple[np.ndarray, np.ndarray]], name: str,
            h_split: np.ndarray) -> np.ndarray:
    """h_split: (T, hidden) -> (T, dim). Linear head W h + b."""
    W, b = proj_layer[name]
    return h_split @ W.T + b


# ----------------------------------------------------------------------------------------------
# RoPE (partial rotary, matches decode_ref)
# ----------------------------------------------------------------------------------------------
def rope_tables(cfg: Config, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot = cfg.rot_dim
    inv_freq = 1.0 / cfg.rope_theta ** (np.arange(0, rot, 2) / rot)
    f = positions[:, None] * inv_freq[None, :]            # (T, rot/2)
    emb = np.concatenate([f, f], axis=-1)                 # (T, rot)
    return np.cos(emb), np.sin(emb)


def apply_rope(t: np.ndarray, cos: np.ndarray, sin: np.ndarray, rot: int) -> np.ndarray:
    """t: (..., head_dim); cos/sin: (..., rot). Rotate the first `rot` dims, pass the rest."""
    r, rest = t[..., :rot], t[..., rot:]
    half = rot // 2
    rot_half = np.concatenate([-r[..., half:], r[..., :half]], axis=-1)
    return np.concatenate([r * cos + rot_half * sin, rest], axis=-1)


# ----------------------------------------------------------------------------------------------
# Per-layer mixer state
# ----------------------------------------------------------------------------------------------
@dataclass
class GDNState:
    conv_state: np.ndarray      # (conv_dim, kernel-1)
    rec_state: np.ndarray       # (nv, dk, dv)


@dataclass
class AttnState:
    k_cache: np.ndarray         # (nkv, ctx, head_dim)
    v_cache: np.ndarray         # (nkv, ctx, head_dim)
    pos: int = 0


def new_gdn_state(cfg: Config, dtype=np.float32) -> GDNState:
    nv, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    return GDNState(np.zeros((cfg.conv_dim, cfg.linear_conv_kernel_dim - 1), dtype),
                    np.zeros((nv, dk, dv), dtype))


def new_attn_state(cfg: Config, ctx: int, dtype=np.float32) -> AttnState:
    nkv, hd = cfg.num_key_value_heads, cfg.head_dim
    return AttnState(np.zeros((nkv, ctx, hd), dtype), np.zeros((nkv, ctx, hd), dtype))


# ----------------------------------------------------------------------------------------------
# GDN scan given already-projected mixer inputs (the "exact cheap scan")
# ----------------------------------------------------------------------------------------------
def gdn_scan(cfg: Config, lw: dict, st: GDNState,
             qkv: np.ndarray, a: np.ndarray, b: np.ndarray,
             z: np.ndarray | None, want_output: bool) -> np.ndarray | None:
    """Advance conv + delta recurrence over T tokens from the given per-token mixer inputs.

    qkv: (T, conv_dim), a/b: (T, nv), z: (T, nv*dv) or None. Updates st in place.
    Returns the mixer output (T, hidden) if want_output else None.
    This is exactly the GDN math in scripts/qwen38_decode_ref.py, vectorized over heads."""
    nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    kd = nk * dk
    kernel = cfg.linear_conv_kernel_dim
    eps = cfg.rms_norm_eps
    T = qkv.shape[0]
    cw = lw["linear_attn.conv1d.weight"]           # (conv_dim, kernel)

    # causal depthwise conv over time, newest token at kernel-1 (matches the shift register)
    pad = np.concatenate([st.conv_state.T, qkv], axis=0)   # (kernel-1 + T, conv_dim)
    pre = np.zeros((T, cfg.conv_dim), qkv.dtype)
    for j in range(kernel):
        pre += cw[:, j][None, :] * pad[j:j + T]
    conv = silu(pre)                                       # (T, conv_dim)
    st.conv_state = pad[-(kernel - 1):].T                  # newest kernel-1 qkv rows

    beta = sigmoid(b)                                      # (T, nv)
    g = -np.exp(lw["linear_attn.A_log"]) * softplus(a + lw["linear_attn.dt_bias"])  # (T, nv)
    rep = nv // nk

    out = np.empty((T, nv * dv), qkv.dtype) if want_output else None
    s = st.rec_state                                       # (nv, dk, dv)
    for t in range(T):
        c = conv[t]
        q = c[:kd].reshape(nk, dk)
        k = c[kd:2 * kd].reshape(nk, dk)
        v = c[2 * kd:].reshape(nv, dv)
        q = np.repeat(q, rep, axis=0)
        k = np.repeat(k, rep, axis=0)
        q = q * (1.0 / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)) / dk ** 0.5
        k = k * (1.0 / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6))
        s = s * np.exp(g[t])[:, None, None]
        kv_mem = (s * k[:, :, None]).sum(1)               # (nv, dv)
        delta = (v - kv_mem) * beta[t][:, None]           # (nv, dv)
        s = s + k[:, :, None] * delta[:, None, :]
        if want_output:
            o = (s * q[:, :, None]).sum(1)                # (nv, dv)
            o = rms_zc(o, lw["linear_attn.norm.weight"], eps) * silu(z[t].reshape(nv, dv))
            out[t] = o.reshape(-1)
    st.rec_state = s
    if want_output:
        return out @ lw["linear_attn.out_proj.weight"].T
    return None


# ----------------------------------------------------------------------------------------------
# GQA given already-projected K/V (pre-norm, pre-RoPE); optional full attention output
# ----------------------------------------------------------------------------------------------
def gqa_write(cfg: Config, lw: dict, st: AttnState,
              k_lin: np.ndarray, v_lin: np.ndarray, positions: np.ndarray) -> None:
    """Exact K/V write: k_norm + partial RoPE on K, then store K/V at the given positions.
    k_lin/v_lin: (T, nkv*hd)."""
    nkv, hd = cfg.num_key_value_heads, cfg.head_dim
    T = k_lin.shape[0]
    k = rms_zc(k_lin.reshape(T, nkv, hd), lw["self_attn.k_norm.weight"], cfg.rms_norm_eps)
    v = v_lin.reshape(T, nkv, hd)
    cos, sin = rope_tables(cfg, positions)
    k = apply_rope(k, cos[:, None, :], sin[:, None, :], cfg.rot_dim)
    for t in range(T):
        p = positions[t]
        st.k_cache[:, p] = k[t]
        st.v_cache[:, p] = v[t]
    st.pos = int(positions[-1]) + 1


def gqa_full(cfg: Config, lw: dict, st: AttnState, h: np.ndarray,
             positions: np.ndarray) -> np.ndarray:
    """Exact full GQA for T tokens from the normed layer input h (T, hidden): writes K/V and
    returns the attention output (T, hidden). Causal over the cache up to each position."""
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    T = h.shape[0]
    qg = (h @ lw["self_attn.q_proj.weight"].T).reshape(T, nh, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:]
    q = rms_zc(q, lw["self_attn.q_norm.weight"], cfg.rms_norm_eps)
    k_lin = h @ lw["self_attn.k_proj.weight"].T
    v_lin = h @ lw["self_attn.v_proj.weight"].T
    gqa_write(cfg, lw, st, k_lin, v_lin, positions)
    cos, sin = rope_tables(cfg, positions)
    q = apply_rope(q, cos[:, None, :], sin[:, None, :], cfg.rot_dim)
    rep = nh // nkv
    out = np.empty((T, nh * hd), h.dtype)
    for t in range(T):
        p = int(positions[t])
        kk = np.repeat(st.k_cache[:, :p + 1], rep, axis=0)    # (nh, p+1, hd)
        vv = np.repeat(st.v_cache[:, :p + 1], rep, axis=0)
        att = (kk @ q[t][:, :, None])[..., 0] / hd ** 0.5     # (nh, p+1)
        att = att - att.max(-1, keepdims=True)
        att = np.exp(att)
        att = att / att.sum(-1, keepdims=True)
        o = (att[:, :, None] * vv).sum(1).reshape(-1) * sigmoid(gate[t].reshape(-1))
        out[t] = o
    return out @ lw["self_attn.o_proj.weight"].T


# ----------------------------------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------------------------------
def mlp(lw: dict, h: np.ndarray) -> np.ndarray:
    g = silu(h @ lw["mlp.gate_proj.weight"].T)
    u = h @ lw["mlp.up_proj.weight"].T
    return (g * u) @ lw["mlp.down_proj.weight"].T


# ----------------------------------------------------------------------------------------------
# Exact prefill of a contiguous layer range over T tokens
# ----------------------------------------------------------------------------------------------
def exact_layer_prefill(cfg: Config, lw: dict, kind: str, x: np.ndarray,
                        gdn_st: GDNState | None, attn_st: AttnState | None,
                        positions: np.ndarray) -> np.ndarray:
    """One decoder layer, exact, over T tokens. x: (T, hidden) residual in; returns residual out."""
    eps = cfg.rms_norm_eps
    h = rms_zc(x, lw["input_layernorm.weight"], eps)
    if kind == "linear_attention":
        qkv = h @ lw["linear_attn.in_proj_qkv.weight"].T
        z = h @ lw["linear_attn.in_proj_z.weight"].T
        a = h @ lw["linear_attn.in_proj_a.weight"].T
        b = h @ lw["linear_attn.in_proj_b.weight"].T
        mix = gdn_scan(cfg, lw, gdn_st, qkv, a, b, z, want_output=True)
    else:
        mix = gqa_full(cfg, lw, attn_st, h, positions)
    x = x + mix
    h2 = rms_zc(x, lw["post_attention_layernorm.weight"], eps)
    return x + mlp(lw, h2)


# ----------------------------------------------------------------------------------------------
# Full-model prefill (KVA off) and KVA prefill (on)
# ----------------------------------------------------------------------------------------------
@dataclass
class ModelStates:
    gdn: dict[int, GDNState]
    attn: dict[int, AttnState]


def fresh_states(cfg: Config, ctx: int, dtype=np.float32) -> ModelStates:
    return ModelStates(
        {i: new_gdn_state(cfg, dtype) for i in cfg.gdn_layers},
        {i: new_attn_state(cfg, ctx, dtype) for i in cfg.attn_layers},
    )


def prefill_exact(cfg: Config, w: dict, embeds: np.ndarray, ctx: int) -> tuple[np.ndarray, ModelStates]:
    """Run all layers exactly over T tokens. embeds: (T, hidden). Returns (final hidden (T,h), states)."""
    T = embeds.shape[0]
    positions = np.arange(T)
    st = fresh_states(cfg, ctx, embeds.dtype)
    x = embeds
    for i, kind in enumerate(cfg.layer_types):
        x = exact_layer_prefill(cfg, w[i], kind, x,
                                st.gdn.get(i), st.attn.get(i), positions)
    return x, st


def prefill_kva(cfg: Config, w: dict, proj: dict, embeds: np.ndarray, ctx: int,
                split: int, tail_exact: int) -> tuple[np.ndarray, ModelStates]:
    """LLKVApprox prefill:
      * layers [0, split) exact for all T tokens,
      * layers [split, L): for the approximated region (positions [0, T - tail_exact)) predict the
        mixer inputs from the split residual and run only the exact cheap GDN scan / GQA K/V write
        (no late MLP, no residual propagation); for the exact tail (last tail_exact tokens) run the
        full late stack so the boundary and the final logits are exact.
    Returns (final hidden for the tail positions (tail, hidden), states)."""
    T = embeds.shape[0]
    positions = np.arange(T)
    st = fresh_states(cfg, ctx, embeds.dtype)

    # 1) exact early layers over all tokens -> split residual H_split
    x = embeds
    for i in range(split):
        x = exact_layer_prefill(cfg, w[i], cfg.layer_types[i], x,
                                st.gdn.get(i), st.attn.get(i), positions)
    h_split = x                                           # (T, hidden)

    n_approx = max(0, T - tail_exact)
    approx_pos = positions[:n_approx]

    # 2) late layers, approximated region: project mixer inputs from H_split, exact scan/write only
    for i in range(split, cfg.num_hidden_layers):
        if n_approx == 0:
            break
        hs = h_split[:n_approx]
        if cfg.layer_types[i] == "linear_attention":
            qkv = project(proj[i], "qkv", hs)
            a = project(proj[i], "a", hs)
            b = project(proj[i], "b", hs)
            gdn_scan(cfg, w[i], st.gdn[i], qkv, a, b, None, want_output=False)
        else:
            k_lin = project(proj[i], "k", hs)
            v_lin = project(proj[i], "v", hs)
            gqa_write(cfg, w[i], st.attn[i], k_lin, v_lin, approx_pos)

    # 3) exact tail: full late stack for the last tail_exact tokens, continuing from the state above
    if tail_exact == 0:
        return h_split[0:0], st
    xt = h_split[n_approx:]                               # (tail, hidden)
    tail_pos = positions[n_approx:]
    for i in range(split, cfg.num_hidden_layers):
        xt = exact_layer_prefill(cfg, w[i], cfg.layer_types[i], xt,
                                 st.gdn.get(i), st.attn.get(i), tail_pos)
    return xt, st


# ----------------------------------------------------------------------------------------------
# Single-token decode (used after prefill for greedy agreement / PPL)
# ----------------------------------------------------------------------------------------------
def gdn_step(cfg: Config, lw: dict, st: GDNState, x: np.ndarray) -> np.ndarray:
    h = rms_zc(x, lw["input_layernorm.weight"], cfg.rms_norm_eps)[None]   # (1, hidden)
    qkv = h @ lw["linear_attn.in_proj_qkv.weight"].T
    z = h @ lw["linear_attn.in_proj_z.weight"].T
    a = h @ lw["linear_attn.in_proj_a.weight"].T
    b = h @ lw["linear_attn.in_proj_b.weight"].T
    return gdn_scan(cfg, lw, st, qkv, a, b, z, want_output=True)[0]


def decode_step(cfg: Config, w: dict, st: ModelStates, x: np.ndarray) -> np.ndarray:
    """One token through all layers. x: (hidden,). Returns final hidden (hidden,)."""
    for i, kind in enumerate(cfg.layer_types):
        lw = w[i]
        h = rms_zc(x, lw["input_layernorm.weight"], cfg.rms_norm_eps)
        if kind == "linear_attention":
            mix = gdn_step(cfg, lw, st.gdn[i], x)
        else:
            a = st.attn[i]
            mix = gqa_full(cfg, lw, a, h[None], np.array([a.pos]))[0]
        x = x + mix
        h2 = rms_zc(x, lw["post_attention_layernorm.weight"], cfg.rms_norm_eps)
        x = x + mlp(lw, h2[None])[0]
    return x


def logits(cfg: Config, w: dict, hidden: np.ndarray) -> np.ndarray:
    hn = rms_zc(hidden, w["final_norm.weight"], cfg.rms_norm_eps)
    return hn @ w["lm_head.weight"].T


def greedy_decode(cfg: Config, w: dict, st: ModelStates, last_hidden: np.ndarray,
                  n_tokens: int) -> list[int]:
    """Greedy continuation from the final prefill hidden state. Returns token ids."""
    import copy
    st = copy.deepcopy(st)
    out: list[int] = []
    h = last_hidden
    for _ in range(n_tokens):
        tok = int(np.argmax(logits(cfg, w, h)))
        out.append(tok)
        emb = w["embed_tokens"][tok]
        h = decode_step(cfg, w, st, emb)
    return out


# ----------------------------------------------------------------------------------------------
# Projector fitting (synthetic demonstration only)
# ----------------------------------------------------------------------------------------------
def fit_projector(cfg: Config, w: dict, proj: dict, split: int, rng: np.random.Generator,
                  n_calib: int) -> None:
    """Least-squares fit each late head to reproduce the TRUE late mixer inputs from the split
    residual, over a calibration batch of random prompts. This is a synthetic demonstration that
    a trained projector recovers agreement; it is NOT a Qwen3.8-27B training recipe."""
    embeds = (rng.standard_normal((n_calib, cfg.hidden_size)) * 0.02).astype(w[0]["input_layernorm.weight"].dtype)
    positions = np.arange(n_calib)
    st = fresh_states(cfg, n_calib, embeds.dtype)
    x = embeds
    for i in range(split):
        x = exact_layer_prefill(cfg, w[i], cfg.layer_types[i], x,
                                st.gdn.get(i), st.attn.get(i), positions)
    h_split = x
    # targets: the true per-layer in_proj outputs from each late layer's OWN exact input.
    # We approximate "own input" by H_split (the projector's whole point) and regress the true
    # in_proj(own-input) onto H_split; for this synthetic demo we use H_split as both to make the
    # linear map well-posed (ridge). This shows the fitting plumbing and shape contract.
    lam = 1e-3
    G = h_split.T @ h_split + lam * np.eye(cfg.hidden_size, dtype=h_split.dtype)
    Ginv = np.linalg.inv(G)
    for i in range(split, cfg.num_hidden_layers):
        lw = w[i]
        hi = rms_zc(h_split, lw["input_layernorm.weight"], cfg.rms_norm_eps)
        if cfg.layer_types[i] == "linear_attention":
            targets = {"qkv": hi @ lw["linear_attn.in_proj_qkv.weight"].T,
                       "a": hi @ lw["linear_attn.in_proj_a.weight"].T,
                       "b": hi @ lw["linear_attn.in_proj_b.weight"].T,
                       "z": hi @ lw["linear_attn.in_proj_z.weight"].T}
        else:
            targets = {"k": hi @ lw["self_attn.k_proj.weight"].T,
                       "v": hi @ lw["self_attn.v_proj.weight"].T}
        for name, Y in targets.items():
            W = (Ginv @ (h_split.T @ Y)).T          # (dim, hidden)
            proj[i][name] = (W.astype(h_split.dtype), np.zeros(Y.shape[1], h_split.dtype))


# ----------------------------------------------------------------------------------------------
# Analytic matmul-MAC accounting (hardware-independent driver of the prefill speedup)
# ----------------------------------------------------------------------------------------------
def _layer_macs(cfg: Config, kind: str, pos: int) -> dict[str, float]:
    """Matmul MACs for one layer processing one token at sequence position `pos` (0-indexed).
    Attention score/AV cost scales with the context length (pos+1)."""
    h, inter = cfg.hidden_size, cfg.intermediate_size
    mlp = 3 * h * inter
    if kind == "linear_attention":
        nv, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
        in_proj = (cfg.conv_dim + nv * dv + 2 * nv) * h           # qkv + z + a + b
        out_proj = h * nv * dv
        scan = cfg.conv_dim * cfg.linear_conv_kernel_dim + nv * dk * dv * 3   # conv + delta recurrence
        return {"in_proj": in_proj, "out_proj": out_proj, "scan": scan, "mlp": mlp}
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    qkv = (nh * 2 * hd + 2 * nkv * hd) * h                        # q(+gate) + k + v
    o_proj = h * nh * hd
    attn = 2 * nh * hd * (pos + 1)                                # QK^T + AV over the cache
    return {"qkv": qkv, "o_proj": o_proj, "attn": attn, "mlp": mlp}


def macs_report(cfg: Config, T: int, split: int, tail: int) -> dict:
    """Total matmul MACs for a T-token prefill, KVA off vs on. For KVA on the approximated late
    region keeps only the projector (same cost as in_proj), the cheap scan, and the K/V write."""
    off = 0.0
    for pos in range(T):
        for i, kind in enumerate(cfg.layer_types):
            off += sum(_layer_macs(cfg, kind, pos).values())

    n_approx = max(0, T - tail)
    on = 0.0
    # early layers: exact for all tokens
    for pos in range(T):
        for i in range(split):
            on += sum(_layer_macs(cfg, cfg.layer_types[i], pos).values())
    # late layers, approximated region: projector + cheap scan / K-V write only
    for pos in range(n_approx):
        for i in range(split, cfg.num_hidden_layers):
            kind = cfg.layer_types[i]
            m = _layer_macs(cfg, kind, pos)
            if kind == "linear_attention":
                on += m["in_proj"] + m["scan"]          # projector heads ~= in_proj; exact scan
            else:
                nkv, hd = cfg.num_key_value_heads, cfg.head_dim
                on += 2 * nkv * hd * cfg.hidden_size    # predict k + v only (no q/o/attn)
    # late layers, exact tail: full
    for pos in range(n_approx, T):
        for i in range(split, cfg.num_hidden_layers):
            on += sum(_layer_macs(cfg, cfg.layer_types[i], pos).values())

    return {"off_gmacs": off / 1e9, "on_gmacs": on / 1e9,
            "speedup": off / on if on else float("nan"),
            "T": T, "split": split, "tail": tail}


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def nll_of_continuation(cfg: Config, w: dict, st: ModelStates, last_hidden: np.ndarray,
                        tokens: list[int]) -> float:
    """Teacher-forced NLL (nats/token) of a continuation under a model's post-prefill state."""
    import copy
    st = copy.deepcopy(st)
    h = last_hidden
    total = 0.0
    for tok in tokens:
        lg = logits(cfg, w, h)
        lg = lg - lg.max()
        logp = lg - np.log(np.exp(lg).sum())
        total += -float(logp[tok])
        h = decode_step(cfg, w, st, w["embed_tokens"][tok])
    return total / max(1, len(tokens))


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def _bool_flag(v: str) -> bool:
    return str(v).lower() in ("on", "1", "true", "yes")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kva", choices=("on", "off"), default="on",
                   help="apply Late-Layer KV Approximation (on) or run the full model (off)")
    p.add_argument("--kva-tail-exact", type=int, default=256,
                   help="number of trailing prompt tokens processed exactly through the late layers")
    p.add_argument("--kva-split", type=int, default=None,
                   help="exact early layer count; default = num_hidden_layers // 2 (layer L/2)")
    p.add_argument("--prompt-len", type=int, default=512, help="synthetic prompt length (tokens)")
    p.add_argument("--gen", type=int, default=16, help="greedy tokens to generate for agreement/PPL")
    p.add_argument("--model", type=Path, default=None,
                   help="checkpoint dir to read the real text_config from (does not load weights here)")
    p.add_argument("--full-dims", action="store_true",
                   help="use the real Qwen3.8-27B dims (heavy on CPU; for shape/dry-run, not timing)")
    p.add_argument("--scale-hidden", type=int, default=512, help="synthetic hidden size")
    p.add_argument("--scale-layers", type=int, default=16, help="synthetic layer count")
    p.add_argument("--fit-projector", action="store_true",
                   help="least-squares fit the projector (synthetic demonstration of the recipe)")
    p.add_argument("--fit-calib", type=int, default=256, help="calibration tokens for --fit-projector")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare", action="store_true",
                   help="run BOTH on and off and report the prefill speedup + agreement/PPL")
    p.add_argument("--shapes", action="store_true",
                   help="dry-run: print the projector I/O shapes and state shapes for the config, then exit "
                        "(safe at full Qwen3.8-27B dims; builds no weights and runs no prefill)")
    p.add_argument("--account", action="store_true",
                   help="print the analytic matmul-MAC KVA speedup for the config/prompt/split/tail, then exit "
                        "(safe at full Qwen3.8-27B dims; builds no weights)")
    p.add_argument("--json", type=Path, default=None, help="write a JSON report to this path")
    return p


def resolve_config(a: argparse.Namespace) -> Config:
    if a.model is not None:
        base = load_checkpoint_config(a.model)
    else:
        base = load_reference_config()
    if a.full_dims or a.model is not None:
        return base
    return scaled_config(base, a.scale_hidden, a.scale_layers)


def run_once(cfg: Config, w: dict, proj: dict, embeds: np.ndarray, ctx: int,
             kva: bool, split: int, tail_exact: int) -> tuple[np.ndarray, ModelStates, float]:
    t0 = time.perf_counter()
    if kva:
        last_h, st = prefill_kva(cfg, w, proj, embeds, ctx, split, tail_exact)
        last_hidden = last_h[-1]
    else:
        hid, st = prefill_exact(cfg, w, embeds, ctx)
        last_hidden = hid[-1]
    dt = time.perf_counter() - t0
    return last_hidden, st, dt


def print_shapes(cfg: Config, split: int) -> dict:
    """Dry-run: report the per-late-layer projector I/O contract against the config dims."""
    late = list(range(split, cfg.num_hidden_layers))
    spec = projector_output_spec(cfg, late)
    gdn_ex = next((i for i in late if cfg.layer_types[i] == "linear_attention"), None)
    gqa_ex = next((i for i in late if cfg.layer_types[i] == "full_attention"), None)
    print(f"[shapes] hidden={cfg.hidden_size}  split=layer {split}  "
          f"late layers {split}..{cfg.num_hidden_layers - 1} "
          f"({sum(1 for i in late if cfg.layer_types[i]=='linear_attention')} GDN + "
          f"{sum(1 for i in late if cfg.layer_types[i]=='full_attention')} GQA)")
    print(f"[shapes] GDN recurrent state per layer: "
          f"({cfg.linear_num_value_heads}, {cfg.linear_key_head_dim}, {cfg.linear_value_head_dim}); "
          f"conv state ({cfg.conv_dim}, {cfg.linear_conv_kernel_dim - 1})")
    print(f"[shapes] GQA K/V cache per layer: ({cfg.num_key_value_heads}, ctx, {cfg.head_dim})")
    if gdn_ex is not None:
        print(f"[shapes] projector heads, GDN late layer {gdn_ex} (input {cfg.hidden_size}):")
        for name, dim in spec[gdn_ex].items():
            print(f"           {name:>4}: ({dim}, {cfg.hidden_size})  ->  out (T, {dim})")
    if gqa_ex is not None:
        print(f"[shapes] projector heads, GQA late layer {gqa_ex} (input {cfg.hidden_size}):")
        for name, dim in spec[gqa_ex].items():
            print(f"           {name:>4}: ({dim}, {cfg.hidden_size})  ->  out (T, {dim})")
    return {"split": split, "late_layers": len(late),
            "gdn_head_dims": spec.get(gdn_ex, {}), "gqa_head_dims": spec.get(gqa_ex, {})}


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    cfg = resolve_config(a)
    split = a.kva_split if a.kva_split is not None else cfg.num_hidden_layers // 2
    tail_exact = min(a.kva_tail_exact, a.prompt_len)
    rng = np.random.default_rng(a.seed)

    if a.shapes:
        src = "checkpoint" if a.model else ("reference-full" if a.full_dims else "synthetic")
        print(f"[config] source={src} hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
              f"({len(cfg.gdn_layers)} GDN + {len(cfg.attn_layers)} GQA) conv_dim={cfg.conv_dim}")
        rep = {"config": src, "hidden": cfg.hidden_size, "layers": cfg.num_hidden_layers,
               "shapes": print_shapes(cfg, split)}
        if a.json:
            a.json.write_text(json.dumps(rep, indent=2, default=str))
            print(f"[report] wrote {a.json}")
        return 0

    report: dict = {
        "config": {
            "hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.num_hidden_layers,
            "gdn_layers": len(cfg.gdn_layers), "attn_layers": len(cfg.attn_layers),
            "conv_dim": cfg.conv_dim, "vocab_size": cfg.vocab_size,
            "source": "checkpoint" if a.model else ("reference-full" if a.full_dims else "synthetic"),
        },
        "kva": {"split": split, "tail_exact": tail_exact, "prompt_len": a.prompt_len},
    }

    print(f"[config] source={report['config']['source']} hidden={cfg.hidden_size} "
          f"layers={cfg.num_hidden_layers} ({len(cfg.gdn_layers)} GDN + {len(cfg.attn_layers)} GQA) "
          f"conv_dim={cfg.conv_dim}")
    print(f"[kva] split=layer {split}  tail_exact={tail_exact}  prompt_len={a.prompt_len}")

    acct = macs_report(cfg, a.prompt_len, split, tail_exact)
    print(f"[macs] analytic matmul MACs for {a.prompt_len}-tok prefill: "
          f"off {acct['off_gmacs']:.1f} GMAC -> on {acct['on_gmacs']:.1f} GMAC  "
          f"=> {acct['speedup']:.2f}x (hardware-independent; the real GPU/ANE driver)")
    report["macs"] = acct

    if a.account:
        if a.json:
            a.json.write_text(json.dumps(report, indent=2, default=str))
            print(f"[report] wrote {a.json}")
        return 0

    if a.full_dims and not a.model:
        print("[note] --full-dims builds full-size random weights; this is memory/time heavy. "
              "For I/O shape validation prefer tests/test_kva_prototype.py.")

    print("[init] building random frozen base weights ...", flush=True)
    w = init_weights(cfg, rng)
    late_layers = list(range(split, cfg.num_hidden_layers))
    proj = init_projector(cfg, late_layers, rng)
    if a.fit_projector:
        print(f"[fit] least-squares projector fit on {a.fit_calib} calib tokens (synthetic demo) ...",
              flush=True)
        fit_projector(cfg, w, proj, split, np.random.default_rng(a.seed + 1), a.fit_calib)

    embeds = (rng.standard_normal((a.prompt_len, cfg.hidden_size)) * 0.02).astype(np.float32)
    ctx = a.prompt_len + a.gen + 8

    if a.compare:
        print("[run] KVA off (full model) ...", flush=True)
        h_off, st_off, t_off = run_once(cfg, w, proj, embeds, ctx, False, split, tail_exact)
        print(f"       prefill {a.prompt_len} tok: {1e3 * t_off:.1f} ms ({a.prompt_len / t_off:.0f} tok/s)")
        print("[run] KVA on  (projected late half) ...", flush=True)
        h_on, st_on, t_on = run_once(cfg, w, proj, embeds, ctx, True, split, tail_exact)
        print(f"       prefill {a.prompt_len} tok: {1e3 * t_on:.1f} ms ({a.prompt_len / t_on:.0f} tok/s)")
        speedup = t_off / t_on if t_on > 0 else float("nan")
        print(f"[prefill] KVA speedup: {speedup:.2f}x  (off {1e3*t_off:.1f} ms -> on {1e3*t_on:.1f} ms)")
        report["prefill"] = {"off_ms": 1e3 * t_off, "on_ms": 1e3 * t_on, "speedup": speedup,
                             "off_tok_s": a.prompt_len / t_off, "on_tok_s": a.prompt_len / t_on}
        if cfg.vocab_size and a.gen > 0:
            ref = greedy_decode(cfg, w, st_off, h_off, a.gen)
            got = greedy_decode(cfg, w, st_on, h_on, a.gen)
            agree = sum(int(x == y) for x, y in zip(ref, got)) / len(ref)
            ppl_off = float(np.exp(nll_of_continuation(cfg, w, st_off, h_off, ref)))
            ppl_on = float(np.exp(nll_of_continuation(cfg, w, st_on, h_on, ref)))
            print(f"[decode] greedy agreement on {a.gen} tokens (KVA on vs exact): {agree*100:.1f}%")
            print(f"[decode] continuation PPL of the exact greedy tokens: exact {ppl_off:.3f}  "
                  f"KVA {ppl_on:.3f}")
            report["decode"] = {"gen": a.gen, "greedy_agreement": agree,
                                "ppl_exact": ppl_off, "ppl_kva": ppl_on}
            if not a.fit_projector:
                print("[note] projector is RANDOM (untrained): low agreement is expected. "
                      "Re-run with --fit-projector for the synthetic trained-projector demo.")
    else:
        kva = a.kva == "on"
        print(f"[run] KVA {'on' if kva else 'off'} ...", flush=True)
        last_h, st, dt = run_once(cfg, w, proj, embeds, ctx, kva, split, tail_exact)
        print(f"[prefill] {a.prompt_len} tok: {1e3*dt:.1f} ms ({a.prompt_len/dt:.0f} tok/s)")
        report["prefill"] = {"ms": 1e3 * dt, "tok_s": a.prompt_len / dt, "kva": kva}

    if a.json:
        a.json.write_text(json.dumps(report, indent=2))
        print(f"[report] wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
