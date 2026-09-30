"""Core AI port of the Qwen3.8-27B v4 chunk L00-03 (3 Gated DeltaNet + 1 gated attention; T-row lazy-commit DeltaNet
with host-owned conv / rec / pend buffers; KV caches as read-only inputs with mask (1, CTX), block rows out as
k3_new / v3_new) - a torch mirror of qwen38_ane_chunk.py (gdn_lazy_block, delta_core / delta_out, attn_prefill KV_IN,
rms_hidden, rot_conv, lut_linear) on the exported weights (scripts/coreai_chunk_ref.py dump).

    cpu    : torch (fp32 / fp16 CPU) vs the Core ML v4 chunk outputs (coreai_chunk_ref.py ref) - validates the port
    export : Core AI program with entry points v8_<ctx> (T=8 at each ctx in CTXS), exact LUTs injected (vector 2x16 MLP,
             scalar LUT4 mixers; per-channel scales as a mul after the conv), -> artifacts_chunk/<name>.aimodel
    ane    : load on the ANE, parity vs Core ML per entry point, call ms, wired memory with 1 .. n entry points, placement
    .venv/bin/python coreai_chunk_port.py cpu|export|ane   (LAYERS=0,1,2,3 CTXS=2048,8192)"""
from __future__ import annotations

import asyncio
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(os.path.expanduser("~/Models/vq27b/coreai_port"))
CFG = json.loads(Path(os.path.expanduser("~/Models/Qwen3.8-27B/config.json")).read_text())["text_config"]
LAYERS = [int(x) for x in os.environ.get("LAYERS", "0,1,2,3").split(",")]
CTXS = [int(x) for x in os.environ.get("CTXS", "2048,8192").split(",")]
T, P = 8, 8
NAME = os.environ.get("NAME", f"chunk_L{''.join(map(str, LAYERS))}_" + "_".join(str(c // 1024) + "k" for c in CTXS))
ROOT = Path(__file__).resolve().parent / "artifacts_chunk"
CACHE = Path.home() / "Library/Caches/coreai-cache"
KNOWN_LUTS: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

nk, nv = CFG["linear_num_key_heads"], CFG["linear_num_value_heads"]
dk, dv = CFG["linear_key_head_dim"], CFG["linear_value_head_dim"]
kd, vd = nk * dk, nv * dv
cdim = 2 * kd + vd
nh, nkv, hd, hid = CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"], CFG["hidden_size"]
grp, rot = nh // nkv, int(hd * CFG["rope_parameters"]["partial_rotary_factor"])
EPS = CFG["rms_norm_eps"]


def wired_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * page / 2**30


# ---- weights -----------------------------------------------------------------------------------------------------
class QConv(nn.Module):
    """1x1 conv with the export's weight: LUT (dense lut[idx] here, registered for exact palettization) + per-channel
    scale as a mul after the conv, int8 (dequantized fp16) or dense."""

    def __init__(self, W: np.lib.npyio.NpzFile, key: str) -> None:
        super().__init__()
        self.register_buffer("scale", None)
        if f"{key}/lut" in W:
            lut, idx = W[f"{key}/lut"], W[f"{key}/idx"]
            k, cd = lut.shape
            w = lut[idx].transpose(0, 2, 1).reshape(idx.shape[0] * cd, idx.shape[1]).astype(np.float16)
            KNOWN_LUTS[w.tobytes()] = (lut, idx)
            if f"{key}/scale" in W:
                self.scale = torch.from_numpy(W[f"{key}/scale"].astype(np.float16)).view(1, -1, 1, 1)
        elif f"{key}/int8" in W:
            w = (W[f"{key}/int8"].astype(np.float32) * W[f"{key}/scale"].astype(np.float32)[:, None]).astype(np.float16)
        else:
            w = W[f"{key}/dense"].astype(np.float16)
        self.conv = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=False)
        with torch.no_grad():
            self.conv.weight = nn.Parameter(torch.from_numpy(w).view(w.shape[0], w.shape[1], 1, 1), requires_grad=False)

    def forward(self, x):
        y = self.conv(x)
        return y if self.scale is None else y * self.scale


class Hadamard(nn.Module):
    """x (1, n, 1, T) -> x M, M = blockdiag(diag(signs) H_1024) / 32 (the pipeline's online rotation, same seeds)."""

    def __init__(self, n: int, seed: int, block: int = 1024) -> None:
        super().__init__()
        from scipy.linalg import hadamard
        h = hadamard(block)
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], n)
        wt = np.concatenate([(signs[b * block:(b + 1) * block, None] * h).T for b in range(n // block)])
        self.conv = nn.Conv2d(n, n, 1, groups=n // block, bias=False)
        with torch.no_grad():
            self.conv.weight = nn.Parameter(torch.from_numpy((wt / np.sqrt(block)).astype(np.float16)).view(n, block, 1, 1),
                                            requires_grad=False)

    def forward(self, x):
        return self.conv(x)


def buf(m: nn.Module, name: str, a: np.ndarray) -> None:
    m.register_buffer(name, torch.from_numpy(np.ascontiguousarray(a, np.float16)))


def rms_hidden(x, w_plus):
    """Scale-free RMSNorm over channels (the drafter's rms_robust): xs = x / max|x| keeps the squares in [0, 1], so
    neither tiny embeddings (x/64 squares underflow fp16: Core AI's ANE lowering then mis-normalizes, 0.70x at layer
    0) nor massive activations (overflow) break it. Exact up to rounding: rsqrt(mean(xs^2) + eps / m^2) * xs."""
    m = x.abs().amax(1, keepdim=True).clamp_min(1e-3)
    xs = x / m
    return xs * torch.rsqrt((xs * xs).mean(1, keepdim=True) + EPS / (m * m)) * w_plus


def rms_last(x, w):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * w


# ---- layers ------------------------------------------------------------------------------------------------------
class GDN(nn.Module):
    def __init__(self, W, i: int) -> None:
        super().__init__()
        p = f"{i}/linear_attn."
        self.qkv, self.z, self.out = QConv(W, p + "in_proj_qkv.weight"), QConv(W, p + "in_proj_z.weight"), QConv(W, p + "out_proj.weight")
        self.a = QConv({f"{p}in_proj_a.weight/dense": W[p + "in_proj_a.weight"]}, p + "in_proj_a.weight")
        self.b = QConv({f"{p}in_proj_b.weight/dense": W[p + "in_proj_b.weight"]}, p + "in_proj_b.weight")
        buf(self, "cw", W[p + "conv1d.weight"][:, 0].T)                                   # (4, cdim)
        buf(self, "neg_a", (-np.exp(W[p + "A_log"])).reshape(nv, 1, 1))
        buf(self, "dt", W[p + "dt_bias"].reshape(nv, 1, 1))
        buf(self, "normw", W[p + "norm.weight"])
        i_, j_ = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
        buf(self, "l_inc", (i_ >= j_))
        buf(self, "l_str", (i_ > j_))
        buf(self, "eye", np.eye(T))

    def core(self, kh, vh, beta, g):
        cum = (g.reshape(nv, 1, T) @ self.l_inc.T).reshape(nv, T, 1)
        pair = torch.exp(torch.clamp(cum - cum.reshape(nv, 1, T), max=0)) * self.l_inc
        kb, vb = kh * beta, vh * beta
        n = (kb @ kh.transpose(1, 2)) * (pair * self.l_str)
        # (I + N)^-1 [vb | kb exp(cum)] by forward substitution over the T rows: the doubling-inverse chain of
        # computed square matmuls (MIL: I - (I - N)(I + N) ...) keeps MPSGraph from placing the graph on the ANE
        rhs = torch.cat([vb, kb * torch.exp(cum)], -1)                                      # (nv, T, dv + dk)
        xs = [rhs[:, 0:1]]
        for t in range(1, T):
            xs.append(rhs[:, t:t + 1] - (n[:, t, 0:t].unsqueeze(-1) * torch.cat(xs, 1)).sum(1, keepdim=True))
        x = torch.cat(xs, 1)
        return cum, pair, x[..., :dv], x[..., dv:]

    def forward(self, h, conv_rows, conv_sel, rec, pend, commit, commit_last):
        qkv = self.qkv(h).reshape(cdim, T)
        z = self.z(h).reshape(nv, dv, T).permute(0, 2, 1)
        b, a = self.b(h).reshape(nv, T, 1), self.a(h).reshape(nv, T, 1)
        rows = torch.cat([conv_sel @ conv_rows, qkv.transpose(0, 1)], 0)                  # (T + 3, cdim)
        conv = rows[0:T] * self.cw[0:1] + rows[1:T + 1] * self.cw[1:2] + rows[2:T + 2] * self.cw[2:3] + rows[3:T + 3] * self.cw[3:4]
        conv = F.silu(conv).transpose(0, 1)
        qq, kk, vv = conv[:kd], conv[kd:2 * kd], conv[2 * kd:]

        def heads(t):
            t = t.reshape(nk, dk, T).permute(0, 2, 1)
            return t.reshape(nk, 1, T, dk).repeat(1, nv // nk, 1, 1).reshape(nv, T, dk)

        def l2n(t, s):
            return t * torch.rsqrt((t * t).sum(-1, keepdim=True) + 1e-6) * s
        qh, kh = l2n(heads(qq), dk ** -0.5), l2n(heads(kk), 1.0)
        vh = vv.reshape(nv, dv, T).permute(0, 2, 1)
        beta = torch.sigmoid(b)
        ad = a + self.dt   # softplus without exp overflow (the ANE's fp16 softplus returns 0 above ~11)
        g = (F.relu(ad) + torch.log(1 + torch.exp(-torch.abs(ad)))) * self.neg_a
        kp, up, wkp = pend[:, 0:P, 0:dk], pend[:, P:2 * P, 0:dv], pend[:, 2 * P:3 * P, 0:dk]
        cum_p = pend[:, 3 * P:3 * P + 1, 0:P].reshape(nv, P, 1)
        total = (cum_p * commit_last).sum(1, keepdim=True)
        kd_ = kp * commit * torch.exp(torch.clamp(total - cum_p, max=0))
        s1 = rec * torch.exp(total) + kd_.transpose(1, 2) @ (up - wkp @ rec)
        cum, pair, u, wk = self.core(kh, vh, beta, g)
        crow = torch.cat([cum.reshape(nv, 1, T), torch.zeros(nv, 1, dv - T, dtype=cum.dtype)], 2)
        pend_out = torch.cat([kh, u, wk, crow], 1)
        vn = u - wk @ s1
        o = (qh * torch.exp(cum)) @ s1 + ((qh @ kh.transpose(1, 2)) * pair) @ vn
        o = rms_last(o, self.normw) * F.silu(z)
        return self.out(o.permute(0, 2, 1).reshape(1, vd, 1, T)), rows, s1, pend_out


class Attn(nn.Module):
    def __init__(self, W, i: int) -> None:
        super().__init__()
        p = f"{i}/self_attn."
        self.q, self.k, self.v, self.o = (QConv(W, p + f"{m}_proj.weight") for m in "qkvo")
        buf(self, "qn", 1 + W[p + "q_norm.weight"])
        buf(self, "kn", 1 + W[p + "k_norm.weight"])
        i_, j_ = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
        buf(self, "causal", np.where(j_ <= i_, 0, -1e4))

    def forward(self, h, cos, sin, mask, k_st, v_st, ctx: int):
        def tmajor(x, c):
            return x.reshape(c, T).transpose(0, 1)
        qg = tmajor(self.q(h), 2 * nh * hd).reshape(T, nh, 2 * hd)
        qh, gate = qg[:, :, :hd], qg[:, :, hd:].reshape(T, nh * hd)
        qh = rms_last(qh, self.qn)
        kh = rms_last(tmajor(self.k(h), nkv * hd).reshape(T, nkv, hd), self.kn)
        vh = tmajor(self.v(h), nkv * hd).reshape(T, nkv, hd)
        c3, s3 = cos.reshape(T, 1, rot), sin.reshape(T, 1, rot)

        def rope(t):
            r, rest = t[..., :rot], t[..., rot:]
            rh = torch.cat([-r[..., rot // 2:], r[..., :rot // 2]], -1)
            return torch.cat([r * c3 + rh * s3, rest], -1)
        qh = rope(qh)
        kt, vt = rope(kh).permute(1, 0, 2), vh.permute(1, 0, 2)                            # (nkv, T, hd)
        qg4 = qh.reshape(T, nkv, grp, hd).permute(1, 2, 0, 3).reshape(nkv, grp * T, hd)
        sc_h = ((qg4 @ k_st.transpose(1, 2)) * hd ** -0.5).reshape(nkv, grp, T, ctx) + mask.reshape(1, 1, 1, ctx)
        sc_b = ((qg4 @ kt.transpose(1, 2)) * hd ** -0.5).reshape(nkv, grp, T, T) + self.causal
        pr = torch.softmax(torch.cat([sc_h, sc_b], -1), -1).reshape(nkv, grp * T, ctx + T)
        o = pr[:, :, :ctx] @ v_st + pr[:, :, ctx:] @ vt
        o = o.reshape(nkv, grp, T, hd).permute(2, 0, 1, 3).reshape(T, nh * hd) * torch.sigmoid(gate)
        return self.o(o.transpose(0, 1).reshape(1, nh * hd, 1, T)), kt, vt


class Layer(nn.Module):
    def __init__(self, W, i: int) -> None:
        super().__init__()
        self.kind = CFG["layer_types"][i]
        self.mix = GDN(W, i) if self.kind == "linear_attention" else Attn(W, i)
        buf(self, "ln1", 1 + W[f"{i}/input_layernorm.weight"].reshape(1, -1, 1, 1))
        buf(self, "ln2", 1 + W[f"{i}/post_attention_layernorm.weight"].reshape(1, -1, 1, 1))
        self.gate, self.up, self.down = (QConv(W, f"{i}/mlp.{m}_proj.weight") for m in ("gate", "up", "down"))
        seeds = W[f"{i}/mlp.rotation"] if f"{i}/mlp.rotation" in W else None
        self.rin = Hadamard(hid, int(seeds[0])) if seeds is not None else None
        self.rmid = Hadamard(CFG["intermediate_size"], int(seeds[1])) if seeds is not None else None

    def mlp(self, x):
        h = rms_hidden(x, self.ln2)
        h = self.rin(h) if self.rin is not None else h
        a = F.silu(self.gate(h)) * self.up(h)
        a = self.rmid(a) if self.rmid is not None else a
        return x + self.down(a)


class Chunk(nn.Module):
    def __init__(self, W, ctx: int) -> None:
        super().__init__()
        self.ctx = ctx
        self.layers = nn.ModuleList(Layer(W, i) for i in LAYERS)
        self.gdn_j = [j for j, i in enumerate(LAYERS) if CFG["layer_types"][i] == "linear_attention"]
        self.att_j = [j for j, i in enumerate(LAYERS) if CFG["layer_types"][i] != "linear_attention"]

    def input_names(self):
        return (["x", "cos", "sin", "mask", "conv_sel", "commit", "commit_last"]
                + [f"{s}{j}" for j in self.gdn_j for s in ("conv", "rec", "pend")]
                + [f"{s}{j}" for j in self.att_j for s in ("k", "v")])

    def output_names(self):
        return (["y"] + [f"{s}{j}_out" for j in self.gdn_j for s in ("conv", "rec", "pend")]
                + [f"{s}{j}_new" for j in self.att_j for s in ("k", "v")])

    def forward(self, x, cos, sin, mask, conv_sel, commit, commit_last, *states):
        it = iter(states)
        gdn_in = {j: (next(it), next(it), next(it)) for j in self.gdn_j}
        att_in = {j: (next(it), next(it)) for j in self.att_j}
        gdn_out, att_out = [], []
        for j, layer in enumerate(self.layers):
            h = rms_hidden(x, layer.ln1)
            if j in gdn_in:
                cr, rc, pd = gdn_in[j]
                y, rows, s1, pend_out = layer.mix(h, cr, conv_sel, rc, pd, commit, commit_last)
                gdn_out += [rows, s1, pend_out]
            else:
                ks, vs = att_in[j]
                y, kt, vt = layer.mix(h, cos, sin, mask, ks, vs, self.ctx)
                att_out += [kt, vt]
            x = layer.mlp(x + y)
        return (x, *gdn_out, *att_out)


# ---- modes -------------------------------------------------------------------------------------------------------
def load_weights():
    return np.load(DATA / "chunk_L00-03_weights.npz")


def ref_io(ctx: int, call: int, names):
    R = np.load(DATA / f"chunk_L00-03_ref_ctx{ctx}.npz")
    ins = [R[f"c{call}/in/{n}"] for n in names]
    return ins, {k.split("/out/")[1]: R[k] for k in R.files if k.startswith(f"c{call}/out/")}


def cos_err(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)), float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))


def mode_cpu():
    W = load_weights()
    ctx = CTXS[0]
    m = Chunk(W, ctx).eval()
    for dt in ({"fp32": torch.float32, "fp16": torch.float16}[os.environ.get("DT", "fp16")],):
        m = m.to(dt)
        for call in range(3):
            ins, ref = ref_io(ctx, call, m.input_names())
            with torch.no_grad():
                outs = m(*[torch.from_numpy(a.astype(np.float32)).to(dt) for a in ins])
            rep = [f"{n} cos {cos_err(o.float().numpy(), ref[n])[0]:.5f}" for n, o in zip(m.output_names(), outs)
                   if n in ref]
            print(f"[cpu {dt} ctx {ctx} call {call}] " + " | ".join(rep), flush=True)


def patch_palettizer():
    from coreai_opt.coreai_utils._utils.palettize_utils import LutParams
    from coreai_opt.coreai_utils.passes import weight_palettization as wp
    orig = wp._blockwise_compress

    def compress(original_data, mode, *args, **kwargs):
        known = KNOWN_LUTS.get(np.ascontiguousarray(original_data, np.float16).tobytes())
        if known is not None:
            lut, idx = known
            cd = lut.shape[1]
            extra = (1,) * (original_data.ndim - 2)
            return LutParams(indices=idx.reshape(*idx.shape, *extra).astype(np.uint8),
                             lut=lut.reshape(*(1,) * original_data.ndim, *lut.shape), vector_axis=0 if cd > 1 else None)
        return orig(original_data, "UNIQUE", *args, **kwargs)   # Hadamard (+-1/32): exact unique values
    wp._blockwise_compress = compress
    wp._is_cluster_dim_valid = lambda op, cluster_dim, channel_axis: list(op.result.type.shape)[channel_axis] % cluster_dim == 0


def mode_export():
    import coreai_torch
    from coreai_opt.casting import cast_to_16_bit_precision
    from coreai_opt.coreai_utils.common import CompressionGranularity
    from coreai_opt.coreai_utils.passes.weight_palettization import palettize_weights
    out = ROOT / f"{NAME}.aimodel"
    W = load_weights()
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    t0 = time.time()
    base = None
    for ctx in CTXS:
        m = Chunk(W, ctx).eval().to(torch.float16)
        if base is None:
            base = m
        else:  # same weight tensors in every entry point
            for a, b in zip(m.layers, base.layers):
                a.load_state_dict(b.state_dict(), assign=True)
        ins, _ = ref_io(ctx, 1, m.input_names())
        ex = tuple(torch.from_numpy(a.astype(np.float16)) for a in ins)
        ep = torch.export.export(m, ex, strict=False).run_decompositions(coreai_torch.get_decomp_table())
        cast_to_16_bit_precision(ep)
        conv.add_exported_program(ep, input_names=m.input_names(), output_names=m.output_names(),
                                  entrypoint_name=f"v8_{ctx // 1024}k")
        print(f"exported entry v8_{ctx // 1024}k ({time.time() - t0:.0f}s)", flush=True)
    prog = conv.to_coreai()
    patch_palettizer()
    # palettize before optimize(): optimize folds the per-channel-scale mul into the conv weight (W * s), which is no
    # longer a per-tensor LUT and no longer matches the exported indices
    prog = palettize_weights(prog, lut_dtype=None, n_bits=4, granularity=CompressionGranularity.PER_TENSOR,
                             cluster_dim=2, weight_num_threshold=1024, enable_fast_kmeans_mode=False)
    matched = sum(1 for k in KNOWN_LUTS)  # registered
    prog.optimize()
    ir = str(prog)
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)
    (ROOT / f"{NAME}.mlir").write_text(ir)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"saved {out.name}: {size:.0f} MB, {ir.count('lut_to_dense')} lut_to_dense ops "
          f"({time.time() - t0:.0f}s)", flush=True)


def placement(since: float) -> str:
    mans = [p for p in CACHE.glob("*/*/*/*/model.aimodelx/**/manifest.plist") if p.stat().st_mtime >= since]
    if not mans:
        return "cached"
    text = max(mans, key=lambda p: p.stat().st_mtime).read_bytes()
    n = text.count(b"_ANE_region_")
    return ("fully on ANE" if b"mps.fullyPlacedOnANE" in text else "NOT fully on ANE") + f", {n} ANE region refs"


async def mode_ane():
    from coreai.runtime import AIModel, NDArray
    from coreai.runtime._ndarray import StorageKind
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # coreai
    from coreai_util import specialization_for
    path = ROOT / f"{NAME}.aimodel"
    names = Chunk.input_names(type("X", (), {"gdn_j": [j for j, i in enumerate(LAYERS) if CFG["layer_types"][i] == "linear_attention"],
                                              "att_j": [j for j, i in enumerate(LAYERS) if CFG["layer_types"][i] != "linear_attention"]})())
    gc.collect()
    w0, t0 = wired_gb(), time.time()
    model = await AIModel.load(path, specialization_options=specialization_for("ane"))
    print(f"load {time.time() - t0:.0f}s, wired +{wired_gb() - w0:.2f} GB; {placement(t0)}", flush=True)
    fns = {}
    for ctx in CTXS:
        e = f"v8_{ctx // 1024}k"
        fns[ctx] = model.load_function(e)
        print(f"load_function {e}: wired +{wired_gb() - w0:.2f} GB", flush=True)
    tref = None
    if os.environ.get("TORCH_REF", "1") == "1":   # the port on the CPU (correct softplus) as a second reference
        tm = Chunk(load_weights(), CTXS[0]).eval().to(torch.float16)
        tref = {}
    for ctx in CTXS:
        fn = fns[ctx]
        for call in range(3):
            ins, ref = ref_io(ctx, call, names)
            tr = {}
            if tref is not None and ctx == CTXS[0]:
                with torch.no_grad():
                    tm.ctx = ctx
                    touts = tm(*[torch.from_numpy(a.astype(np.float16)) for a in ins])
                tr = {n_: o.float().numpy() for n_, o in zip(tm.output_names(), touts)}
            feed = {n: NDArray(a.astype(np.float16), StorageKind.IO_SURFACE) for n, a in zip(names, ins)}
            out = await fn(inputs=feed)
            rep = []
            for n_ in ("y", "rec0_out", "pend0_out", "conv0_out", "rec2_out", "k3_new", "v3_new"):
                if n_ in out and (n_ in ref or n_ in tr):
                    a_ = out[n_].numpy()
                    s_ = n_ + ":"
                    if n_ in ref and not (n_ == "y" and len(LAYERS) < 4):
                        s_ += f" vs CoreML {cos_err(a_, ref[n_])[0]:.5f}"
                    if n_ in tr:
                        s_ += f" vs torch {cos_err(a_, tr[n_])[0]:.5f}"
                    rep.append(s_)
            print(f"[ane ctx {ctx} call {call}] " + " | ".join(rep) + f"; wired +{wired_gb() - w0:.2f} GB", flush=True)
        ins, _ = ref_io(ctx, 1, names)
        feed = {n: NDArray(a.astype(np.float16), StorageKind.IO_SURFACE) for n, a in zip(names, ins)}
        ts = []
        for _ in range(15):
            t = time.perf_counter()
            await fn(inputs=feed)
            ts.append(1e3 * (time.perf_counter() - t))
        print(f"[ane ctx {ctx}] call median {np.median(ts):.2f} ms (p10 {np.percentile(ts, 10):.2f})", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "cpu":
        mode_cpu()
    elif mode == "export":
        mode_export()
    else:
        asyncio.run(mode_ane())
