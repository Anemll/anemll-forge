"""Full-attention core on the M6 ANE: the release formulation (qwen38_coreai_build.AttnW, V8 history) against rewrites,
timed through the Swift bridge at real history lengths.

One program holds L attention cores (real q/k norm weights of the first L full-attention layers) between their
projections: inputs are the q (with gate), k and v projection outputs in the chunk's (1, C, 1, T) layout, RoPE tables,
the mask and the V8 history (FP16 K, INT8 V codes, FP16 token/head scales); outputs are the o_proj input and the new
k / v rows. `ref` is AttnW.forward itself; `opt` with no options reproduces it and the options change one thing each.
Every variant's ANE output is compared with an FP32 host evaluation of the same inputs.

    <coreai venv>/bin/python scripts/m6_attn_bench.py check --variants ref,qscale
    <coreai venv>/bin/python scripts/m6_attn_bench.py build --variants ref,qscale --ctx 65472 --out DIR
    <coreai venv>/bin/python scripts/m6_attn_bench.py time  --variants ref,qscale --out DIR
Env: MODEL, MPSGRAPH_ANE_BONDED_COMPILE_MODE (default: the SoC policy, 2 on M6, 1 on M5)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
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
os.environ.setdefault("KV_CACHE_DTYPE", "v8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))
sys.path.insert(0, str(ROOT / "coreai" / "swift_bridge"))
sys.path.insert(0, str(ROOT / "scripts"))
import coreai_bridge as B  # noqa: E402
import inspect_coreai_cache as IC  # noqa: E402
import m6_entry_sweep as S  # noqa: E402
import qwen38_coreai_build as Bld  # noqa: E402

nh, nkv, hd, grp, rot = Bld.nh, Bld.nkv, Bld.hd, Bld.grp, Bld.rot
ATT_LAYERS = [i for i, t in enumerate(Bld.CFG["layer_types"]) if t != "linear_attention"]


# ---- the parameterized core ----------------------------------------------------------------------------------------
def attn_opt(self, h, cos, sin, mask, k_st, v_st, ctx, T, vscale=None, cache_v8=None, *, opt=None, host=False):
    """AttnW.forward (V8 history, stable global exp/sum in ATT_BLOCK tiles) with switchable pieces:
    qscale  fold hd^-0.5 into q (one (T, nh, hd) multiply instead of one per score tile)
    block   history tile width (release 16384)
    vtok    dequantize V with its token scale per KV head (no per-score scale multiply)
    denmm   softmax denominators as e @ ones (matmul) instead of a reduce_sum
    nomask / nomax   timing only: drop the mask add / the running max."""
    o_ = dict(opt or {})
    blk = o_.get("block_prefill", o_.get("block", Bld.ATT_BLOCK)) if T > Bld.P else o_.get("block", Bld.ATT_BLOCK)

    def tmajor(x, c):
        return x.reshape(c, T).transpose(0, 1)
    qg = tmajor(self.q(h), 2 * nh * hd).reshape(T, nh, 2 * hd)
    qh, gate = Bld.rms_last(qg[:, :, :hd], self.qn), qg[:, :, hd:].reshape(T, nh * hd)
    kh = Bld.rms_last(tmajor(self.k(h), nkv * hd).reshape(T, nkv, hd), self.kn)
    vh = tmajor(self.v(h), nkv * hd).reshape(T, nkv, hd)
    c3, s3 = cos.reshape(T, 1, rot), sin.reshape(T, 1, rot)

    def rope(t):
        r, rest = t[..., :rot], t[..., rot:]
        return torch.cat([r * c3 + torch.cat([-r[..., rot // 2:], r[..., :rot // 2]], -1) * s3, rest], -1)
    qh = rope(qh)
    kt, vt = rope(kh).permute(1, 0, 2), vh.permute(1, 0, 2)
    qg4 = qh.reshape(T, nkv, grp, hd).permute(1, 2, 0, 3).reshape(nkv, grp * T, hd)
    sc = hd ** -0.5
    if o_.get("qscale"):
        qg4, sc = qg4 * sc, None
    causal = (1 - Bld.tri(T, False).to(qg4.dtype)) * -1e4

    def scale(x):
        return x if sc is None else x * sc
    sc_b = scale(qg4 @ kt.transpose(1, 2)).reshape(nkv, grp * T, T) + causal.repeat(grp, 1)
    if host:
        v_unit = v_st.to(qg4.dtype) / 128
    elif o_.get("vtok"):
        v_unit = None
    else:
        v_unit = torch.ops.coreai.dequantize(v_st, self.v8_unit, zero_point=self.v8_zero, output_dtype=torch.float16)
    edges = list(range(0, ctx, blk)) + [ctx]
    spans = list(zip(edges[:-1], edges[1:]))

    def history(qg4, sc_b, vt, k_st, v_st, v_unit, vscale, nh_):
        scs = []
        for a, b in spans:
            s_ = scale(qg4 @ k_st[:, a:b].transpose(1, 2))
            scs.append(s_ if o_.get("nomask") else s_ + mask[:, a:b].reshape(1, 1, b - a))
        if o_.get("nomax"):
            m = torch.zeros_like(sc_b[..., :1])
        else:
            m = sc_b.amax(-1, keepdim=True)
            for s_ in scs:
                m = torch.maximum(m, s_.amax(-1, keepdim=True))
        e_b = torch.exp(sc_b - m)
        den, num = e_b.sum(-1, keepdim=True), e_b @ vt
        for s_, (a, b) in zip(scs, spans):
            e_ = torch.exp(s_ - m)
            if o_.get("denmm"):
                den = den + e_ @ torch.ones(b - a, 1, dtype=e_.dtype)
            else:
                den = den + e_.sum(-1, keepdim=True)
            if o_.get("vtok") and not host:
                vb = torch.stack([torch.ops.coreai.dequantize(v_st[j, a:b], vscale[j, a:b], zero_point=None,
                                                              output_dtype=torch.float16, axis=0) for j in range(nh_)])
                num = num + e_ @ vb
            elif o_.get("vtok"):
                num = num + e_ @ (v_st[:, a:b].to(e_.dtype) * vscale[:, a:b].reshape(nh_, b - a, 1))
            else:
                e_ = e_ * (vscale[:, a:b].reshape(nh_, 1, b - a) * 128)
                num = num + e_ @ v_unit[:, a:b]
        return num, den

    def history_batched(qg4, sc_b, vt):
        """All full-width tiles as one batched op over a (nkv, tiles, blk, hd) view; a narrower last tile apart."""
        nt = ctx // blk
        full = nt * blk
        s4 = scale(qg4.unsqueeze(1) @ k_st[:, :full].reshape(nkv, nt, blk, hd).transpose(-1, -2))
        s4 = s4 + mask[:, :full].reshape(1, nt, 1, blk)                                    # (nkv, nt, rows, blk)
        m = torch.maximum(sc_b.amax(-1, keepdim=True), s4.amax(-1, keepdim=True).amax(1))
        rest = None
        if full < ctx:
            rest = scale(qg4 @ k_st[:, full:].transpose(1, 2)) + mask[:, full:].reshape(1, 1, ctx - full)
            m = torch.maximum(m, rest.amax(-1, keepdim=True))
        e_b = torch.exp(sc_b - m)
        den, num = e_b.sum(-1, keepdim=True), e_b @ vt
        e4 = torch.exp(s4 - m.unsqueeze(1))
        den = den + e4.sum(-1, keepdim=True).sum(1)
        v4 = v_unit[:, :full].reshape(nkv, nt, blk, hd)
        e4 = e4 * (vscale[:, :full].reshape(nkv, nt, 1, blk) * 128)
        num = num + (e4 @ v4).sum(1)
        if rest is not None:
            e_ = torch.exp(rest - m)
            den = den + e_.sum(-1, keepdim=True)
            num = num + (e_ * (vscale[:, full:].reshape(nkv, 1, ctx - full) * 128)) @ v_unit[:, full:]
        return num, den

    if o_.get("batched"):
        num, den = history_batched(qg4, sc_b, vt)
    elif o_.get("perhead"):  # one tile chain per KV head
        parts = [history(qg4[j:j + 1], sc_b[j:j + 1], vt[j:j + 1], k_st[j:j + 1], v_st[j:j + 1],
                         None if v_unit is None else v_unit[j:j + 1], vscale[j:j + 1], 1) for j in range(nkv)]
        num, den = torch.cat([x[0] for x in parts], 0), torch.cat([x[1] for x in parts], 0)
    else:
        num, den = history(qg4, sc_b, vt, k_st, v_st, v_unit, vscale, nkv)
    o = num / den
    o = o.reshape(nkv, grp, T, hd).permute(2, 0, 1, 3).reshape(T, nh * hd) * torch.sigmoid(gate)
    return self.o(o.transpose(0, 1).reshape(1, nh * hd, 1, T)), kt, vt


VARIANTS = {
    "ref": None,                                  # AttnW.forward, unmodified
    "opt": {},
    "qscale": {"qscale": True},
    "b8k": {"block": 8192},
    "b32k": {"block": 32768},
    "b4k": {"block": 4096},
    "b2k": {"block": 2048},
    "b1k": {"block": 1024},
    "b2k_ph": {"block": 2048, "perhead": True},
    "b2k_bat": {"block": 2048, "batched": True},
    "b4k_bat": {"block": 4096, "batched": True},
    "b2k_pf4k": {"block": 2048, "block_prefill": 4096},
    "b4k_ph": {"block": 4096, "perhead": True},
    "vtok": {"vtok": True},
    "denmm": {"denmm": True},
    "nomask": {"nomask": True, "timing_only": True},
    "nomax": {"nomax": True, "timing_only": True},
}


# ---- program -------------------------------------------------------------------------------------------------------
class Pick(nn.Module):
    def __init__(self, i: int):
        super().__init__()
        self.i = i

    def forward(self, h):
        return h[self.i]


def norm_weights(layer: int) -> dict:
    wmap = json.loads((Bld.M.MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    pre = f"model.language_model.layers.{layer}.self_attn."
    out = {}
    for k in ("q_norm.weight", "k_norm.weight"):
        with safe_open(Bld.M.MODEL / wmap[pre + k], framework="pt") as f:
            out[k] = f.get_tensor(pre + k).float().numpy()
    return out


def make_attn(layer: int, variant: str, host: bool = False) -> nn.Module:
    nw = norm_weights(layer)
    p = f"{layer}/self_attn."
    W = {p + f"{m}_proj.weight/dense": np.zeros((8, 8)) for m in "qkvo"}
    W[p + "q_norm.weight"], W[p + "k_norm.weight"] = nw["q_norm.weight"], nw["k_norm.weight"]
    mod = Bld.AttnW(W, layer)
    mod.q, mod.k, mod.v, mod.o = Pick(0), Pick(1), Pick(2), nn.Identity()
    spec = VARIANTS[variant]
    if spec is not None or host:
        opt = {k: v for k, v in (spec or {}).items() if k != "timing_only"}
        mod.forward = types.MethodType(lambda self, *a, **k: attn_opt(self, *a, **k, opt=opt, host=host), mod)
    return mod


class Cores(nn.Module):
    def __init__(self, mods, ctx: int, T: int):
        super().__init__()
        self.mods, self.ctx, self.T = nn.ModuleList(mods), ctx, T

    def input_names(self):
        return ["q", "k", "v", "cos", "sin", "mask", "k_st", "v_st", "vscale"]

    def output_names(self):
        return [f"{s}{j}" for j in range(len(self.mods)) for s in ("o", "k_new", "v_new")]

    def forward(self, q, k, v, cos, sin, mask, k_st, v_st, vscale):
        out = []
        for m in self.mods:
            out += list(m((q, k, v), cos, sin, mask, k_st, v_st, self.ctx, self.T, vscale, True))
        return tuple(out)

    def example(self):
        return tuple(torch.from_numpy(x) for x in example_inputs(self.ctx, self.T, np.random.default_rng(0)))


def example_inputs(ctx: int, T: int, rng, visible: float = 0.75) -> list[np.ndarray]:
    def n(*shape, s=1.0):
        return (rng.standard_normal(shape) * s).astype(np.float16)
    pos = int(ctx * visible)
    inv = 1.0 / Bld.CFG["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
    ang = np.concatenate([np.outer(np.arange(pos, pos + T), inv)] * 2, axis=1)
    mask = np.full((1, ctx), -1e4, np.float16)
    mask[0, :pos] = 0
    vreal = rng.standard_normal((nkv, ctx, hd)) * 0.5
    vscale = (np.abs(vreal).max(-1) / 127).astype(np.float16)
    codes = np.clip(np.round(vreal / vscale[..., None].astype(np.float64)), -127, 127).astype(np.int8)
    return [n(1, 2 * nh * hd, 1, T), n(1, nkv * hd, 1, T), n(1, nkv * hd, 1, T, s=0.5),
            np.cos(ang).astype(np.float16), np.sin(ang).astype(np.float16), mask, n(nkv, ctx, hd), codes, vscale]


def build_cores(variant: str, layers: int, ctx: int, T: int, host: bool = False) -> Cores:
    return Cores([make_attn(ATT_LAYERS[j], variant, host) for j in range(layers)], ctx, T).eval().to(torch.float16)


_TRI = Bld.tri


def host_outputs(layers: int, ctx: int, T: int) -> list[np.ndarray]:
    """FP32 host evaluation of the release arithmetic (V codes / 128, scale on the scores)."""
    mod = build_cores("opt", layers, ctx, T, host=True).float()
    data = example_inputs(ctx, T, np.random.default_rng(0))
    ins = [torch.from_numpy(x).float() if x.dtype != np.int8 else torch.from_numpy(x) for x in data]
    Bld.tri = lambda n, strict: _TRI(n, strict).float()
    try:
        with torch.no_grad():
            return [o.float().numpy() for o in mod(*ins)]
    finally:
        Bld.tri = _TRI


def rel(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-30))


# ---- commands ------------------------------------------------------------------------------------------------------
def cmd_check(a):
    """Host: each variant's arithmetic (FP32, dequantize emulated) vs the release arithmetic."""
    for T in a.rows:
        ref = host_outputs(a.layers, a.ctx, T)
        for v in a.variants:
            spec = VARIANTS[v]
            if spec is None:
                continue
            opt = {k: x for k, x in spec.items() if k != "timing_only"}
            mod = build_cores("opt", a.layers, a.ctx, T, host=True).float()
            for m in mod.mods:
                m.forward = types.MethodType(lambda self, *x, **k: attn_opt(self, *x, **k, opt=opt, host=True), m)
            data = example_inputs(a.ctx, T, np.random.default_rng(0))
            ins = [torch.from_numpy(x).float() if x.dtype != np.int8 else torch.from_numpy(x) for x in data]
            Bld.tri = lambda n, strict: _TRI(n, strict).float()
            try:
                with torch.no_grad():
                    out = [o.float().numpy() for o in mod(*ins)]
            finally:
                Bld.tri = _TRI
            err = max(rel(x, y) for x, y in zip(out, ref))
            tag = " (timing only)" if spec.get("timing_only") else ""
            print(f"T={T:3d} ctx={a.ctx} {v:8s} max rel diff vs release arithmetic (fp32 host): {err:.2e}{tag}")


def cmd_build(a):
    a.out.mkdir(parents=True, exist_ok=True)
    for v in a.variants:
        dst = a.out / f"{v}.aimodel"
        if dst.exists() and not a.force:
            print(f"{v}: exists")
            continue
        entries = []
        for T in a.rows:
            mod = build_cores(v, a.layers, a.ctx, T)
            entries.append((f"t{T}", mod, mod.input_names(), mod.output_names()))
        t = time.time()
        try:
            Bld.save_program(entries, dst)
            print(f"{v}: built in {time.time() - t:.0f}s", flush=True)
        except Exception as e:  # a variant the converter rejects is a result, not a crash
            print(f"{v}: build failed: {str(e)[:300]}", flush=True)


def cmd_time(a):
    pkgs = [a.out / f"{v}.aimodel" for v in a.variants if (a.out / f"{v}.aimodel").exists()]
    build_id = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    refs, runs, res = {}, {}, {}
    cache = Path.home() / "Library/Caches/coreai-cache" / build_id / Path(sys.executable).name
    for p in pkgs:
        if a.cold:  # this probe package's own specialization only, so the load below is a cold compile
            shutil.rmtree(cache / (p / "main.hash").read_bytes().hex(), ignore_errors=True)
        t0 = time.time()
        try:
            m = B.Model(p, compute="ane")
        except B.BridgeError as e:
            print(f"{p.stem}: load failed: {str(e)[:200]}")
            continue
        print(f"{p.stem}: load {time.time() - t0:.1f}s{' (cold compile)' if a.cold else ''}", flush=True)
        for name in m.function_names:
            T = int(name[1:])
            fn = m.function(name)
            data = example_inputs(a.ctx, T, np.random.default_rng(0))
            ins = {}
            for n, x in zip(fn.input_names, data):
                b = fn.buffer("input", n)
                b.np[...] = x
                ins[n] = b
            outs = {n: fn.buffer("output", n) for n in fn.output_names}
            plan = B.Plan([fn.bind(ins, outs)])
            plan.run()
            if T not in refs:
                refs[T] = host_outputs(a.layers, a.ctx, T)
            errs = [rel(outs[n].np.astype(np.float32), r) for n, r in zip(fn.output_names, refs[T])]
            runs[(p.stem, name)] = (plan, m, ins, outs)
            res.setdefault(p.stem, {})[name] = {"err_o": max(errs[0::3])}
        res[p.stem]["placement"] = IC.inspect_package(p, m.function_names, Path.home() / "Library/Caches/coreai-cache",
                                                      build_id, sys.executable)["status"]
    acc = {k: [] for k in runs}
    for _ in range(a.rounds):
        for k, (plan, *_) in runs.items():
            acc[k].append(S.timed(plan, a.n, 2)["median_ms"])
    for (v, name), xs in acc.items():
        res[v][name].update(median_ms=float(np.median(xs)), min_ms=float(np.min(xs)))
    for v, r in res.items():
        cells = "  ".join(f"{n}: {x['median_ms']:7.3f} ms (o err {x['err_o']:.1e})" for n, x in r.items()
                          if n != "placement")
        print(f"{v:8s} [{r['placement']}] {cells}", flush=True)
    (a.out / "timing.json").write_text(json.dumps(res, indent=1))


def cmd_kernel(a):
    """One weight-free history-attention package per variant holding every context entry (t<T>_<ctx>k), built and
    then cold-compiled: what a single attention program shared by all 16 attention layers would cost to compile."""
    a.out.mkdir(parents=True, exist_ok=True)
    build_id = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    cache = Path.home() / "Library/Caches/coreai-cache" / build_id / Path(sys.executable).name
    rng = np.random.default_rng(0)
    for v in a.variants:
        dst = a.out / f"kernel_{v}.aimodel"
        if not dst.exists() or a.force:
            entries = []
            for ctx in a.ctxs:
                for T in a.rows:
                    mod = build_cores(v, 1, ctx, T)
                    entries.append((f"t{T}_{ctx // 1024}k", mod, mod.input_names(), mod.output_names()))
            Bld.save_program(entries, dst)
        shutil.rmtree(cache / (dst / "main.hash").read_bytes().hex(), ignore_errors=True)
        t0 = time.time()
        m = B.Model(dst, compute="ane")
        cold = time.time() - t0
        place = IC.inspect_package(dst, m.function_names, Path.home() / "Library/Caches/coreai-cache", build_id,
                                   sys.executable)["status"]
        cells = []
        for name in sorted(m.function_names, key=lambda n: (int(n.split("_")[0][1:]), int(n.split("_")[1][:-1]))):
            fn = m.function(name)
            ins = S.fill(fn, rng, 0.75)
            outs = {o: fn.buffer("output", o) for o in fn.output_names}
            plan = B.Plan([fn.bind(ins, outs)])
            plan.run()
            cells.append(f"{name} {S.timed(plan, a.n, 2)['median_ms']:.2f}")
        print(f"{v:8s} [{place}] cold compile {cold:.1f}s | " + "  ".join(cells), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("check", "build", "time", "kernel"))
    ap.add_argument("--variants", default="ref,opt")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--ctxs", default="8192,16384,32768,49152,65472", help="kernel: context entries in one package")
    ap.add_argument("--rows", default="8,64")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--cold", action="store_true", help="time: drop each probe package's cached specialization first")
    a = ap.parse_args()
    a.variants = [v for v in a.variants.split(",") if v]
    a.rows = [int(x) for x in a.rows.split(",")]
    a.ctxs = [int(x) for x in a.ctxs.split(",")]
    {"check": cmd_check, "build": cmd_build, "time": cmd_time, "kernel": cmd_kernel}[a.cmd](a)


if __name__ == "__main__":
    main()
