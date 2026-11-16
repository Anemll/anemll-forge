"""KV8 attention with a deferred in-graph append into ONE shared state, against the release form, for one real layer.

Both packages hold layer LAYER's real attention (input norm, q / k / v / o projections from the export, q / k norms,
RoPE, the release attention forms) as entries v8_<ctx>k (8 rows) and p64_<ctx>k (64 rows) per context:
    release  the KV8 history comes in as inputs sized kv_len(ctx) (keys (nkv, hd, L) with KV_KEYS_T=1, values
             (nkv, L, hd), FP16 scales (nkv, L)); the call returns its new rows in FP16 and the host quantizes and
             appends them (qwen38_kv_cache.quantize_values / append_rows), as the runtime does today.
    state    the history is four Core AI states sized for the largest context (SMAX) and shared by every entry: INT8
             codes in the release layouts and FP16 scales (nkv, SMAX, 32) (a trailing axis of 64 bytes: narrower scale
             states were silently not written on the M6 ANE, 12 Nov 2026). A call first writes the PREVIOUS call's
             rows, quantized by the host and given as inputs (kc, kcs, vc, vcs at wpos: no quantize in the graph, which
             ANEC rejects in front of a state write), then attends over its window [:kv_len(ctx)] exactly as the
             release graph does. Rejected verify rows are harmless: the mask hides them and the next write replaces
             them.
Checks per entry, on dense random KV8 history filled to near the top of the window: the state package's outputs vs the
release package's (y and the new rows), the state contents vs the host-appended release buffers, a ladder step (16K
entry then 96K entry on the same state, against a resized release buffer), and steady per-call times (graph; host
quantize / append separately). Placement: MPSGRAPH_PRINT_ANE_PLACEMENT_ANALYSIS=1 in the log. Every model load is
journaled when DJ_DIR names a deploy_journal.py directory.
    <coreai venv>/bin/python scripts/m6_kv_state_check.py build --out DIR [--ctx 16384,32768,98304]
    <coreai venv>/bin/python scripts/m6_kv_state_check.py run   --out DIR
    <coreai venv>/bin/python scripts/m6_kv_state_check.py place --out DIR    (load the state package once: placement only)
    MODES=release,seg ... build --out DIR; ... seg --out DIR   (segmented history: KV_SEG-row segment inputs, no resize)
Env (release defaults set here unless given): EXPORT_DIR, KV_KEYS_T, ATT_INT8MM, ATT_BLOCK, ... ; LAYER (default 3)."""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REL = {"KV_CACHE_DTYPE": "kv8", "KV_KEYS_T": "1", "ATT_INT8MM": "s8,s8b,sm8,pvf8", "ATT_S8_UNIT": "0.25",
       "ATT_S8B_UNIT": "0.25", "ATT_PF8_UNIT": "0.015625", "ATT_BLOCK": "2048", "ATT_BLOCK_PREFILL": "4096",
       "ATT_BLOCK_LONG": "4096", "ATT_LONG_FROM": "131072", "TPS": "64", "TPS_LONG": "32", "PREFILL_LONG_FROM": "98304",
       "ATT_BLOCK_PREFILL_LONG": "8192", "GDN_FAST": "1", "QCONV_INT8": "0",
       "EXPORT_DIR": os.path.expanduser("~/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat")}
for _k, _v in REL.items():
    os.environ.setdefault(_k, _v)
ROOT = Path(__file__).resolve().parents[1]
for p in ("scripts", "coreai", "coreai/swift_bridge"):
    sys.path.insert(0, str(ROOT / p))
import ane_compile_mode  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import qwen38_coreai_build as Bld  # noqa: E402
from qwen38_kv_cache import quantize_values  # noqa: E402

LAYER = int(os.environ.get("LAYER", "3"))
nkv, hd, rot, hid = Bld.nkv, Bld.hd, Bld.rot, Bld.hid
KT = Bld.KV_KEYS_T
SW = 32                                  # scale state trailing width (64 bytes)
SEG = int(os.environ.get("KV_SEG", "16384"))  # seg mode: rows per KV segment (tile widths must divide it)
STATES = ("k", "v", "ks", "vs")


def kshape(L):
    return (nkv, hd, L) if KT else (nkv, L, hd)


class Core(nn.Module):
    """input norm + the layer's attention mixer; mode release (history inputs) or state (shared states)."""

    def __init__(self, layer, ctx: int, T: int, mode: str, smax: int) -> None:
        super().__init__()
        self.mix, self.ln1, self.T, self.mode, self.smax = layer.mix, layer.ln1, T, mode, smax
        self.L = Bld.kv_len(ctx, T)
        if mode == "seg":
            assert self.L % SEG == 0, (self.L, SEG)
            self.n = self.L // SEG
        if mode == "state":
            self.register_buffer("k", torch.zeros(*kshape(smax), dtype=torch.int8))
            self.register_buffer("v", torch.zeros(nkv, smax, hd, dtype=torch.int8))
            self.register_buffer("ks", torch.zeros(nkv, smax, SW, dtype=torch.float16))
            self.register_buffer("vs", torch.zeros(nkv, smax, SW, dtype=torch.float16))

    def input_names(self):
        base = ["x", "cos", "sin", "mask"]
        if self.mode == "seg":
            return base + [f"{s}_{i}" for s in ("k", "v", "ks", "vs") for i in range(self.n)]
        return base + (["k", "v", "ks", "vs"] if self.mode == "release" else ["wpos", "kc", "kcs", "vc", "vcs"])

    def output_names(self):
        return ["y", "k_new", "v_new"]

    def forward(self, x, cos, sin, mask, *rest):
        T, L = self.T, self.L
        h = Bld.rms_hidden(x, self.ln1)
        if self.mode == "release":
            k, v, ks, vs = rest
        elif self.mode == "seg":
            n = self.n
            k = Bld.SegKV(rest[:n], 2 if KT else 1)
            v, ks, vs = Bld.SegKV(rest[n:2 * n], 1), Bld.SegKV(rest[2 * n:3 * n], 1), Bld.SegKV(rest[3 * n:], 1)
        else:
            wpos, kc, kcs, vc, vcs = rest
            w = wpos.item()
            torch._check(w >= 0)
            torch._check(w <= self.smax - T)
            if KT:
                self.k[:, :, w:w + T] = kc
            else:
                self.k[:, w:w + T] = kc
            self.v[:, w:w + T] = vc
            self.ks[:, w:w + T] = kcs
            self.vs[:, w:w + T] = vcs
            k = self.k[:, :, :L] if KT else self.k[:, :L]
            v, ks, vs = self.v[:, :L], self.ks[:, :L, 0], self.vs[:, :L, 0]
        y, kt, vt = self.mix(h, cos, sin, mask, k, v, L, T, vs, True, ks, True)
        return y, kt, vt

    def example(self):
        T, L, f = self.T, self.L, torch.float16
        ex = [torch.randn(1, hid, 1, T, dtype=f), torch.ones(T, rot, dtype=f), torch.zeros(T, rot, dtype=f), torch.zeros(1, L, dtype=f)]
        if self.mode == "release":
            ex += [torch.zeros(*kshape(L), dtype=torch.int8), torch.zeros(nkv, L, hd, dtype=torch.int8),
                   torch.ones(nkv, L, dtype=f) / 128, torch.ones(nkv, L, dtype=f) / 128]
        elif self.mode == "seg":
            ex += [torch.zeros(*kshape(SEG), dtype=torch.int8) for _ in range(self.n)]
            ex += [torch.zeros(nkv, SEG, hd, dtype=torch.int8) for _ in range(self.n)]
            ex += [torch.ones(nkv, SEG, dtype=f) / 128 for _ in range(2 * self.n)]
        else:
            ex += [torch.tensor([0], dtype=torch.int32), torch.zeros(*((nkv, hd, T) if KT else (nkv, T, hd)), dtype=torch.int8),
                   torch.ones(nkv, T, SW, dtype=f) / 128, torch.zeros(nkv, T, hd, dtype=torch.int8), torch.ones(nkv, T, SW, dtype=f) / 128]
        return tuple(ex)


def shapes(ctxs):
    return [(f"v8_{c // 1024}k", c, 8) for c in ctxs] + [(f"p64_{c // 1024}k", c, 64) for c in ctxs]


def build(a):
    import qwen38_ane_model as M
    Bld.rot1_basis_check(M.EXPORT_DIR, M.MODEL)
    ck = M.Checkpoint()
    Bld.KNOWN_LUTS.clear()
    W = Bld.layer_arrays(ck, LAYER)
    layer = Bld.LayerW(W, LAYER).eval().to(torch.float16)
    del W
    gc.collect()
    ctxs = [int(c) for c in a.ctx.split(",")]
    smax = max(Bld.kv_len(c, t) for _, c, t in shapes(ctxs))
    a.out.mkdir(parents=True, exist_ok=True)
    info = {"layer": LAYER, "ctxs": ctxs, "smax": smax, "env": {k: os.environ[k] for k in REL},
            "fuse": os.environ.get("ATT_FUSE", ""), "packages": {}}
    for mode in os.environ.get("MODES", "release,state").split(","):
        entries = []
        for name, c, t in shapes(ctxs):
            m = Core(layer, c, t, mode, smax)
            entries.append((name, m, m.input_names(), m.output_names(), STATES if mode == "state" else None))
        out = a.out / f"attn_L{LAYER:02d}_{mode}.aimodel"
        t0 = time.time()
        mb = Bld.save_program(entries, out)
        info["packages"][mode] = {"file": out.name, "mb": round(mb, 1), "build_s": round(time.time() - t0, 1),
                                  "main_hash": (out / "main.hash").read_bytes().hex(),
                                  "entries": [e[0] for e in entries], "kv_len": {e[0]: e[1].L for e in entries}}
        print(f"{mode}: {out.name} {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)
    (a.out / "build.json").write_text(json.dumps(info, indent=1) + "\n")


# ---- run -------------------------------------------------------------------------------------------------------------
def writable_as(nd, dtype):
    import ctypes
    mv = memoryview(nd._tensor)  # noqa: SLF001
    ptr = np.frombuffer(mv, dtype=np.uint8).__array_interface__["data"][0]
    return np.ctypeslib.as_array((ctypes.c_uint8 * mv.nbytes).from_address(ptr)).view(dtype).reshape(nd.shape)


def rope(p0, T):
    c = Bld.CFG
    inv = 1.0 / c["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
    ang = np.concatenate([np.outer(np.arange(p0, p0 + T), inv)] * 2, axis=1)
    return np.cos(ang).astype(np.float16), np.sin(ang).astype(np.float16)


def fill_history(rng, n):
    """dense KV8 rows for positions [0, n): N(0,1) keys and values quantized the host's way, in 8K-row pieces."""
    kc = np.zeros((nkv, n, hd), np.int8); vc = np.zeros((nkv, n, hd), np.int8)
    ks = np.zeros((nkv, n), np.float16); vs = np.zeros((nkv, n), np.float16)
    for a in range(0, n, 8192):
        b = min(n, a + 8192)
        kc[:, a:b], ks[:, a:b] = quantize_values(rng.standard_normal((nkv, b - a, hd)).astype(np.float16))
        vc[:, a:b], vs[:, a:b] = quantize_values((rng.standard_normal((nkv, b - a, hd)) * 0.5).astype(np.float16))
    return kc, ks, vc, vs


class Release:
    """host-owned release buffers for one entry (kv_len rows), appended like qwen38_kv_cache.append_rows."""

    def __init__(self, ND, IOS, L):
        self.L = L
        self.nd = {"k": ND(np.zeros(kshape(L), np.int8), IOS), "v": ND(np.zeros((nkv, L, hd), np.int8), IOS),
                   "ks": ND(np.zeros((nkv, L), np.float16), IOS), "vs": ND(np.zeros((nkv, L), np.float16), IOS)}
        self.w = {"k": writable_as(self.nd["k"], np.int8), "v": writable_as(self.nd["v"], np.int8),
                  "ks": writable_as(self.nd["ks"], np.float16), "vs": writable_as(self.nd["vs"], np.float16)}

    def load(self, kc, ks, vc, vs, n):
        n = min(n, self.L)
        self.w["k"][:] = 0; self.w["v"][:] = 0; self.w["ks"][:] = 0; self.w["vs"][:] = 0
        if KT:
            self.w["k"][:, :, :n] = kc[:, :n].transpose(0, 2, 1)
        else:
            self.w["k"][:, :n] = kc[:, :n]
        self.w["v"][:, :n] = vc[:, :n]; self.w["ks"][:, :n] = ks[:, :n]; self.w["vs"][:, :n] = vs[:, :n]

    def append(self, kt, vt, pos):
        T = kt.shape[1]
        c, s = quantize_values(kt)
        if KT:
            self.w["k"][:, :, pos:pos + T] = c.transpose(0, 2, 1)
        else:
            self.w["k"][:, pos:pos + T] = c
        self.w["ks"][:, pos:pos + T] = s
        c, s = quantize_values(vt)
        self.w["v"][:, pos:pos + T] = c
        self.w["vs"][:, pos:pos + T] = s


class State:
    def __init__(self, ND, IOS, smax):
        self.smax = smax
        self.nd = {"k": ND(np.zeros(kshape(smax), np.int8), IOS), "v": ND(np.zeros((nkv, smax, hd), np.int8), IOS),
                   "ks": ND(np.zeros((nkv, smax, SW), np.float16), IOS), "vs": ND(np.zeros((nkv, smax, SW), np.float16), IOS)}
        self.w = {"k": writable_as(self.nd["k"], np.int8), "v": writable_as(self.nd["v"], np.int8),
                  "ks": writable_as(self.nd["ks"], np.float16), "vs": writable_as(self.nd["vs"], np.float16)}

    def load(self, kc, ks, vc, vs, n):
        for x in self.w.values():
            x[:] = 0
        if KT:
            self.w["k"][:, :, :n] = kc[:, :n].transpose(0, 2, 1)
        else:
            self.w["k"][:, :n] = kc[:, :n]
        self.w["v"][:, :n] = vc[:, :n]
        self.w["ks"][:, :n] = ks[:, :n, None]; self.w["vs"][:, :n] = vs[:, :n, None]

    def contents(self, a, b):
        k = self.w["k"][:, :, a:b].transpose(0, 2, 1) if KT else self.w["k"][:, a:b]
        return k.copy(), self.w["ks"][:, a:b, 0].copy(), self.w["v"][:, a:b].copy(), self.w["vs"][:, a:b, 0].copy()


def pending_inputs(ND, IOS, kt, vt, T):
    """the previous call's rows as the state entry's write inputs (host quantize; zeros before the first call)."""
    if kt is None:
        kc = np.zeros((nkv, T, hd), np.int8); vc = kc.copy(); ks = np.zeros((nkv, T), np.float16); vs = ks.copy()
    else:
        (kc, ks), (vc, vs) = quantize_values(kt), quantize_values(vt)
    kin = kc.transpose(0, 2, 1).copy() if KT else kc
    rep = lambda s: np.repeat(s[:, :, None], SW, 2).astype(np.float16)  # noqa: E731
    return {"kc": ND(kin, IOS), "kcs": ND(rep(ks), IOS), "vc": ND(vc, IOS), "vcs": ND(rep(vs), IOS)}


async def run(a):
    from coreai.runtime import AIModel, NDArray as ND
    from coreai.runtime._ndarray import StorageKind
    from coreai_bench_helpers import specialization_for
    IOS = StorageKind.IO_SURFACE
    info = json.loads((a.out / "build.json").read_text())
    smax, ctxs = info["smax"], info["ctxs"]
    DJ = None
    if os.environ.get("DJ_DIR"):
        sys.path.insert(0, os.environ["DJ_DIR"])
        import deploy_journal as DJ  # noqa: N812
        J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                        "out": str(a.out), "mode": "m6_kv_state_check"}, f"kvstate-{os.getpid()}-{int(time.time())}")
        dev = DJ.m6_device()
    models, fns = {}, {}
    for mode in ("release", "state"):
        p = a.out / info["packages"][mode]["file"]
        pkg = {"path": str(p), "main_hash": info["packages"][mode]["main_hash"], "label": f"attn L{LAYER} {mode}"}
        aid = J.attempt(dev, pkg, functions=info["packages"][mode]["entries"], request={"compute": "ane"}) if DJ else None
        t0 = time.time()
        try:
            models[mode] = await AIModel.load(p, specialization_options=specialization_for("ane"))
        except Exception as e:  # noqa: BLE001
            if DJ:
                J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}")
            raise
        if DJ:
            J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
        fns[mode] = {n: models[mode].load_function(n) for n in info["packages"][mode]["entries"]}
        print(f"{mode}: load {time.time() - t0:.1f}s", flush=True)
    rng = np.random.default_rng(20261112)
    hist = fill_history(rng, smax)
    st = State(ND, IOS, smax)
    res = {"entries": {}, "ladder": None}

    async def call_release(rel, name, x, p0, T):
        L = rel.L
        cos, sin = rope(p0, T)
        mask = np.where(np.arange(L)[None] < p0, 0, -1e4).astype(np.float16)
        t = time.perf_counter()
        out = await fns["release"][name](inputs={"x": ND(x, IOS), "cos": ND(cos, IOS), "sin": ND(sin, IOS), "mask": ND(mask, IOS), **rel.nd})
        tg = (time.perf_counter() - t) * 1e3
        y, kt, vt = (out[n].numpy() for n in ("y", "k_new", "v_new"))
        t = time.perf_counter(); rel.append(kt, vt, p0); ta = (time.perf_counter() - t) * 1e3
        return y, kt, vt, tg, ta

    async def call_state(name, x, p0, T, L, prev):
        cos, sin = rope(p0, T)
        mask = np.where(np.arange(L)[None] < p0, 0, -1e4).astype(np.float16)
        t = time.perf_counter()
        pin = pending_inputs(ND, IOS, prev[0], prev[1], T) if prev else pending_inputs(ND, IOS, None, None, T)
        wpos = prev[2] if prev else p0
        th = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        out = await fns["state"][name](inputs={"x": ND(x, IOS), "cos": ND(cos, IOS), "sin": ND(sin, IOS), "mask": ND(mask, IOS),
                                               "wpos": ND(np.array([wpos], np.int32)), **pin}, state=st.nd)
        tg = (time.perf_counter() - t) * 1e3
        return out["y"].numpy(), out["k_new"].numpy(), out["v_new"].numpy(), tg, th

    for name, c, T in [(n, c, t) for n, c, t in shapes(ctxs)]:
        L = info["packages"]["release"]["kv_len"][name]
        p0 = L - 6 * T
        rel = Release(ND, IOS, L)
        rel.load(*hist, p0); st.load(*hist, p0)
        r = {"L": L, "p0": p0, "calls": []}
        prev = None
        for i in range(4):
            pos = p0 + i * T
            x = rng.standard_normal((1, hid, 1, T)).astype(np.float16)
            yr, ktr, vtr, _, _ = await call_release(rel, name, x, pos, T)
            ys, kts, vts, _, _ = await call_state(name, x, pos, T, L, prev)
            prev = (kts, vts, pos)
            d = float(np.abs(yr.astype(np.float32) - ys.astype(np.float32)).max())
            r["calls"].append({"pos": pos, "y_equal": bool(np.array_equal(yr, ys)), "y_maxdiff": d, "y_absmax": float(np.abs(yr).max()),
                               "k_new_equal": bool(np.array_equal(ktr, kts)), "v_new_equal": bool(np.array_equal(vtr, vts))})
        # state rows written so far (calls 0..2, deferred) vs the release buffers
        a0, b0 = p0, p0 + 3 * T
        sk, sks, sv, svs = st.contents(a0, b0)
        rk = rel.w["k"][:, :, a0:b0].transpose(0, 2, 1) if KT else rel.w["k"][:, a0:b0]
        r["state_rows_equal"] = {"k": bool(np.array_equal(sk, rk)), "ks": bool(np.array_equal(sks, rel.w["ks"][:, a0:b0])),
                                 "v": bool(np.array_equal(sv, rel.w["v"][:, a0:b0])), "vs": bool(np.array_equal(svs, rel.w["vs"][:, a0:b0]))}
        # steady timing at p0 (same inputs each call; the state call rewrites the same rows)
        x = rng.standard_normal((1, hid, 1, T)).astype(np.float16)
        tr, ta, ts, th = [], [], [], []
        for i in range(22):
            _, _, _, g, h_ = await call_release(rel, name, x, p0, T)
            tr.append(g); ta.append(h_)
            _, _, _, g, h_ = await call_state(name, x, p0, T, L, prev)
            ts.append(g); th.append(h_)
        med = lambda v: float(np.median(v[2:]))  # noqa: E731
        r["ms"] = {"release_graph": med(tr), "release_host_append": med(ta), "state_graph": med(ts), "state_host_prep": med(th)}
        res["entries"][name] = r
        c0 = r["calls"]
        print(f"{name} (L {L}): y equal {[x['y_equal'] for x in c0]} maxdiff {max(x['y_maxdiff'] for x in c0):.3g} "
              f"(|y| {c0[0]['y_absmax']:.3g}); new rows equal {all(x['k_new_equal'] and x['v_new_equal'] for x in c0)}; "
              f"state rows == release {r['state_rows_equal']}; ms release {r['ms']['release_graph']:.2f} + host {r['ms']['release_host_append']:.2f}"
              f" | state {r['ms']['state_graph']:.2f} + host {r['ms']['state_host_prep']:.2f}", flush=True)
        (a.out / "result.json").write_text(json.dumps(res, indent=1) + "\n")

    # ladder: 16K entry then the largest entry on the SAME state; release resizes (copies rows) into a new buffer
    small, big = f"p64_{ctxs[0] // 1024}k", f"p64_{ctxs[-1] // 1024}k"
    Ls, Lb = info["packages"]["release"]["kv_len"][small], info["packages"]["release"]["kv_len"][big]
    p0 = Ls - 2 * 64
    rel = Release(ND, IOS, Ls); rel.load(*hist, p0); st.load(*hist, p0)
    prev, lad = None, []
    for i, (name, L) in enumerate(((small, Ls), (small, Ls), (big, Lb), (big, Lb))):
        pos = p0 + i * 64
        if name == big and rel.L != Lb:
            relb = Release(ND, IOS, Lb)
            for n in ("k", "v", "ks", "vs"):
                if n == "k" and KT:
                    relb.w[n][:, :, :Ls] = rel.w[n]
                else:
                    relb.w[n][:, :Ls] = rel.w[n]
            rel = relb
        x = rng.standard_normal((1, hid, 1, 64)).astype(np.float16)
        yr, _, _, _, _ = await call_release(rel, name, x, pos, 64)
        ys, kts, vts, _, _ = await call_state(name, x, pos, 64, L, prev)
        prev = (kts, vts, pos)
        lad.append({"entry": name, "pos": pos, "y_equal": bool(np.array_equal(yr, ys)),
                    "y_maxdiff": float(np.abs(yr.astype(np.float32) - ys.astype(np.float32)).max())})
    res["ladder"] = lad
    print("ladder:", [(x["entry"], x["pos"], x["y_equal"], round(x["y_maxdiff"], 6)) for x in lad], flush=True)
    (a.out / "result.json").write_text(json.dumps(res, indent=1) + "\n")


class Segs:
    """the segmented history: NSEG fixed-shape buffers per tensor, shared by every entry (an entry binds the first n)."""

    def __init__(self, ND, IOS, nseg):
        self.nd, self.w = {}, {}
        for i in range(nseg):
            for n, shp, dt in (("k", kshape(SEG), np.int8), ("v", (nkv, SEG, hd), np.int8), ("ks", (nkv, SEG), np.float16), ("vs", (nkv, SEG), np.float16)):
                self.nd[f"{n}_{i}"] = ND(np.zeros(shp, dt), IOS)
                self.w[f"{n}_{i}"] = writable_as(self.nd[f"{n}_{i}"], dt)
        self.nseg = nseg

    def load(self, kc, ks, vc, vs, n):
        for x in self.w.values():
            x[:] = 0
        for i in range(self.nseg):
            a, b = i * SEG, min(n, (i + 1) * SEG)
            if b <= a:
                break
            if KT:
                self.w[f"k_{i}"][:, :, :b - a] = kc[:, a:b].transpose(0, 2, 1)
            else:
                self.w[f"k_{i}"][:, :b - a] = kc[:, a:b]
            self.w[f"v_{i}"][:, :b - a] = vc[:, a:b]; self.w[f"ks_{i}"][:, :b - a] = ks[:, a:b]; self.w[f"vs_{i}"][:, :b - a] = vs[:, a:b]

    def append(self, kt, vt, pos):
        T = kt.shape[1]
        i, r = divmod(pos, SEG)
        assert r + T <= SEG
        c, s = quantize_values(kt)
        if KT:
            self.w[f"k_{i}"][:, :, r:r + T] = c.transpose(0, 2, 1)
        else:
            self.w[f"k_{i}"][:, r:r + T] = c
        self.w[f"ks_{i}"][:, r:r + T] = s
        c, s = quantize_values(vt)
        self.w[f"v_{i}"][:, r:r + T] = c; self.w[f"vs_{i}"][:, r:r + T] = s

    def bind(self, n):
        return {f"{t}_{i}": self.nd[f"{t}_{i}"] for t in ("k", "v", "ks", "vs") for i in range(n)}

    def rows(self, a, b):
        i, r = divmod(a, SEG)
        k = self.w[f"k_{i}"][:, :, r:r + b - a].transpose(0, 2, 1) if KT else self.w[f"k_{i}"][:, r:r + b - a]
        return k, self.w[f"ks_{i}"][:, r:r + b - a], self.w[f"v_{i}"][:, r:r + b - a], self.w[f"vs_{i}"][:, r:r + b - a]


async def run_seg(a):
    from coreai.runtime import AIModel, NDArray as ND
    from coreai.runtime._ndarray import StorageKind
    from coreai_bench_helpers import specialization_for
    IOS = StorageKind.IO_SURFACE
    info = json.loads((a.out / "build.json").read_text())
    smax, ctxs = info["smax"], info["ctxs"]
    sys.path.insert(0, os.environ["DJ_DIR"])
    import deploy_journal as DJ  # noqa: N812
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "m6_kv_state_check/seg"}, f"kvseg-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    fns = {}
    for mode in ("release", "seg"):
        p = a.out / info["packages"][mode]["file"]
        pkg = {"path": str(p), "main_hash": info["packages"][mode]["main_hash"], "label": f"attn L{LAYER} {mode}"}
        aid = J.attempt(dev, pkg, functions=info["packages"][mode]["entries"], request={"compute": "ane"})
        t0 = time.time()
        try:
            mdl = await AIModel.load(p, specialization_options=specialization_for("ane"))
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
        J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
        fns[mode] = {n: mdl.load_function(n) for n in info["packages"][mode]["entries"]}
        print(f"{mode}: load {time.time() - t0:.1f}s", flush=True)
    rng = np.random.default_rng(20261112)
    hist = fill_history(rng, smax)
    sg = Segs(ND, IOS, smax // SEG)
    res = {"seg_rows": SEG, "entries": {}, "growth": None}

    async def call(mode, name, x, p0, T, L, kv):
        cos, sin = rope(p0, T)
        mask = np.where(np.arange(L)[None] < p0, 0, -1e4).astype(np.float16)
        t = time.perf_counter()
        out = await fns[mode][name](inputs={"x": ND(x, IOS), "cos": ND(cos, IOS), "sin": ND(sin, IOS), "mask": ND(mask, IOS), **kv})
        return [out[n].numpy() for n in ("y", "k_new", "v_new")], (time.perf_counter() - t) * 1e3

    for name, c, T in shapes(ctxs):
        L = info["packages"]["release"]["kv_len"][name]
        n = L // SEG
        p0 = L - 6 * T
        rel = Release(ND, IOS, L); rel.load(*hist, p0); sg.load(*hist, p0)
        r = {"L": L, "segments": n, "p0": p0, "calls": []}
        for i in range(4):
            pos = p0 + i * T
            x = rng.standard_normal((1, hid, 1, T)).astype(np.float16)
            (yr, ktr, vtr), _ = await call("release", name, x, pos, T, L, rel.nd)
            rel.append(ktr, vtr, pos)
            (ys, kts, vts), _ = await call("seg", name, x, pos, T, L, sg.bind(n))
            sg.append(kts, vts, pos)
            r["calls"].append({"pos": pos, "y_equal": bool(np.array_equal(yr, ys)), "new_equal": bool(np.array_equal(ktr, kts) and np.array_equal(vtr, vts)),
                               "y_maxdiff": float(np.abs(yr.astype(np.float32) - ys.astype(np.float32)).max())})
        sk, sks, sv, svs = sg.rows(p0, p0 + 4 * T)
        rk = rel.w["k"][:, :, p0:p0 + 4 * T].transpose(0, 2, 1) if KT else rel.w["k"][:, p0:p0 + 4 * T]
        r["appended_rows_equal"] = bool(np.array_equal(sk, rk) and np.array_equal(sks, rel.w["ks"][:, p0:p0 + 4 * T])
                                        and np.array_equal(sv, rel.w["v"][:, p0:p0 + 4 * T]) and np.array_equal(svs, rel.w["vs"][:, p0:p0 + 4 * T]))
        x = rng.standard_normal((1, hid, 1, T)).astype(np.float16)
        tr, ts = [], []
        for i in range(22):
            tr.append((await call("release", name, x, p0, T, L, rel.nd))[1])
            ts.append((await call("seg", name, x, p0, T, L, sg.bind(n)))[1])
        r["ms"] = {"release": float(np.median(tr[2:])), "seg": float(np.median(ts[2:]))}
        res["entries"][name] = r
        print(f"{name} (L {L}, {n} segments): y equal {[x_['y_equal'] for x_ in r['calls']]} maxdiff {max(x_['y_maxdiff'] for x_ in r['calls']):.3g}; "
              f"new rows equal {all(x_['new_equal'] for x_ in r['calls'])}; appended rows equal {r['appended_rows_equal']}; "
              f"ms release {r['ms']['release']:.2f} | seg {r['ms']['seg']:.2f}", flush=True)
        (a.out / "result_seg.json").write_text(json.dumps(res, indent=1) + "\n")

    # growth: 16K entry, then the largest entry over the SAME segment buffers (no copy); release copies into a new buffer
    small, big = f"p64_{ctxs[0] // 1024}k", f"p64_{ctxs[-1] // 1024}k"
    Ls, Lb = info["packages"]["release"]["kv_len"][small], info["packages"]["release"]["kv_len"][big]
    p0 = Ls - 2 * 64
    rel = Release(ND, IOS, Ls); rel.load(*hist, p0); sg.load(*hist, p0)
    grow = []
    for i, (name, L) in enumerate(((small, Ls), (small, Ls), (big, Lb), (big, Lb))):
        pos = p0 + i * 64
        if name == big and rel.L != Lb:
            relb = Release(ND, IOS, Lb)
            t = time.perf_counter()
            for n_ in ("k", "v", "ks", "vs"):
                if n_ == "k" and KT:
                    relb.w[n_][:, :, :Ls] = rel.w[n_]
                else:
                    relb.w[n_][:, :Ls] = rel.w[n_]
            grow.append({"release_resize_copy_ms": (time.perf_counter() - t) * 1e3})
            rel = relb
        x = rng.standard_normal((1, hid, 1, 64)).astype(np.float16)
        (yr, ktr, vtr), _ = await call("release", name, x, pos, 64, L, rel.nd)
        rel.append(ktr, vtr, pos)
        (ys, kts, vts), _ = await call("seg", name, x, pos, 64, L, sg.bind(L // SEG))
        sg.append(kts, vts, pos)
        grow.append({"entry": name, "pos": pos, "y_equal": bool(np.array_equal(yr, ys)),
                     "y_maxdiff": float(np.abs(yr.astype(np.float32) - ys.astype(np.float32)).max())})
    res["growth"] = grow
    print("growth:", grow, flush=True)
    (a.out / "result_seg.json").write_text(json.dumps(res, indent=1) + "\n")


async def run_ab(a):
    """two release-form builds (a.out = A, a.other = B, e.g. ATT_FUSE): the same history and inputs, outputs bitwise
    and interleaved per-call times."""
    from coreai.runtime import AIModel, NDArray as ND
    from coreai.runtime._ndarray import StorageKind
    from coreai_bench_helpers import specialization_for
    IOS = StorageKind.IO_SURFACE
    infos = {k: json.loads((d / "build.json").read_text()) for k, d in (("A", a.out), ("B", a.other))}
    sys.path.insert(0, os.environ["DJ_DIR"])
    import deploy_journal as DJ  # noqa: N812
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.other), "mode": "m6_kv_state_check/ab"}, f"kvab-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    fns = {}
    for k, d in (("A", a.out), ("B", a.other)):
        p = d / infos[k]["packages"]["release"]["file"]
        pkg = {"path": str(p), "main_hash": infos[k]["packages"]["release"]["main_hash"], "label": f"attn L{LAYER} release {k} ({infos[k].get('fuse')})"}
        aid = J.attempt(dev, pkg, functions=infos[k]["packages"]["release"]["entries"], request={"compute": "ane"})
        t0 = time.time()
        try:
            mdl = await AIModel.load(p, specialization_options=specialization_for("ane"))
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
        J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
        fns[k] = {n: mdl.load_function(n) for n in infos[k]["packages"]["release"]["entries"]}
        print(f"{k} ({infos[k].get('fuse') or 'none'}): load {time.time() - t0:.1f}s", flush=True)
    rng = np.random.default_rng(20261112)
    smax = infos["A"]["smax"]
    hist = fill_history(rng, smax)
    res = {"A": str(a.out), "B": str(a.other), "fuse": {k: infos[k].get("fuse") for k in infos}, "entries": {}}
    for name, c, T in shapes(infos["A"]["ctxs"]):
        L = infos["A"]["packages"]["release"]["kv_len"][name]
        p0 = L - 6 * T
        bufs = {k: Release(ND, IOS, L) for k in ("A", "B")}
        for b in bufs.values():
            b.load(*hist, p0)
        x = rng.standard_normal((1, hid, 1, T)).astype(np.float16)
        cos, sin = rope(p0, T)
        mask = np.where(np.arange(L)[None] < p0, 0, -1e4).astype(np.float16)
        feed = {"x": ND(x, IOS), "cos": ND(cos, IOS), "sin": ND(sin, IOS), "mask": ND(mask, IOS)}
        outs, ts = {}, {"A": [], "B": []}
        for k in ("A", "B"):
            o = await fns[k][name](inputs={**feed, **bufs[k].nd})
            outs[k] = [o[n].numpy() for n in ("y", "k_new", "v_new")]
        for i in range(24):
            for k in (("A", "B") if i % 2 == 0 else ("B", "A")):
                t = time.perf_counter()
                await fns[k][name](inputs={**feed, **bufs[k].nd})
                ts[k].append((time.perf_counter() - t) * 1e3)
        eq = [bool(np.array_equal(u, v)) for u, v in zip(outs["A"], outs["B"])]
        md = [float(np.abs(u.astype(np.float32) - v.astype(np.float32)).max()) for u, v in zip(outs["A"], outs["B"])]
        r = {"equal_y_k_v": eq, "maxdiff_y_k_v": md, "y_absmax": float(np.abs(outs["A"][0]).max()),
             "ms": {k: float(np.median(v[4:])) for k, v in ts.items()}, "ms_p10": {k: float(np.percentile(v[4:], 10)) for k, v in ts.items()}}
        res["entries"][name] = r
        print(f"{name}: equal y/k/v {eq} maxdiff {[round(m, 5) for m in md]} (|y| {r['y_absmax']:.3g}); "
              f"ms A {r['ms']['A']:.2f} B {r['ms']['B']:.2f} (p10 {r['ms_p10']['A']:.2f} / {r['ms_p10']['B']:.2f})", flush=True)
    (a.other / "result_ab.json").write_text(json.dumps(res, indent=1) + "\n")


async def place(a):
    from coreai.runtime import AIModel
    from coreai_bench_helpers import specialization_for
    info = json.loads((a.out / "build.json").read_text())
    p = a.out / info["packages"]["state"]["file"]
    sys.path.insert(0, os.environ["DJ_DIR"])
    import deploy_journal as DJ  # noqa: N812
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "m6_kv_state_check/place"}, f"kvstate-place-{os.getpid()}-{int(time.time())}")
    dev, pkg = DJ.m6_device(), {"path": str(p), "main_hash": info["packages"]["state"]["main_hash"], "label": f"attn L{LAYER} state (placement)"}
    aid = J.attempt(dev, pkg, functions=info["packages"]["state"]["entries"], request={"compute": "ane"})
    t0 = time.time()
    try:
        await AIModel.load(p, specialization_options=specialization_for("ane"))
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    print(f"state: load {time.time() - t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("build", "run", "place", "seg", "ab"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ctx", default="16384,32768,98304")
    ap.add_argument("--other", type=Path, help="ab: the second build directory (B)")
    a = ap.parse_args()
    if a.what == "build":
        build(a)
    elif a.what == "place":
        asyncio.run(place(a))
    elif a.what == "seg":
        asyncio.run(run_seg(a))
    elif a.what == "ab":
        asyncio.run(run_ab(a))
    else:
        asyncio.run(run(a))


if __name__ == "__main__":
    main()
