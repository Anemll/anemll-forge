"""v5: KV8 in-graph append at Qwen3.8 FA shapes (nkv 4, head dim 256, 24 query heads = 6 per KV head).
States per layer: k, v INT8 codes (NKV, SMAX, HD) and ks, vs FP16 scales (NKV, SMAX, 1) (token axis in the middle:
SCALE_LAYOUT=2d is the (NKV, SMAX) form, which failed ANEC "Parent tensor wrap axis must be found"), shared by every entry s<ctx>k.
Each call: new K / V rows (FP16, from a conv) are quantized IN THE GRAPH the host's way (qwen38_kv_cache.quantize_values:
scale = max|x| / 127 per token and KV head, codes = round(x / scale)) as FP16 arithmetic followed by a constant-unit (1)
quantize (a runtime per-row quantize scale is rejected by ANEC, M6 research 3 Oct), and slice-assigned at runtime
position pos into the four states. Attention = the Forge release kv8 default: history in TILE-row tiles, keys and
values dequantized per tile (codes / 128) with key scales on the scores and value scales on the exp weights, two-pass
softmax over [history | the call's own block] with the block term from the FP16 rows (causal) and the history mask
hiding [pos, pos + T). Checks: codes / scales vs quantize_values of the graph's own FP16 rows, rows persist across
entries, o vs a numpy reference built from the stored states; placement (MPSGRAPH_PRINT_ANE_PLACEMENT_ANALYSIS);
ms per call. Every AIModel.load journaled (deploy_journal v7).
    T_ROWS=8|64 TILE=4096 SMAX=98304 CTXS=16384,32768,98304 python kv_slice_probe_v5.py"""
import os, sys
from pathlib import Path
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
sys.path.insert(0, str(V02 / "scripts")); sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")
import ane_compile_mode, deploy_journal as DJ, hashlib
ane_compile_mode.apply(log=lambda m: None)

import asyncio
import shutil
import time

import numpy as np
import torch
import torch.nn as nn

import coreai_torch
import coreai_torch._compression.custom_layers  # noqa: F401  (registers coreai.quantize / dequantize)
from coreai.runtime import AIModel, NDArray
from coreai.runtime._ndarray import StorageKind
from coreai_opt.casting import cast_to_16_bit_precision

from coreai_bench_helpers import specialization_for
from qwen38_kv_cache import quantize_values

C, NKV, HD, GRP = 1024, 4, 256, 6
T = int(os.environ.get("T_ROWS", "8"))
TILE = int(os.environ.get("TILE", "4096"))
SMAX = int(os.environ.get("SMAX", "98304"))
CTXS = tuple(int(c) for c in os.environ.get("CTXS", "16384,32768,98304").split(","))
SCALE_3D = os.environ.get("SCALE_LAYOUT", "3d") == "3d"
SW = int(os.environ.get("SCALE_W", "1"))                 # 3d scale state width (NKV, SMAX, SW): the value repeated SW times
KV8 = os.environ.get("KVFMT", "kv8") == "kv8"          # bisect: fp16 = FP16 k / v states, no scales, no quantize
QMODE = os.environ.get("QMODE", "quantize")             # kv8 codes: quantize (native op, unit 1) | cast (round + int8 cast)
# | qdq (the codes also feed a dequantize: the block term reads the dequantized rows) | qdq2 (quantize -> dequantize -> quantize)
KVWRITE = os.environ.get("KVWRITE", "graph")             # graph: slice-assigned in the graph | input: k, v, ks, vs are plain INPUTS of the full SMAX size
# (one buffer bound to every entry, each slices its window [:ctx] in the graph); the host writes quantized rows between calls
# | in8: the graph slice-assigns ALREADY-QUANTIZED rows given as inputs (kc, vc INT8 (NKV, T, HD); kcs, vcs FP16 scales)
# into the INT8 / scale states: no quantize in the graph (isolates the INT8 state write)
DQMODE = os.environ.get("DQMODE", "dequantize")         # kv8 history: dequantize (native op) | cast (int8 -> fp16 cast * 1/128)
ROOT = Path(__file__).resolve().parent / (f"artifacts_kv_slice_v5_T{T}_tile{TILE}_S{SMAX}" + ("" if SCALE_3D else "_s2d") + ("" if SW == 1 else f"_sw{SW}") + ("" if KV8 else "_fp16")
             + ("" if QMODE == "quantize" else f"_q{QMODE}") + ("" if DQMODE == "dequantize" else f"_dq{DQMODE}") + ("" if KVWRITE == "graph" else f"_w{KVWRITE}"))
SSH = (NKV, SMAX, SW) if SCALE_3D else (NKV, SMAX)
CACHE = Path.home() / "Library/Caches/coreai-cache"
STATES = ("k", "v", "ks", "vs") if os.environ.get("KVFMT", "kv8") == "kv8" else ("k", "v")


def q8(x, unit):
    return torch.ops.coreai.quantize(x, unit, torch.int8, zero_point=torch.tensor(0, dtype=torch.int8), minval=None, axis=0)


def dq8(c, unit):
    return torch.ops.coreai.dequantize(c, unit, zero_point=torch.tensor(0, dtype=torch.int8), minval=None, axis=0,
                                       input_dtype=None, output_dtype=torch.float16)


class KV8Attn(nn.Module):
    def __init__(self, ctx: int) -> None:
        super().__init__()
        self.ctx = ctx
        torch.manual_seed(0)
        self.conv = nn.Conv2d(C, (GRP + 2) * NKV * HD, 1, bias=False)     # q (24 x 256), k, v (4 x 256 each)
        kdt = torch.int8 if KV8 else torch.float16
        self.register_buffer("k", torch.zeros(NKV, SMAX, HD, dtype=kdt))
        self.register_buffer("v", torch.zeros(NKV, SMAX, HD, dtype=kdt))
        if KV8:
            self.register_buffer("ks", torch.zeros(*SSH))
            self.register_buffer("vs", torch.zeros(*SSH))
        self.register_buffer("one", torch.tensor(1.0, dtype=torch.float16))
        self.register_buffer("u128", torch.tensor(1 / 128, dtype=torch.float16))

    def quant_rows(self, x):                              # x (NKV, T, HD) fp16 -> codes int8, scales (NKV, T) fp16
        s = torch.clamp_min(x.abs().amax(-1), 1.27e-4) * (1 / 127)
        z = x * (1 / s).unsqueeze(-1)
        if QMODE == "cast":
            return torch.round(z).to(torch.int8), s
        c = q8(z, self.one)
        if QMODE == "qdq2":
            c = q8(dq8(c, self.one), self.one)
        return c, s

    def deq(self, c):
        return dq8(c, self.u128) if DQMODE == "dequantize" else c.to(torch.float16) * (1 / 128)

    def forward(self, x, pos, mask, *kvin):                # x (1, C, 1, T); pos (1,) int32; mask (1, ctx) additive; kvin (input mode)
        if KVWRITE == "in8":
            kc_in, kcs_in, vc_in, vcs_in = kvin
            kvin = ()
        if kvin:
            K, V, KS, VS = kvin
        else:
            K, V, KS, VS = self.k, self.v, getattr(self, "ks", None), getattr(self, "vs", None)
        y = self.conv(x)[0, :, 0, :]                      # ((GRP+2)*NKV*HD, T)
        nq = GRP * NKV * HD
        q = y[:nq].reshape(NKV, GRP, HD, T).permute(0, 1, 3, 2).reshape(NKV, GRP * T, HD)   # rows (g, t)
        kt = y[nq:nq + NKV * HD].reshape(NKV, HD, T).transpose(1, 2)                         # (NKV, T, HD)
        vt = y[nq + NKV * HD:].reshape(NKV, HD, T).transpose(1, 2)
        p = pos.item()
        torch._check(p >= 0)
        torch._check(p <= SMAX - T)
        if KVWRITE == "input":
            pass
        elif KVWRITE == "in8":
            self.k[:, p:p + T] = kc_in
            self.v[:, p:p + T] = vc_in
            self.ks[:, p:p + T] = kcs_in.expand(NKV, T, SW) if SCALE_3D else kcs_in.reshape(NKV, T)
            self.vs[:, p:p + T] = vcs_in.expand(NKV, T, SW) if SCALE_3D else vcs_in.reshape(NKV, T)
        elif KV8:
            kc, ksc = self.quant_rows(kt)
            vc, vsc = self.quant_rows(vt)
            self.k[:, p:p + T] = kc
            self.v[:, p:p + T] = vc
            self.ks[:, p:p + T] = ksc.unsqueeze(-1).expand(NKV, T, SW) if SCALE_3D else ksc
            self.vs[:, p:p + T] = vsc.unsqueeze(-1).expand(NKV, T, SW) if SCALE_3D else vsc
            if QMODE == "qdq":                             # q/dq pair: the block term reads the dequantized rows
                kt = dq8(kc, self.one) * ksc.unsqueeze(-1)
                vt = dq8(vc, self.one) * vsc.unsqueeze(-1)
        else:
            self.k[:, p:p + T] = kt
            self.v[:, p:p + T] = vt
        causal = torch.triu(torch.full((T, T), -1e4, dtype=torch.float16), 1)
        sc_b = (q @ kt.transpose(1, 2)) * HD ** -0.5 + causal.repeat(GRP, 1)                 # (NKV, GRP*T, T)
        spans = [(a, min(a + TILE, self.ctx)) for a in range(0, self.ctx, TILE)]
        scs = []
        for a, b in spans:
            if KV8:
                kd = self.deq(K[:, a:b]).transpose(1, 2)                                     # (NKV, HD, tile)
                s_ = (q @ kd) * HD ** -0.5 * ((KS[:, a:b, :1] if KS.dim() == 3 else KS[:, a:b]).reshape(NKV, 1, b - a) * 128) + mask[:, a:b].reshape(1, 1, b - a)
            else:
                s_ = (q @ K[:, a:b].transpose(1, 2)) * HD ** -0.5 + mask[:, a:b].reshape(1, 1, b - a)
            scs.append(s_)
        m = sc_b.amax(-1, keepdim=True)
        for s_ in scs:
            m = torch.maximum(m, s_.amax(-1, keepdim=True))
        e_b = torch.exp(sc_b - m)
        den, num = e_b.sum(-1, keepdim=True), e_b @ vt
        for s_, (a, b) in zip(scs, spans):
            e_ = torch.exp(s_ - m)
            den = den + e_.sum(-1, keepdim=True)
            if KV8:
                e_ = e_ * ((VS[:, a:b, :1] if VS.dim() == 3 else VS[:, a:b]).reshape(NKV, 1, b - a) * 128)
                num = num + e_ @ self.deq(V[:, a:b])
            else:
                num = num + e_ @ V[:, a:b]
        return y, num / den                                # o (NKV, GRP*T, HD)


def build() -> Path:
    out = ROOT / "kv8_slice.aimodel"
    if out.exists():
        return out
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    for ctx in CTXS:
        m = KV8Attn(ctx).eval().to(torch.float16)
        if KV8:
            for n in ("k", "v"):
                setattr(m, n, getattr(m, n).to(torch.int8))
        ex = (torch.randn(1, C, 1, T, dtype=torch.float16), torch.tensor([16], dtype=torch.int32),
              torch.zeros(1, ctx, dtype=torch.float16))
        if KVWRITE == "in8":
            ex += (torch.zeros(NKV, T, HD, dtype=torch.int8), torch.ones(NKV, T, 1, dtype=torch.float16),
                   torch.zeros(NKV, T, HD, dtype=torch.int8), torch.ones(NKV, T, 1, dtype=torch.float16))
        if KVWRITE == "input":
            kdt = torch.int8 if KV8 else torch.float16
            ex += (torch.zeros(NKV, SMAX, HD, dtype=kdt), torch.zeros(NKV, SMAX, HD, dtype=kdt),
                   torch.ones(*SSH, dtype=torch.float16), torch.ones(*SSH, dtype=torch.float16))
        ep = torch.export.export(m, ex, strict=False).run_decompositions(coreai_torch.get_decomp_table())
        cast_to_16_bit_precision(ep)
        conv.add_exported_program(ep, input_names=["x", "pos", "mask"] + (list(("k", "v", "ks", "vs")) if KVWRITE == "input" else
                                                                           ["kc", "kcs", "vc", "vcs"] if KVWRITE == "in8" else []),
                                  output_names=["y", "o"], state_names=[] if KVWRITE == "input" else list(STATES),
                                  entrypoint_name=f"s{ctx // 1024}k")
    prog = conv.to_coreai()
    prog.optimize()
    ir = str(prog)
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)
    print(f"IR ops: slice_update {ir.count('slice_update')}, quantize {ir.count('quantize')}, scatter {ir.count('scatter')}", flush=True)
    return out


def writable_as(nd, dtype):
    import ctypes
    mv = memoryview(nd._tensor)  # noqa: SLF001
    ptr = np.frombuffer(mv, dtype=np.uint8).__array_interface__["data"][0]
    return np.ctypeslib.as_array((ctypes.c_uint8 * mv.nbytes).from_address(ptr)).view(dtype).reshape(nd.shape)


def placement(since: float) -> str:
    mans = [p for p in CACHE.glob("*/*/*/*/model.aimodelx/**/manifest.plist") if p.stat().st_mtime >= since]
    if not mans:
        return "cached"
    text = max(mans, key=lambda p: p.stat().st_mtime).read_bytes()
    return ("fully on ANE" if b"mps.fullyPlacedOnANE" in text else "NOT fully on ANE") + \
        f", {text.count(b'_ANE_region_')} ANE region refs, GPU activity {'no' if b'mps.noGPUActivity' in text else 'yes'}"


def ref(y, st, ctx, mask):
    """numpy reference from the graph's own FP16 rows and the stored states (FP64)."""
    nq = GRP * NKV * HD
    q = y[:nq].reshape(NKV, GRP, HD, T).transpose(0, 1, 3, 2).reshape(NKV, GRP * T, HD).astype(np.float64)
    kt = y[nq:nq + NKV * HD].reshape(NKV, HD, T).transpose(0, 2, 1).astype(np.float64)
    vt = y[nq + NKV * HD:].reshape(NKV, HD, T).transpose(0, 2, 1).astype(np.float64)
    kh = st["k"][:, :ctx].astype(np.float64) * st["ks"][:, :ctx, None].astype(np.float64)
    vh = st["v"][:, :ctx].astype(np.float64) * st["vs"][:, :ctx, None].astype(np.float64)
    causal = np.triu(np.full((T, T), -1e4), 1)
    sb = q @ kt.transpose(0, 2, 1) * HD ** -0.5 + np.tile(causal, (GRP, 1))
    sh = q @ kh.transpose(0, 2, 1) * HD ** -0.5 + mask[0].astype(np.float64)
    s = np.concatenate([sh, sb], -1)
    s -= s.max(-1, keepdims=True); pr = np.exp(s); pr /= pr.sum(-1, keepdims=True)
    return pr[..., :ctx] @ vh + pr[..., ctx:] @ vt


async def main() -> None:
    path = build()
    t0 = time.time()
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(ROOT), "mode": f"kv_slice_probe_v5_T{T}"},
                   f"kvslice5-T{T}-{os.getpid()}-{int(t0)}")
    dev = DJ.m6_device(); pkg = {"path": str(path), "main_hash": (path / "main.hash").read_bytes().hex(), "label": f"kv8 slice probe v5 T{T} TILE{TILE} SMAX{SMAX}"}
    aid = J.attempt(dev, pkg, functions=[f"s{c // 1024}k" for c in CTXS], request={"compute": "ane", "compile_mode": os.environ.get("MPSGRAPH_ANE_BONDED_COMPILE_MODE")})
    try:
        model = await AIModel.load(path, specialization_options=specialization_for("ane"))
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    print(f"T={T} TILE={TILE} SMAX={SMAX}; load {time.time() - t0:.1f}s; {placement(t0)}", flush=True)
    fns = {ctx: model.load_function(f"s{ctx // 1024}k") for ctx in CTXS}
    r = np.random.default_rng(1)
    kdt = np.int8 if KV8 else np.float16
    st = {"k": NDArray(np.zeros((NKV, SMAX, HD), kdt), StorageKind.IO_SURFACE),
          "v": NDArray(np.zeros((NKV, SMAX, HD), kdt), StorageKind.IO_SURFACE),
          "ks": NDArray(np.zeros(SSH, np.float16), StorageKind.IO_SURFACE),
          "vs": NDArray(np.zeros(SSH, np.float16), StorageKind.IO_SURFACE)}
    if not KV8:
        del st["ks"], st["vs"]
    ok = True
    nq = GRP * NKV * HD
    plan, pos = [], 0
    for i, ctx in enumerate(CTXS):                         # contiguous growth through the ladder: two calls per entry
        for _ in range(2):
            plan.append((ctx, pos)); pos += T
        pos = max(pos, CTXS[i] - 4 * T)                   # then jump near the top of this entry (history beyond is 0)
    for ctx, p in plan:
        if p + T > ctx:
            continue
        x = (r.standard_normal((1, C, 1, T)) * 0.05).astype(np.float16)
        mask = np.where(np.arange(ctx)[None, :] < p, 0, -1e4).astype(np.float16)      # history = rows before pos
        extra = {}
        if KVWRITE == "in8":
            rk, rv = (r.standard_normal((2, NKV, T, HD)) * 0.5).astype(np.float16)
            (kci, kcsi), (vci, vcsi) = quantize_values(rk), quantize_values(rv)
            extra = {"kc": NDArray(kci, StorageKind.IO_SURFACE), "kcs": NDArray(kcsi[..., None], StorageKind.IO_SURFACE),
                     "vc": NDArray(vci, StorageKind.IO_SURFACE), "vcs": NDArray(vcsi[..., None], StorageKind.IO_SURFACE)}
        out = await fns[ctx](inputs={"x": NDArray(x, StorageKind.IO_SURFACE), "pos": NDArray(np.array([p], np.int32)),
                                     "mask": NDArray(mask, StorageKind.IO_SURFACE), **(st if KVWRITE == "input" else {}), **extra},
                               **({} if KVWRITE == "input" else {"state": st}))
        y = out["y"].numpy().astype(np.float16)
        if KVWRITE == "input":                              # the host's append (qwen38_kv_cache.append_rows) into the state buffers
            kt_ = y[nq:nq + NKV * HD].reshape(NKV, HD, T).transpose(0, 2, 1); vt_ = y[nq + NKV * HD:].reshape(NKV, HD, T).transpose(0, 2, 1)
            if KV8:
                for n, rows in (("k", kt_), ("v", vt_)):
                    c_, s_ = quantize_values(rows)
                    writable_as(st[n], np.int8)[:, p:p + T] = c_
                    writable_as(st[n + "s"], np.float16).reshape(NKV, SMAX)[:, p:p + T] = s_
            else:
                writable_as(st["k"], np.float16)[:, p:p + T] = kt_; writable_as(st["v"], np.float16)[:, p:p + T] = vt_
        s = {n: st[n].numpy() for n in STATES}
        if KV8:
            s["ks"], s["vs"] = (s["ks"][..., 0], s["vs"][..., 0]) if SCALE_3D else (s["ks"], s["vs"])
        else:
            s["ks"] = s["vs"] = np.ones((NKV, SMAX), np.float16)
        kt = y[nq:nq + NKV * HD].reshape(NKV, HD, T).transpose(0, 2, 1)
        vt = y[nq + NKV * HD:].reshape(NKV, HD, T).transpose(0, 2, 1)
        if KVWRITE == "in8":
            kt, vt = rk, rv                                # the rows the graph was given (already quantized by the host)
        kc_ref, ks_ref = quantize_values(kt) if KV8 else (kt, np.ones((NKV, T), np.float16))
        vc_ref, vs_ref = quantize_values(vt) if KV8 else (vt, None)
        dk = np.abs(s["k"][:, p:p + T].astype(np.float32) - kc_ref.astype(np.float32)).max(); dv = np.abs(s["v"][:, p:p + T].astype(np.float32) - vc_ref.astype(np.float32)).max()
        nk = int((s["k"][:, p:p + T] != kc_ref).sum()); nvv = int((s["v"][:, p:p + T] != vc_ref).sum())
        if KVWRITE == "in8":
            print(f"   scale check: stored ks {s['ks'][0, p:p + 3]} given {kcsi[0, :3]}; stored vs {s['vs'][0, p:p + 3]} given {vcsi[0, :3]}; "
                  f"ks exact {bool(np.array_equal(s['ks'][:, p:p + T], kcsi))} vs exact {bool(np.array_equal(s['vs'][:, p:p + T], vcsi))}", flush=True)
            ks_ref = kcsi
        sks = float(np.abs(s["ks"][:, p:p + T].astype(np.float32) - ks_ref).max() / max(float(ks_ref.max()), 1e-9))
        o = out["o"].numpy().astype(np.float64)
        o_ref = ref(y, s, ctx, mask)
        eo = float(np.abs(o - o_ref).max()); so = float(np.abs(o_ref).max())
        print(f"s{ctx // 1024}k pos {p}: codes != host k {nk} v {nvv} of {NKV * T * HD} (max |d| {dk}/{dv}); "
              f"scale rel err {sks:.2e}; max|o-ref| {eo:.2e} (|o| {so:.2e})", flush=True)
        ok &= dk <= 1 and dv <= 1 and eo <= 2e-2 * max(so, 1e-3)
    for ctx in CTXS:
        p = min(ctx, SMAX) - 2 * T
        feed = {"x": NDArray((r.standard_normal((1, C, 1, T)) * 0.05).astype(np.float16), StorageKind.IO_SURFACE),
                "pos": NDArray(np.array([p], np.int32)),
                "mask": NDArray(np.where(np.arange(ctx)[None, :] < p, 0, -1e4).astype(np.float16), StorageKind.IO_SURFACE)}
        if KVWRITE == "in8":
            feed.update(kc=NDArray(np.zeros((NKV, T, HD), np.int8), StorageKind.IO_SURFACE), kcs=NDArray(np.ones((NKV, T, 1), np.float16), StorageKind.IO_SURFACE),
                        vc=NDArray(np.zeros((NKV, T, HD), np.int8), StorageKind.IO_SURFACE), vcs=NDArray(np.ones((NKV, T, 1), np.float16), StorageKind.IO_SURFACE))
        kw = {"inputs": {**feed, **st}} if KVWRITE == "input" else {"inputs": feed, "state": st}
        await fns[ctx](**kw)
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            await fns[ctx](**kw)
            ts.append(1e3 * (time.perf_counter() - t))
        print(f"s{ctx // 1024}k: {np.median(ts):.3f} ms (p10 {np.percentile(ts, 10):.3f})", flush=True)
    print("RESULT:", "kv8 in-graph writes + attention correct" if ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
