"""v4 = v3 with the window attended in history tiles of TILE rows (env TILE, default 4096), two-pass softmax
(global max over all tiles first, as the Forge release graph default), so no tensor wider than TILE along the history.
v3: v1's state layout (NKV, SMAX, D) (its slice_update was ANE-placed) with ANE-friendlier attention:
scores (NKV, ctx, T) = k_win @ q^T, softmax over ctx (axis 1), o (NKV, T, D) = p^T @ v_win; no transpose of the window,
no broadcast matmul, no repeat. (v2 = (1, NKV, D, SMAX) layout: 0% ANE.)
Two Core AI states k, v.
Each entry s<ctx>k writes the call's T rows at runtime position pos with a slice assignment (-> coreai.slice_update)
into both states, then attends over the static window [..., :ctx] with 4D batched matmuls and no transpose of the
window: scores^T = k_win^T q is computed as (q^T k_win) -> (1, NKV, T, ctx); softmax over ctx; o = v_win p^T
-> (1, NKV, D, T). Checks: rows land at pos and persist (both states), earlier rows untouched, o == numpy reference,
placement (MPSGRAPH_PRINT_ANE_PLACEMENT_ANALYSIS in the log, plus manifest flags), ms per call per window.
Every AIModel.load is journaled (deploy_journal v7).
    T_ROWS=8|64 python kv_slice_probe_v2.py"""
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
from coreai.runtime import AIModel, NDArray
from coreai.runtime._ndarray import StorageKind
from coreai_opt.casting import cast_to_16_bit_precision

from coreai_bench_helpers import specialization_for

C, D, NKV = 512, 256, 4
SMAX = int(os.environ.get("SMAX", "65536"))
T = int(os.environ.get("T_ROWS", "8"))
TILE = int(os.environ.get("TILE", "4096"))
CTXS = tuple(int(c) for c in os.environ.get("CTXS", "2048,16384,65536").split(","))
ROOT = Path(__file__).resolve().parent / (f"artifacts_kv_slice_v4_tile{TILE}_T{T}" + ("" if SMAX == 65536 else f"_S{SMAX}"))
CACHE = Path.home() / "Library/Caches/coreai-cache"


class SliceKV(nn.Module):
    def __init__(self, ctx: int) -> None:
        super().__init__()
        self.ctx = ctx
        torch.manual_seed(0)
        self.conv = nn.Conv2d(C, 3 * D, 1, bias=False)               # q, k, v: D channels each (shared by the NKV heads)
        self.register_buffer("k", torch.zeros(NKV, SMAX, D))
        self.register_buffer("v", torch.zeros(NKV, SMAX, D))

    def forward(self, x, pos, mask):                     # x (1, C, 1, T), pos (1,) int32, mask (1, ctx, 1) additive
        y = self.conv(x)                                 # (1, 3D, 1, T)
        q = y[0, :D, 0, :]                               # (D, T)
        kk = y[0, D:2 * D, 0, :].transpose(0, 1)         # (T, D)
        vv = y[0, 2 * D:, 0, :].transpose(0, 1)
        p = pos.item()
        torch._check(p >= 0)
        torch._check(p <= SMAX - T)
        self.k[:, p:p + T] = kk.unsqueeze(0).expand(NKV, T, D)
        self.v[:, p:p + T] = vv.unsqueeze(0).expand(NKV, T, D)
        qe = q.unsqueeze(0).expand(NKV, D, T) * D ** -0.5
        ss = []
        for a in range(0, self.ctx, TILE):               # pass 1: scores per tile and the global row max
            b = min(a + TILE, self.ctx)
            ss.append(torch.matmul(self.k[:, a:b], qe) + mask[:, a:b])          # (NKV, tile, T)
        m = torch.stack([si.amax(1) for si in ss], 0).amax(0).unsqueeze(1)     # (NKV, 1, T)
        num, den = None, None
        for i, a in enumerate(range(0, self.ctx, TILE)):  # pass 2: exp against the global max, PV per tile
            b = min(a + TILE, self.ctx)
            e = torch.exp(ss[i] - m)
            pv = torch.matmul(e.transpose(1, 2), self.v[:, a:b])                # (NKV, T, D)
            sm = e.sum(1)                                                        # (NKV, T)
            num = pv if num is None else num + pv
            den = sm if den is None else den + sm
        o = num / den.unsqueeze(2)
        return y, o


def build() -> Path:
    out = ROOT / "slice_kv_v4.aimodel"
    if out.exists():
        return out
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    for ctx in CTXS:
        m = SliceKV(ctx).eval().to(torch.float16)
        ex = (torch.randn(1, C, 1, T, dtype=torch.float16), torch.tensor([16], dtype=torch.int32),
              torch.zeros(1, ctx, 1, dtype=torch.float16))
        ep = torch.export.export(m, ex, strict=False).run_decompositions(coreai_torch.get_decomp_table())
        cast_to_16_bit_precision(ep)
        conv.add_exported_program(ep, input_names=["x", "pos", "mask"], output_names=["y", "o"], state_names=["k", "v"],
                                  entrypoint_name=f"s{ctx // 1024}k")
    prog = conv.to_coreai()
    prog.optimize()
    ir = str(prog)
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)
    print(f"IR ops: slice_update {ir.count('slice_update')}, scatter {ir.count('scatter')}, transpose {ir.count('transpose')}", flush=True)
    return out


def placement(since: float) -> str:
    mans = [p for p in CACHE.glob("*/*/*/*/model.aimodelx/**/manifest.plist") if p.stat().st_mtime >= since]
    if not mans:
        return "cached"
    text = max(mans, key=lambda p: p.stat().st_mtime).read_bytes()
    return ("fully on ANE" if b"mps.fullyPlacedOnANE" in text else "NOT fully on ANE") + \
        f", {text.count(b'_ANE_region_')} ANE region refs, GPU activity {'no' if b'mps.noGPUActivity' in text else 'yes'}"


def ref_attn(q, kw, vw, mask):                          # q (D, T); kw, vw (NKV, ctx, D); mask (1, ctx, 1)
    sc = np.einsum("hcd,dt->hct", kw, q) * D ** -0.5 + mask
    sc = sc - sc.max(1, keepdims=True); pr = np.exp(sc); pr /= pr.sum(1, keepdims=True)
    return np.einsum("hct,hcd->htd", pr, vw)


async def main() -> None:
    path = build()
    t0 = time.time()
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(ROOT), "mode": f"kv_slice_probe_v4_T{T}"},
                   f"kvslice4-T{T}-{os.getpid()}-{int(t0)}")
    dev = DJ.m6_device(); pkg = {"path": str(path), "main_hash": (path / "main.hash").read_bytes().hex(), "label": f"kv_slice probe v4 T{T} (SMAX {SMAX})"}
    aid = J.attempt(dev, pkg, functions=[f"s{c // 1024}k" for c in CTXS], request={"compute": "ane", "compile_mode": os.environ.get("MPSGRAPH_ANE_BONDED_COMPILE_MODE")})
    try:
        model = await AIModel.load(path, specialization_options=specialization_for("ane"))
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    print(f"T={T} rows, TILE={TILE}; load {time.time() - t0:.1f}s; {placement(t0)}", flush=True)
    fns = {ctx: model.load_function(f"s{ctx // 1024}k") for ctx in CTXS}
    r = np.random.default_rng(1)
    ks = NDArray(np.zeros((NKV, SMAX, D), np.float16), StorageKind.IO_SURFACE)
    vs = NDArray(np.zeros((NKV, SMAX, D), np.float16), StorageKind.IO_SURFACE)
    xs = [(r.standard_normal((1, C, 1, T)) * 0.1).astype(np.float16) for _ in range(3)]
    ok = True
    for i, (ctx, pos) in enumerate(zip(CTXS, (100, 108, CTXS[-1] - 5536 if CTXS[-1] > 8192 else CTXS[-1] - 64))):
        fn = fns[ctx]
        mask = np.where(np.arange(ctx)[None, :, None] < pos + T, 0, -1e4).astype(np.float16)
        out = await fn(inputs={"x": NDArray(xs[i], StorageKind.IO_SURFACE), "pos": NDArray(np.array([pos], np.int32)),
                               "mask": NDArray(mask, StorageKind.IO_SURFACE)}, state={"k": ks, "v": vs})
        k, v = ks.numpy().astype(np.float32), vs.numpy().astype(np.float32)
        y = out["y"].numpy()[0, :, 0, :].astype(np.float32)
        q, kk, vv = y[:D], y[D:2 * D].T, y[2 * D:].T
        ek = float(np.abs(k[:, pos:pos + T] - kk[None]).max()); ev = float(np.abs(v[:, pos:pos + T] - vv[None]).max())
        nz = sorted({int(p) for p in np.nonzero(np.abs(k).sum((0, 2)) > 0)[0]})
        o = out["o"].numpy().astype(np.float32)
        eo = float(np.abs(o - ref_attn(q, k[:, :ctx], v[:, :ctx], mask.astype(np.float32))).max())
        print(f"call {i} (s{ctx // 1024}k, pos {pos}): max|k-rows| {ek:.3g} max|v-rows| {ev:.3g}; written cols {len(nz)} "
              f"({nz[:2]}..{nz[-2:]}); max|o-ref| {eo:.3g}", flush=True)
        ok &= ek < 1e-2 and ev < 1e-2 and eo < 2e-2
    for ctx in CTXS:
        mask = np.zeros((1, ctx, 1), np.float16)
        feed = {"x": NDArray(xs[0], StorageKind.IO_SURFACE), "pos": NDArray(np.array([200], np.int32)), "mask": NDArray(mask, StorageKind.IO_SURFACE)}
        await fns[ctx](inputs=feed, state={"k": ks, "v": vs})
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            await fns[ctx](inputs=feed, state={"k": ks, "v": vs})
            ts.append(1e3 * (time.perf_counter() - t))
        print(f"s{ctx // 1024}k: {np.median(ts):.3f} ms (p10 {np.percentile(ts, 10):.3f})", flush=True)
    print("RESULT:", "writes + attention correct" if ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
