"""Decode attention kernel past 96K (17 Nov 2026): engine/libdec_gpu.dylib on one synthetic attention layer with KV
segments 8K + 8K + 16K + 32K + 64K (128K keys; the 5th segment twice the 96K build's, as a 128K split-only ladder
would allocate it), page-aligned host buffers wrapped zero-copy. At p0 near 100K, 120K and 128K: the 8 new K / V rows
written by dec_prep vs the host's quantize (bit-exact), and o vs exact FP32 attention (numpy; the kernel's INT8 q
emulated per row as the reference's q too, so only the kernel's own error remains), plus GPU ms.
    PY dec_gpu_128k.py"""
import ctypes as C_, mmap, os, sys, time
from pathlib import Path

import numpy as np

ENG = Path(__file__).resolve().parents[1] / "engine"
SEG = [8192, 8192, 16384, 32768, 65536]
NH, NKV, HD, T = 24, 4, 256, 8
keep = []


def abuf(shape, dtype):
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    m = mmap.mmap(-1, (n + 16383) // 16384 * 16384)
    keep.append(m)
    return np.frombuffer(m, dtype, count=int(np.prod(shape))).reshape(shape)


def quant(x):   # the host's quantize_values: per (head, row) scale over the head dim
    s = np.maximum(np.abs(x.astype(np.float32)).max(-1) / 127, 1e-6).astype(np.float16)
    return np.clip(np.rint(x.astype(np.float32) / s.astype(np.float32)[..., None]), -127, 127).astype(np.int8), s


def main():
    g = C_.CDLL(str(ENG / os.environ.get("DEC_DYLIB", "libdec_gpu.dylib")))
    g.dec_init.restype, g.dec_init.argtypes = C_.c_int32, [C_.c_char_p]
    g.dec_wrap.restype, g.dec_wrap.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t]
    g.dec_seg.restype, g.dec_seg.argtypes = C_.c_int32, [C_.c_int32] * 7
    g.dec_run.restype, g.dec_run.argtypes = C_.c_double, [C_.c_int32, C_.POINTER(C_.c_int32), C_.POINTER(C_.c_uint32), C_.c_int32]
    assert g.dec_init(str(ENG / "dec_kernels.metal").encode()) > 0
    w = lambda a: g.dec_wrap(a.ctypes.data, a.nbytes)  # noqa: E731
    r = np.random.default_rng(9)
    segs, start = [], 0
    for L in SEG:
        K, V = abuf((NKV, HD, L), np.int8), abuf((NKV, L, HD), np.int8)
        KS, VS = abuf((NKV, L), np.float16), abuf((NKV, L), np.float16)
        K[...] = r.integers(-127, 128, K.shape, dtype=np.int8); V[...] = r.integers(-127, 128, V.shape, dtype=np.int8)
        KS[...] = (0.01 + 0.02 * r.random(KS.shape)).astype(np.float16); VS[...] = (0.003 + 0.01 * r.random(VS.shape)).astype(np.float16)
        assert g.dec_seg(0, start, L, w(K), w(KS), w(V), w(VS)) > 0
        segs.append((start, L, K, KS, V, VS)); start += L
    q, gt = abuf((NH, T, HD), np.float16), abuf((NH, T, HD), np.float16)
    kt, vt = abuf((NKV, T, HD), np.float16), abuf((NKV, T, HD), np.float16)
    o = abuf((NH * HD * T,), np.float16)
    H = (C_.c_int32 * 5)(w(q), w(gt), w(kt), w(vt), w(o))
    st = (C_.c_uint32 * 9)(T * HD, HD, T * HD, HD, T * HD, HD, T * HD, HD, T)
    ok = True
    for p0 in (100000, 120000, sum(SEG) - T):
        q[...] = (r.standard_normal(q.shape) * 1.0).astype(np.float16); gt[...] = r.standard_normal(gt.shape).astype(np.float16)
        kt[...] = (r.standard_normal(kt.shape) * 2.5).astype(np.float16); vt[...] = (r.standard_normal(vt.shape) * 0.5).astype(np.float16)
        ms = [g.dec_run(0, H, st, p0) for _ in range(40)][20:]
        assert min(ms) >= 0, ms
        n = p0 + T
        Kh = np.empty((NKV, n, HD), np.float32); Vh = np.empty((NKV, n, HD), np.float32)
        wrote = {"k": [], "ks": [], "v": [], "vs": []}
        for (s0, L, K, KS, V, VS) in segs:
            a, b = s0, min(s0 + L, n)
            if a >= b:
                continue
            Kh[:, a:b] = K[:, :, :b - a].transpose(0, 2, 1).astype(np.float32) * KS[:, :b - a].astype(np.float32)[..., None]
            Vh[:, a:b] = V[:, :b - a].astype(np.float32) * VS[:, :b - a].astype(np.float32)[..., None]
            lo, hi = max(a, p0), b
            if lo < hi:
                wrote["k"].append(K[:, :, lo - s0:hi - s0].transpose(0, 2, 1)); wrote["ks"].append(KS[:, lo - s0:hi - s0])
                wrote["v"].append(V[:, lo - s0:hi - s0]); wrote["vs"].append(VS[:, lo - s0:hi - s0])
        kc, ksc = quant(kt); vc, vsc = quant(vt)
        dif = {a_: int((np.concatenate(wrote[a_], 1).astype(np.float32) != b_.astype(np.float32)).sum()) for a_, b_ in (("k", kc), ("ks", ksc), ("v", vc), ("vs", vsc))}
        same = not any(dif.values())
        if not same: print("   differing elements", dif, "max code diff", int(np.abs(np.concatenate(wrote["k"], 1).astype(int) - kc.astype(int)).max()), flush=True)
        qc, qs = quant(q)                                   # the kernel's INT8 q (per row), emulated
        qf = qc.astype(np.float32) * qs.astype(np.float32)[..., None]
        ref = np.zeros((NH, T, HD), np.float32)
        for h in range(NH):
            kv = h // (NH // NKV)
            s = qf[h] @ Kh[kv].T / np.sqrt(HD)              # (T, n)
            s[np.arange(n)[None, :] > (p0 + np.arange(T))[:, None]] = -np.inf
            s -= s.max(1, keepdims=True); p = np.exp(s); p /= p.sum(1, keepdims=True)
            ref[h] = p @ Vh[kv]
        ref /= 1 + np.exp(-gt.astype(np.float32))
        got = o.reshape(NH, HD, T).transpose(0, 2, 1).astype(np.float32)
        err = float(np.abs(got - ref).max() / np.abs(ref).max())
        print(f"p0 {p0} ({n} keys): new K/V rows {'bit-exact' if same else 'DIFFER'}; o vs exact (INT8 q emulated) "
              f"{100 * err:.3f}% of max |o|; GPU {np.median(ms[1:]):.3f} ms ({np.median(ms[1:]) / n * 1024:.4f} ms per 1K keys)", flush=True)
        ok = ok and same and err < 0.01
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


main()
