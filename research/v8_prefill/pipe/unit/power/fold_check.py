"""Softmax-fold A/B on the served attention path (18 Nov 2026). The served engine's environment (w_dec8 tables, FUSE / I8
/ ... QK8, TKV 2048) with KV8 + K8V16 (INT8 keys from segments in the ANE's layout, the GPU's fp16 V copy), through
libgpu_v8_fold.dylib (gpu_v8_fold.swift: srv3 + FOLD). Random INT8 KV codes, attention layer 3. Per position, on the
same data in the same process (v8_fold switches at run time): the plain path and each fold form; GPU ms per 128-row
pair call (median) and the mixer output (out - in) against the plain path's. PROFILE=1 (env) adds per-kernel shares.
    MainPY fold_check.py OUT.json P,P,... [CALLS] [LMAX]"""
import ctypes as C, json, mmap, os, sys, time
from pathlib import Path
import numpy as np

P = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe"); E = P / "engine"
out = Path(sys.argv[1]); pos_list = [int(x) for x in sys.argv[2].split(",")]
calls = int(sys.argv[3]) if len(sys.argv) > 3 else 12
LMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 98304
assert max(pos_list) + 128 <= LMAX
os.environ.update({"V8_MANIFEST": str(P / "w_dec8/manifest_best.json"), "V8_KERNELS": str(E / "kernels.metal"), "NT": "128",
                   "LMAX": str(LMAX), "KV8": "1", "K8V16": "1", "QK8": "1", "TKV": "2048", "FUSE": "all", "I8": "1", "I8T": "1",
                   "I8TV": "_k256", "LUTC": "1", "I8A": "1", "I8KV": "1", "LRM": "1"})
g = C.CDLL(str(E / os.environ.get("DYLIB", "libgpu_v8_fold.dylib")))
g.v8_init.restype = C.c_int32
g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
g.v8_wait2.restype, g.v8_wait2.argtypes = C.c_double, [C.c_int32, C.POINTER(C.c_double)]
g.v8_submit_pair.restype, g.v8_submit_pair.argtypes = C.c_int32, [C.c_int32] * 6
g.v8_kv8_seg.restype, g.v8_kv8_seg.argtypes = C.c_int32, [C.c_int32] * 12
g.v8_kv8_v16_fill.restype, g.v8_kv8_v16_fill.argtypes = C.c_double, [C.c_int32]
g.v8_fold.restype, g.v8_fold.argtypes = C.c_int32, [C.c_int32, C.c_int32]
g.v8_prof_dump.restype = None
print("gpu layers", g.v8_init(), "LMAX", LMAX, flush=True)
L = 3
rng = np.random.default_rng(0)
keep = []


def wrap(nbytes):
    m = mmap.mmap(-1, nbytes); keep.append(m)
    return m, g.v8_wrap(C.addressof(C.c_char.from_buffer(m)), nbytes)


start = 0
for rows in (8192, 8192, 16384, 32768, 32768):
    if start >= LMAX:
        break
    rows = min(rows, LMAX - start)
    k, hk = wrap(4 * 256 * rows); v, hv = wrap(4 * rows * 256); ks, hks = wrap(4 * rows * 2); vs, hvs = wrap(4 * rows * 2)
    np.frombuffer(k, np.int8)[:] = rng.integers(-127, 128, 4 * 256 * rows, dtype=np.int8)
    np.frombuffer(v, np.int8)[:] = rng.integers(-127, 128, 4 * 256 * rows, dtype=np.int8)
    np.frombuffer(ks, np.float16)[:] = rng.uniform(0.004, 0.02, 4 * rows).astype(np.float16)
    np.frombuffer(vs, np.float16)[:] = rng.uniform(0.004, 0.02, 4 * rows).astype(np.float16)
    assert g.v8_kv8_seg(L, start, rows, hk, hv, hks, hvs, 256 * rows, rows, rows * 256, 256, rows) > 0
    start += rows
print("K8V16 fill %.1f ms" % g.v8_kv8_v16_fill(LMAX), flush=True)
blk = [wrap(5120 * 64 * 2) for _ in range(4)]
for m, _ in blk[:2]:
    np.frombuffer(m, np.float16)[:] = (rng.standard_normal(5120 * 64) * 0.5).astype(np.float16)
xin = np.concatenate([np.frombuffer(blk[0][0], np.float16), np.frombuffer(blk[1][0], np.float16)]).astype(np.float32)
ts = (C.c_double * 2)()
MODES = [(0, "plain"), (1, "fold64"), (2, "fold64s"), (3, "fold128"), (0, "plain_again")]


def run(p0, n):
    ms = []
    for _ in range(n):
        t = g.v8_submit_pair(L, blk[0][1], blk[1][1], blk[2][1], blk[3][1], p0)
        assert t > 0, t
        ms.append(g.v8_wait2(t, ts))
    y = np.concatenate([np.frombuffer(blk[2][0], np.float16), np.frombuffer(blk[3][0], np.float16)]).astype(np.float32)
    return ms, y - xin


res = []
for p0 in pos_list:
    base = None
    for mode, name in MODES:
        assert g.v8_fold(mode, 0) == 1
        ms, d = run(p0, calls + 2)
        if base is None:
            base = d
        rel = float(np.sqrt(np.mean((d - base) ** 2)) / np.sqrt(np.mean(base ** 2)))
        r = {"pos": p0, "mode": name, "ms_median": round(float(np.median(ms[2:])), 3), "ms_min": round(float(np.min(ms[2:])), 3),
             "rel_rms_vs_plain": rel, "max_abs_vs_plain": float(np.max(np.abs(d - base))), "finite": bool(np.isfinite(d).all())}
        res.append(r)
        print(json.dumps(r), flush=True)
        if os.environ.get("PROFILE"):
            g.v8_prof_dump()
out.write_text(json.dumps(res, indent=1))
