"""FP8 P.V A/B on the served attention path (18 Nov 2026): fold_check.py's setup (the served engine's environment, KV8 +
K8V16, random INT8 KV segments, attention layer 3, seeded) through libgpu_v8_f8.dylib with the mode from the env
(F8 = "" plain fp16 P.V | c | v, F8VS), one mode per process (F8 changes the V copy's layout at load). Per position:
GPU ms per 128-row pair call (median) and the mixer output (out - in), saved for the comparison.
    F8=c MainPY f8_check.py OUT_PREFIX P,P,... [CALLS] [LMAX]      -> OUT_PREFIX.json, OUT_PREFIX.npz
    MainPY f8_check.py --compare BASE.npz OTHER.npz [...]"""
import ctypes as C, json, mmap, os, sys
from pathlib import Path
import numpy as np

if sys.argv[1] == "--compare":
    base = np.load(sys.argv[2])
    for f in sys.argv[3:]:
        o = np.load(f)
        for k in base.files:
            d, b = o[k], base[k]
            rel = float(np.sqrt(np.mean((d - b) ** 2)) / np.sqrt(np.mean(b ** 2)))
            print(json.dumps({"file": Path(f).name, "pos": k, "rel_rms_vs_base": rel, "max_abs": float(np.max(np.abs(d - b))),
                              "finite": bool(np.isfinite(d).all())}))
    sys.exit(0)
P = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe"); E = P / "engine"
out = sys.argv[1]; pos_list = [int(x) for x in sys.argv[2].split(",")]
calls = int(sys.argv[3]) if len(sys.argv) > 3 else 12
LMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 98304
os.environ.update({"V8_MANIFEST": os.environ.get("MANIFEST", str(P / "w_dec8/manifest_best.json")), "V8_KERNELS": str(E / "kernels.metal"), "NT": "128",
                   "LMAX": str(LMAX), "KV8": "1", "K8V16": "1", "QK8": "1", "TKV": "2048", "FUSE": "all", "I8": "1", "I8T": "1",
                   "I8TV": "_k256", "LUTC": "1", "I8A": "1", "I8KV": "1", "LRM": "1"})
g = C.CDLL(str(E / os.environ.get("DYLIB", "libgpu_v8_f8.dylib")))
g.v8_init.restype = C.c_int32
g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
g.v8_wait2.restype, g.v8_wait2.argtypes = C.c_double, [C.c_int32, C.POINTER(C.c_double)]
g.v8_submit_pair.restype, g.v8_submit_pair.argtypes = C.c_int32, [C.c_int32] * 6
g.v8_kv8_seg.restype, g.v8_kv8_seg.argtypes = C.c_int32, [C.c_int32] * 12
g.v8_kv8_v16_fill.restype, g.v8_kv8_v16_fill.argtypes = C.c_double, [C.c_int32]
g.v8_prof_dump.restype = None
mode = os.environ.get("F8", "")
print("gpu layers", g.v8_init(), "LMAX", LMAX, "F8", repr(mode), "F8VS", os.environ.get("F8VS", "1"), flush=True)
if hasattr(g, "v8_variant"):   # the loaded variant (dylib globals), before any timed call; EXPECT_PV: abort unless it matches
    _b = C.create_string_buffer(4096); g.v8_variant.restype, g.v8_variant.argtypes = C.c_int32, [C.c_char_p, C.c_int32]
    _r = g.v8_variant(_b, 4096); _v = _b.value.decode().split(" | kernels=")[0]
    print("BANNER", _v, "| dylib", os.environ.get("DYLIB", "libgpu_v8_f8.dylib"), flush=True)
    if _r < 0 or ("EXPECT_PV" in os.environ and f"pv={os.environ['EXPECT_PV']} " not in _v) \
            or ("EXPECT_QK" in os.environ and f"qk={os.environ['EXPECT_QK']} " not in _v):
        print("ABORT: loaded variant does not match EXPECT_PV / EXPECT_QK", flush=True); sys.exit(3)
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
print("V copy fill %.1f ms" % g.v8_kv8_v16_fill(LMAX), flush=True)
blk = [wrap(5120 * 64 * 2) for _ in range(4)]
for m, _ in blk[:2]:
    np.frombuffer(m, np.float16)[:] = (rng.standard_normal(5120 * 64) * 0.5).astype(np.float16)
xin = np.concatenate([np.frombuffer(blk[0][0], np.float16), np.frombuffer(blk[1][0], np.float16)]).astype(np.float32)
ts = (C.c_double * 2)()
res, outs = [], {}
for p0 in pos_list:
    ms = []
    for _ in range(calls + 2):
        t = g.v8_submit_pair(L, blk[0][1], blk[1][1], blk[2][1], blk[3][1], p0)
        assert t > 0, t
        ms.append(g.v8_wait2(t, ts))
    y = np.concatenate([np.frombuffer(blk[2][0], np.float16), np.frombuffer(blk[3][0], np.float16)]).astype(np.float32)
    outs[str(p0)] = y - xin
    r = {"pos": p0, "F8": mode, "F8VS": os.environ.get("F8VS", "1"), "ms_median": round(float(np.median(ms[2:])), 3),
         "ms_min": round(float(np.min(ms[2:])), 3), "finite": bool(np.isfinite(y).all())}
    res.append(r); print(json.dumps(r), flush=True)
    if os.environ.get("PROFILE"):
        g.v8_prof_dump()
Path(out + ".json").write_text(json.dumps(res, indent=1))
np.savez(out + ".npz", **outs)
