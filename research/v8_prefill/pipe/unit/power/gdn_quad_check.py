"""256-row DeltaNet call check (queue 9b, 19 Nov 2026): one GDN layer (layer 0) on four 64-row blocks, as one quad call
on libgpu_v8_g256.dylib (T = 256) against two pair calls on the served libgpu_v8_f8q.dylib (T = 128), same inputs and
fresh state; outputs compared, then GPU ms per 256 tokens of each, alone. Two engine instances in one process.
    MainPY gdn_quad_check.py [CALLS]"""
import ctypes as C, json, mmap, os, sys
from pathlib import Path
import numpy as np

P = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe"); E = P / "engine"
M0 = "/Volumes/SSD4TB/anemll-forge-research/contention/gdn/manifest_L0.json"
calls = int(sys.argv[1]) if len(sys.argv) > 1 else 50
os.environ.update({"V8_MANIFEST": M0, "V8_KERNELS": str(E / "kernels.metal"), "NT": "128", "LMAX": "98304", "KV8": "1", "K8V16": "1",
                   "QK8": "1", "TKV": "2048", "FUSE": "all", "I8": "1", "I8T": "1", "I8TV": "_k256", "LUTC": "1", "I8A": "1", "I8KV": "1",
                   "LRM": "1", "F8": "x", "QKE": "tg"})
A = C.CDLL(str(E / "libgpu_v8_f8q.dylib"))
os.environ.update({"V8G_NT": "256", "V8G_MANIFEST": M0, "V8G_I8TV": os.environ.get("QUAD_I8TV", "_k256")})
B = C.CDLL(str(E / "libgpu_v8_g256.dylib"))
for g in (A, B):
    g.v8_init.restype = C.c_int32
    g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
    g.v8_wait2.restype, g.v8_wait2.argtypes = C.c_double, [C.c_int32, C.POINTER(C.c_double)]
    g.v8_reset.restype = None
A.v8_submit_pair.restype, A.v8_submit_pair.argtypes = C.c_int32, [C.c_int32] * 6
B.v8_submit_quad.restype, B.v8_submit_quad.argtypes = C.c_int32, [C.c_int32] * 11
print("layers A", A.v8_init(), "B", B.v8_init(), flush=True)
keep = []


def wrap(g, m, n):
    return g.v8_wrap(C.addressof(C.c_char.from_buffer(m)), n)


rng = np.random.default_rng(0)
n = 5120 * 64 * 2
src = [mmap.mmap(-1, n) for _ in range(4)]; outA = [mmap.mmap(-1, n) for _ in range(4)]; outB = [mmap.mmap(-1, n) for _ in range(4)]
for m in src:
    np.frombuffer(m, np.float16)[:] = (rng.standard_normal(5120 * 64) * 0.5).astype(np.float16)
hA = [wrap(A, m, n) for m in src]; oA = [wrap(A, m, n) for m in outA]
hB = [wrap(B, m, n) for m in src]; oB = [wrap(B, m, n) for m in outB]
ts = (C.c_double * 2)()


def runA():
    A.v8_reset(); ms = 0
    for j in (0, 2):
        t = A.v8_submit_pair(0, hA[j], hA[j + 1], oA[j], oA[j + 1], 64 * j); assert t > 0; ms += A.v8_wait2(t, ts)
    return ms


def runB():
    B.v8_reset()
    t = B.v8_submit_quad(0, *hB, *oB, 0, 256); assert t > 0, t
    return B.v8_wait2(t, ts)


runA(); runB()
ya = np.concatenate([np.frombuffer(m, np.float16) for m in outA]); yb = np.concatenate([np.frombuffer(m, np.float16) for m in outB])
print(json.dumps({"bitexact": bool(np.array_equal(ya, yb)), "max_abs": float(np.max(np.abs(ya.astype(np.float32) - yb.astype(np.float32)))),
                  "finite": bool(np.isfinite(yb).all())}), flush=True)
ma = sorted(runA() for _ in range(calls)); mb = sorted(runB() for _ in range(calls))
print(json.dumps({"pairs_ms_per_256": round(ma[len(ma) // 2], 4), "quad_ms_per_256": round(mb[len(mb) // 2], 4)}), flush=True)
runA(); runB()
for j in range(4):
    a = np.frombuffer(outA[j], np.float16).astype(np.float32).reshape(5120, 64); b = np.frombuffer(outB[j], np.float16).astype(np.float32).reshape(5120, 64)
    x = np.frombuffer(src[j], np.float16).astype(np.float32).reshape(5120, 64)
    d = np.abs(a - b); da = (a - x)   # the mixer's contribution (out - in)
    rel = float(np.sqrt(((a - b) ** 2).mean() / (da ** 2).mean()))
    cols = np.where(d.max(0) > 0)[0]
    print(f"block {j}: max|diff| {d.max():.4f}  rel to the mixer delta {rel:.2e}  differing columns {len(cols)} (first {cols[:8].tolist()})", flush=True)
a = np.concatenate([np.frombuffer(m, np.float16) for m in outA]); b = np.concatenate([np.frombuffer(m, np.float16) for m in outB])
ulp = np.abs(np.spacing(a.astype(np.float16))).astype(np.float32)
d = np.abs(a.astype(np.float32) - b.astype(np.float32))
print(f"differing elements {int((d > 0).sum())} of {d.size} ({100 * (d > 0).mean():.3f}%), max diff in ULPs {float((d / ulp).max()):.2f}, "
      f"elements > 1 ULP {int((d > 1.01 * ulp).sum())}", flush=True)
