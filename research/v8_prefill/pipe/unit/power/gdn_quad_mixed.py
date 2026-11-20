"""Mixed chain check (queue 9b debug): layers 0-7 (GDN 0-2, 4-6; attention 3, 7) on 512 tokens, layer-major, no FFN
between layers. Reference: every layer as 128-row pair calls on libgpu_v8_f8q. Test: GDN layers as 256-row quad calls on
libgpu_v8_g256, attention layers as pair calls on libgpu_v8_f8q. KV8 segments in host memory (as f8_check.py).
Outputs compared after every layer."""
import ctypes as C, json, mmap, os
from pathlib import Path
import numpy as np
P = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe"); E = P / "engine"
MA = "/Volumes/SSD4TB/anemll-forge-research/contention/gdn/manifest_L0_7.json"; MG = "/Volumes/SSD4TB/anemll-forge-research/contention/gdn/manifest_G0_7.json"
LMAX = 8192
os.environ.update({"V8_MANIFEST": MA, "V8_KERNELS": str(E / "kernels.metal"), "NT": "128", "LMAX": str(LMAX), "KV8": "1", "K8V16": "1",
                   "QK8": "1", "TKV": "2048", "FUSE": "all", "I8": "1", "I8T": "1", "I8TV": "_k256", "LUTC": "1", "I8A": "1", "I8KV": "1",
                   "LRM": "1", "F8": "x", "QKE": "tg"})
A = C.CDLL(str(E / "libgpu_v8_f8q.dylib"))
os.environ.update({"V8G_NT": "256", "V8G_MANIFEST": MG, "V8G_I8TV": "_k256"})
B = C.CDLL(str(E / "libgpu_v8_g256.dylib"))
for g in (A, B):
    g.v8_init.restype = C.c_int32; g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
    g.v8_wait2.restype, g.v8_wait2.argtypes = C.c_double, [C.c_int32, C.POINTER(C.c_double)]; g.v8_reset.restype = None
A.v8_submit_pair.restype, A.v8_submit_pair.argtypes = C.c_int32, [C.c_int32] * 6
A.v8_kv8_seg.restype, A.v8_kv8_seg.argtypes = C.c_int32, [C.c_int32] * 12
B.v8_submit_quad.restype, B.v8_submit_quad.argtypes = C.c_int32, [C.c_int32] * 11
print("layers", A.v8_init(), B.v8_init())
keep = []
def wrapm(g, m, n): return g.v8_wrap(C.addressof(C.c_char.from_buffer(m)), n)
KV = {}
for L in (3, 7):
    rows = LMAX; k, v, ks, vs = (mmap.mmap(-1, x) for x in (4 * 256 * rows, 4 * rows * 256, 4 * rows * 2, 4 * rows * 2)); keep += [k, v, ks, vs]
    KV[L] = (k, v, ks, vs)
    assert A.v8_kv8_seg(L, 0, rows, *[wrapm(A, x, len(x)) for x in (k, v, ks, vs)], 256 * rows, rows, rows * 256, 256, rows) > 0
n = 5120 * 64 * 2; rng = np.random.default_rng(2)
NB = 8   # 512 tokens
def mk(): return [mmap.mmap(-1, n) for _ in range(NB)]
RA = [mk(), mk()]; RT = [mk(), mk()]
hA = {id(m): wrapm(A, m, n) for s in RA + RT for m in s}; hB = {id(m): wrapm(B, m, n) for s in RT for m in s}
ts = (C.c_double * 2)()
x0 = [(rng.standard_normal(5120 * 64) * 0.5).astype(np.float16) for _ in range(NB)]
def run(test):
    bufs = RT if test else RA
    for s in KV.values():
        for x in s: x[:] = b"\0" * len(x)
    A.v8_reset(); B.v8_reset()
    for j in range(NB): np.frombuffer(bufs[0][j], np.float16)[:] = x0[j]
    cur, outs = 0, []
    for L in range(8):
        nx = 1 - cur
        if test and L not in (3, 7):
            for q in (0, 4):
                t = B.v8_submit_quad(L, *[hB[id(bufs[cur][q + i])] for i in range(4)], *[hB[id(bufs[nx][q + i])] for i in range(4)], 64 * q, 256); assert t > 0, t; B.v8_wait2(t, ts)
        else:
            for p in range(0, NB, 2):
                t = A.v8_submit_pair(L, hA[id(bufs[cur][p])], hA[id(bufs[cur][p + 1])], hA[id(bufs[nx][p])], hA[id(bufs[nx][p + 1])], 64 * p); assert t > 0, t; A.v8_wait2(t, ts)
        outs.append(np.concatenate([np.frombuffer(m, np.float16).copy() for m in bufs[nx]]))
        cur = nx
    return outs
ra, rt = run(False), run(True)
for L in range(8):
    d = np.abs(ra[L].astype(np.float32) - rt[L].astype(np.float32))
    print(f"layer {L} ({'att' if L in (3, 7) else 'gdn'}): bit-exact {np.array_equal(ra[L], rt[L])}  max|diff| {d.max():.4g}")
