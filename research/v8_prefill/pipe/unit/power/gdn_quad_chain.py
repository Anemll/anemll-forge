"""Chained 256-row check (queue 9b): the first six GDN layers in sequence on four blocks (mixer only, no FFN between),
quad calls on libgpu_v8_g256 vs pair calls on libgpu_v8_f8q, p0 = 0 then a second 256-token span at p0 = 256 (state
carried), outputs compared after every layer."""
import ctypes as C, json, mmap, os, sys
from pathlib import Path
import numpy as np
P = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe"); E = P / "engine"
M6 = "/Volumes/SSD4TB/anemll-forge-research/contention/gdn/manifest_G6.json"
os.environ.update({"V8_MANIFEST": M6, "V8_KERNELS": str(E / "kernels.metal"), "NT": "128", "LMAX": "98304", "KV8": "1", "K8V16": "1",
                   "QK8": "1", "TKV": "2048", "FUSE": "all", "I8": "1", "I8T": "1", "I8TV": "_k256", "LUTC": "1", "I8A": "1", "I8KV": "1",
                   "LRM": "1", "F8": "x", "QKE": "tg"})
A = C.CDLL(str(E / "libgpu_v8_f8q.dylib"))
os.environ.update({"V8G_NT": "256", "V8G_MANIFEST": M6, "V8G_I8TV": os.environ.get("QUAD_I8TV", "_k256ws")})
B = C.CDLL(str(E / "libgpu_v8_g256.dylib"))
for g in (A, B):
    g.v8_init.restype = C.c_int32; g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
    g.v8_wait2.restype, g.v8_wait2.argtypes = C.c_double, [C.c_int32, C.POINTER(C.c_double)]; g.v8_reset.restype = None
A.v8_submit_pair.restype, A.v8_submit_pair.argtypes = C.c_int32, [C.c_int32] * 6
B.v8_submit_quad.restype, B.v8_submit_quad.argtypes = C.c_int32, [C.c_int32] * 11
print("layers", A.v8_init(), B.v8_init())
LAY = [l["i"] for l in json.loads(Path(M6).read_text())["layers"]]
n = 5120 * 64 * 2; rng = np.random.default_rng(1)
def mk(): return [mmap.mmap(-1, n) for _ in range(4)]
bufA = [mk(), mk()]; bufB = [mk(), mk()]   # ping-pong
ts = (C.c_double * 2)()
hA = [[A.v8_wrap(C.addressof(C.c_char.from_buffer(m)), n) for m in s] for s in bufA]
hB = [[B.v8_wrap(C.addressof(C.c_char.from_buffer(m)), n) for m in s] for s in bufB]
A.v8_reset(); B.v8_reset()
for span, p0 in enumerate((0, 256)):
    x0 = [(rng.standard_normal(5120 * 64) * 0.5).astype(np.float16) for _ in range(4)]
    for j in range(4):
        np.frombuffer(bufA[0][j], np.float16)[:] = x0[j]; np.frombuffer(bufB[0][j], np.float16)[:] = x0[j]
    cur = 0
    for L in LAY:
        nx = 1 - cur
        for j in (0, 2):
            t = A.v8_submit_pair(L, hA[cur][j], hA[cur][j + 1], hA[nx][j], hA[nx][j + 1], p0 + 64 * j); assert t > 0; A.v8_wait2(t, ts)
        t = B.v8_submit_quad(L, *hB[cur], *hB[nx], p0, 256); assert t > 0, t; B.v8_wait2(t, ts)
        ya = np.concatenate([np.frombuffer(m, np.float16) for m in bufA[nx]]); yb = np.concatenate([np.frombuffer(m, np.float16) for m in bufB[nx]])
        print(f"span {span} (p0 {p0}) layer {L}: bit-exact {np.array_equal(ya, yb)}  max|diff| {float(np.max(np.abs(ya.astype(np.float32) - yb.astype(np.float32)))):.4g}")
        cur = nx
