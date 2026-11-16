"""A8 expander unit check (GPU only, no ANE): v8_a8_expand of one layer into host buffers vs the host's codes from the
export (pal8[idx] in the bg / bu / bd layouts); expansion time.   MainPY a8_check.py LAYER [REPS]   (env DYLIB, MAN)"""
import ctypes as C, mmap, os, sys, time
from pathlib import Path
import numpy as np
from safetensors import safe_open

P = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe")
E = P / "engine"
EX = "/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat/"
L, reps = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 5
os.environ.update({"V8_MANIFEST": os.environ.get("MAN", str(P / "w_a8/manifest_a8.json")), "V8_KERNELS": str(E / "kernels.metal"),
                   "NT": "128", "LMAX": "256"})
g = C.CDLL(str(E / os.environ.get("DYLIB", "libgpu_v8_a8.dylib")))
g.v8_init.restype = C.c_int32
g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
g.v8_wait2.restype, g.v8_wait2.argtypes = C.c_double, [C.c_int32, C.POINTER(C.c_double)]
g.v8_a8_setup.restype, g.v8_a8_setup.argtypes = C.c_int32, [C.c_void_p, C.c_size_t, C.c_int32, C.c_int32, C.c_int32]
g.v8_a8_expand.restype, g.v8_a8_expand.argtypes = C.c_int32, [C.c_int32] * 4
print("gpu layers", g.v8_init())
pal8 = np.fromfile(P / "w_a8/pal8.bin", np.int8)
assert g.v8_a8_setup(pal8.ctypes.data, pal8.nbytes, 5120, 2, 5120) == 0
pal8 = pal8.reshape(64, 3, 64, 2)
n = 17 * 5120 * 1024
bufs = [mmap.mmap(-1, n) for _ in range(3)]
hs = [g.v8_wrap(C.addressof(C.c_char.from_buffer(b)), n) for b in bufs]
ts = (C.c_double * 2)()
ms = []
for r in range(reps):
    t = g.v8_a8_expand(L, *hs)
    assert t > 0, t
    ms.append(g.v8_wait2(t, ts))
print("expand ms per layer: %s (median %.2f, %.0f GB/s written)" % (" ".join("%.2f" % x for x in ms), np.median(ms), 3 * n / np.median(ms) / 1e6))
f = safe_open(EX + "layer_%02d.safetensors" % L, "np")
for mi, (m, b) in enumerate(zip(("gate", "up", "down"), bufs)):
    idx = f.get_tensor(m + ".idx")
    codes = np.stack([pal8[L, mi][:, d][idx] for d in range(2)], 1).reshape(-1, idx.shape[1])   # [out rows][in]
    got = np.frombuffer(b, np.int8)
    if m == "down":   # [17][1024][5120]: (x, o)
        want = codes.T.reshape(17, 1024, 5120)
    else:             # [17][5120][1024]: (o block, c, o % 1024)
        want = codes.reshape(17, 1024, 5120).transpose(0, 2, 1)
    eq = np.array_equal(got.reshape(want.shape), want)
    print("%s: %s (mismatches %d of %d)" % (m, "MATCH" if eq else "DIFFER", int((got.reshape(want.shape) != want).sum()), want.size), flush=True)
