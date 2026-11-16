"""A8 attention expander check (GPU only): v8_a8_att_expand of q / o into host buffers vs host codes (att_pal8[idx]).
    MainPY a8_att_check.py LAYER [REPS]"""
import ctypes as C, mmap, os, sys
from pathlib import Path
import numpy as np
from safetensors import safe_open
P = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe"); E = P / "engine"
EX = "/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat/"
L, reps = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 5
os.environ.update({"V8_MANIFEST": str(P / "w_a8/manifest_a8.json"), "V8_KERNELS": str(E / "kernels.metal"), "NT": "128", "LMAX": "256"})
g = C.CDLL(str(E / "libgpu_v8_a8.dylib"))
g.v8_init.restype = C.c_int32
g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
g.v8_wait2.restype, g.v8_wait2.argtypes = C.c_double, [C.c_int32, C.POINTER(C.c_double)]
g.v8_a8_att_setup.restype, g.v8_a8_att_setup.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
g.v8_a8_att_expand.restype, g.v8_a8_att_expand.argtypes = C.c_int32, [C.c_int32] * 3
print("gpu layers", g.v8_init())
pal = np.fromfile(P / "w_a8/att_pal8.bin", np.int8)
print("att setup", g.v8_a8_att_setup(pal.ctypes.data, pal.nbytes))
pal = pal.reshape(64, 2, 64, 2)
bq, bo = mmap.mmap(-1, 12 * 5120 * 1024), mmap.mmap(-1, 6 * 1024 * 5120)
hq, ho = (g.v8_wrap(C.addressof(C.c_char.from_buffer(b)), len(b)) for b in (bq, bo))
ts = (C.c_double * 2)(); ms = []
for r in range(reps):
    t = g.v8_a8_att_expand(L, hq, ho); assert t > 0, t; ms.append(g.v8_wait2(t, ts))
print("att expand ms: %s (median %.2f)" % (" ".join("%.2f" % x for x in ms), np.median(ms)))
f = safe_open(EX + "layer_%02d_mixer.safetensors" % L, "np")
for mi, (n, b) in enumerate((("q", bq), ("o", bo))):
    idx = f.get_tensor("self_attn.%s_proj.idx" % n)
    cd = f.get_slice("self_attn.%s_proj.lut" % n).get_shape()[1]
    codes = np.stack([pal[L, mi][:, d][idx] for d in range(cd)], 1).reshape(-1, idx.shape[1])
    want = codes.reshape(12, 1024, 5120).transpose(0, 2, 1) if n == "q" else codes.T.reshape(6, 1024, 5120)
    got = np.frombuffer(b, np.int8).reshape(want.shape)
    print("%s (cd %d): %s (mismatches %d)" % (n, cd, "MATCH" if np.array_equal(got, want) else "DIFFER", int((got != want).sum())), flush=True)
