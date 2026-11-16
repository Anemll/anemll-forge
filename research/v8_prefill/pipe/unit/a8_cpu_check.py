"""A8 CPU expander check: liba8cpu a8_cpu_expand of one layer (HWX mmap, prep_a8 tables sorted by row) vs the host's
codes from the export; ms per layer.    MainPY a8_cpu_check.py LAYER [REPS]"""
import ctypes as C, json, sys, time
from pathlib import Path
import numpy as np
from safetensors import safe_open
P = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe")
EX = "/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat/"
L, reps = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 5
lib = C.CDLL(str(P / "engine/liba8cpu.dylib"))
lib.a8_cpu_expand.restype = C.c_int
lib.a8_cpu_expand.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
man = json.loads((P / "w_a8/manifest_a8.json").read_text())
Ly = man["layers"][L]
pal8 = np.fromfile(P / "w_a8/pal8.bin", np.int8).reshape(64, 3, 64, 2)
hw = {}
jobs = []
for mi, m in enumerate(("gate", "up", "down")):
    for j in range(8):
        mt = Ly["mats"].get("%s_c%d" % (m, j))
        if mt is None: continue
        h = man["hwx"][mt["hwx"]]
        if mt["hwx"] not in hw:
            hw[mt["hwx"]] = np.memmap(h["path"], np.uint8, "r", offset=h["off"], shape=(h["size"],))
        tab = np.fromfile(mt["tab"], np.uint32).reshape(-1, 3)
        tab = np.ascontiguousarray(tab[np.argsort(tab[:, 2])])
        jobs.append((mi, m, mt, hw[mt["hwx"]], tab))
dst = [np.zeros(17 * 5120 * 1024, np.int8) for _ in range(3)]
ms = []
for r in range(reps):
    t = time.perf_counter()
    for mi, m, mt, K, tab in jobs:
        rc = lib.a8_cpu_expand(K.ctypes.data, tab.ctypes.data, tab.shape[0], mt["cin"], 1 if m == "down" else 0, mt["xoff"] or 0,
                               np.ascontiguousarray(pal8[L, mi]).ctypes.data, dst[mi].ctypes.data)
        assert rc == 0, rc
    ms.append((time.perf_counter() - t) * 1e3)
print("cpu expand ms per layer: %s (median %.2f, %.0f GB/s written)" % (" ".join("%.1f" % x for x in ms), np.median(ms), 3 * dst[0].size / np.median(ms) / 1e6))
f = safe_open(EX + "layer_%02d.safetensors" % L, "np")
for mi, m in enumerate(("gate", "up", "down")):
    idx = f.get_tensor(m + ".idx")
    codes = np.stack([pal8[L, mi][:, d][idx] for d in range(2)], 1).reshape(-1, idx.shape[1])
    want = codes.T.reshape(17, 1024, 5120) if m == "down" else codes.reshape(17, 1024, 5120).transpose(0, 2, 1)
    got = dst[mi].reshape(want.shape)
    print("%s: %s (mismatches %d)" % (m, "MATCH" if np.array_equal(got, want) else "DIFFER", int((got != want).sum())), flush=True)
