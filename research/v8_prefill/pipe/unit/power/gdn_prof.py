"""DeltaNet (GDN) mixer on the served V8 GPU engine, alone (19 Nov 2026): one GDN layer (one-layer manifest), the served
environment (FUSE=all I8 I8T I8A LRM, CONC default on, NT=128), random activations, seeded. Per call (128 tokens, one
pair): GPU ms (median), and with PROFILE=1 the per-kernel GPU ms (each dispatch in its own command buffer).
The DeltaNet cost does not depend on the position (fixed-size state), so p0 only labels the call.
    MANIFEST=... MainPY gdn_prof.py OUT [CALLS]    (env: DYLIB, REC / REC16 / REC8, PROFILE)"""
import ctypes as C, json, mmap, os, sys
from pathlib import Path
import numpy as np

P = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe"); E = P / "engine"
out = sys.argv[1]; calls = int(sys.argv[2]) if len(sys.argv) > 2 else 40
os.environ.update({"V8_MANIFEST": os.environ["MANIFEST"], "V8_KERNELS": str(E / "kernels.metal"), "NT": "128",
                   "LMAX": "98304", "KV8": "1", "K8V16": "1", "QK8": "1", "TKV": "2048", "FUSE": "all", "I8": "1", "I8T": "1",
                   "I8TV": os.environ.get("I8TV", "_k256"), "LUTC": "1", "I8A": "1", "I8KV": "1", "LRM": "1", "F8": os.environ.get("F8", "x")})
dyl = os.environ.get("DYLIB", "libgpu_v8_f8.dylib")
g = C.CDLL(str(E / dyl))
g.v8_init.restype = C.c_int32
g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
g.v8_wait2.restype, g.v8_wait2.argtypes = C.c_double, [C.c_int32, C.POINTER(C.c_double)]
g.v8_submit_pair.restype, g.v8_submit_pair.argtypes = C.c_int32, [C.c_int32] * 6
g.v8_prof_dump.restype = None
nl = g.v8_init()
_b = C.create_string_buffer(4096); g.v8_variant.restype, g.v8_variant.argtypes = C.c_int32, [C.c_char_p, C.c_int32]
g.v8_variant(_b, 4096)
rec = "gdn_rec16" if os.environ.get("REC16") == "1" else "gdn_rec8" if os.environ.get("REC8") == "1" else ("gdn_rec_" + os.environ["REC"]) if os.environ.get("REC") else "gdn_rec"
print(f"BANNER gdn | layers {nl} | recurrence {rec} | lut {os.environ['I8TV'] or '(k64)'} | CONC {os.environ.get('CONC', 'default(on)')} | dylib {dyl} | "
      + _b.value.decode().split(' | kernels=')[0], flush=True)
L = json.loads(Path(os.environ["MANIFEST"]).read_text())["layers"][0]["i"]
rng = np.random.default_rng(0); keep = []


def wrap(nbytes):
    m = mmap.mmap(-1, nbytes); keep.append(m)
    return m, g.v8_wrap(C.addressof(C.c_char.from_buffer(m)), nbytes)


blk = [wrap(5120 * 64 * 2) for _ in range(4)]
for m, _ in blk[:2]:
    np.frombuffer(m, np.float16)[:] = (rng.standard_normal(5120 * 64) * 0.5).astype(np.float16)
ts = (C.c_double * 2)()
ms = []
for i in range(calls + 4):
    t = g.v8_submit_pair(L, blk[0][1], blk[1][1], blk[2][1], blk[3][1], 128 * i)
    assert t > 0, t
    ms.append(g.v8_wait2(t, ts))
y = np.concatenate([np.frombuffer(blk[2][0], np.float16), np.frombuffer(blk[3][0], np.float16)]).astype(np.float32)
r = {"layer": L, "rec": rec, "calls": calls, "ms_median": round(float(np.median(ms[4:])), 4), "ms_min": round(float(np.min(ms[4:])), 4),
     "ms_p90": round(float(np.percentile(ms[4:], 90)), 4), "finite": bool(np.isfinite(y).all())}
print(json.dumps(r), flush=True)
np.save(out + "_y.npy", y)
Path(out + ".json").write_text(json.dumps(r, indent=1))
if os.environ.get("PROFILE"):
    g.v8_prof_dump()
