"""--attproj plumbing check on one shared2 chunk (layers 4c..4c+3, attention layer 4c+3), pair mode (two 64-row blocks):
  current: ANE ffn2_L(La-1) -> GPU v8_submit_pair(La) -> ANE ffn2_L(La)
  fused:   ANE ffnq2_L(La-1) -> GPU v8_submit_attcore(La) -> ANE offn2_L(La)
same random inputs, NP consecutive pairs (KV grows); outputs compared per pair, times per stage. Journaled.
    MainPY attcore_check.py PKG LA [NP]"""
import ctypes as C_, hashlib, os, sys, time
from pathlib import Path
import numpy as np
for p in ("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02/coreai/swift_bridge",
          "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02/scripts",
          "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"):
    sys.path.insert(0, p)
import ane_compile_mode as SOC
import coreai_bridge as B
import deploy_journal as DJ
SOC.apply(strict=True)
path, La = Path(sys.argv[1]), int(sys.argv[2]); NP = int(sys.argv[3]) if len(sys.argv) > 3 else 8
Lp = La - 1
E = "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/engine"
os.environ.update({"V8_MANIFEST": "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared/manifest.json",
                   "V8_KERNELS": E + "/kernels.metal", "NT": "128", "LMAX": str(128 * NP + 64)})
g = C_.CDLL(E + "/libgpu_v8_attcore.dylib")
g.v8_init.restype = C_.c_int32; g.v8_wrap.restype, g.v8_wrap.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t]
g.v8_submit_pair.restype, g.v8_submit_pair.argtypes = C_.c_int32, [C_.c_int32] * 6
g.v8_submit_attcore.restype, g.v8_submit_attcore.argtypes = C_.c_int32, [C_.c_int32, C_.POINTER(C_.c_int32), C_.c_int32]
g.v8_wait.restype, g.v8_wait.argtypes = C_.c_double, [C_.c_int32]
g.v8_init()
FN = [f"ffn2_L{Lp}", f"ffn2_L{La}", f"ffnq2_L{Lp}", f"offn2_L{La}"]
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                "out": "attcore check", "mode": "v8_attcore_check"}, f"attc-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
pk = {"path": str(path), "label": "V8 shared2 chunk (attproj check)", "files": [path.name], "main_hash": (path / "main.hash").read_bytes().hex()}
aid = J.attempt(dev, pk, functions=FN, request={"compute": "ane"})
t0 = time.time()
try:
    m = B.Model(path, "ane")
except Exception as e:  # noqa: BLE001
    J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
fp, fa, fq, fo = (m.function(n) for n in FN)
# current path buffers: x0 (in) -> y0 (ffn2 Lp out = GPU in) -> x1 (GPU out) -> y1 (ffn2 La out)
x0 = {n: fp.buffer("input", n) for n in ("xa", "xb")}; y0 = {n: fp.buffer("output", n) for n in ("ya", "yb")}
x1 = {n: fa.buffer("input", n) for n in ("xa", "xb")}; y1 = {n: fa.buffer("output", n) for n in ("ya", "yb")}
pA = B.Plan([fp.bind(x0, y0)]); pB = B.Plan([fa.bind(x1, y1)])
w = lambda b: g.v8_wrap(b.address, b.nbytes)
hy0 = [w(y0["ya"]), w(y0["yb"])]; hx1 = [w(x1["xa"]), w(x1["xb"])]
# fused path buffers
qo = {n: fq.buffer("output", n) for n in ("ya", "qa", "ka", "va", "yb", "qb", "kb", "vb")}
oi = {n: fo.buffer("input", n) for n in ("oa", "ob")}; y2 = {n: fo.buffer("output", n) for n in ("ya", "yb")}
qA = B.Plan([fq.bind(x0, qo)]); qB = B.Plan([fo.bind({"oa": oi["oa"], "ob": oi["ob"], "xa": qo["ya"], "xb": qo["yb"]}, y2)])
hs = (C_.c_int32 * 8)(*[w(qo[n]) for n in ("qa", "qb", "ka", "kb", "va", "vb")], w(oi["oa"]), w(oi["ob"]))
assert min(list(hs) + hy0 + hx1) >= 0
rng = np.random.default_rng(3)
X = [(rng.standard_normal((2, 5120, 64)) * 0.5).astype(np.float16) for _ in range(NP)]
def run(fused):
    g.v8_reset(); outs, ts = [], []
    for k in range(NP):
        x0["xa"].np[0, :, 0, :] = X[k][0]; x0["xb"].np[0, :, 0, :] = X[k][1]
        t1 = time.perf_counter()
        (qA if fused else pA).run(); t2 = time.perf_counter()
        t = g.v8_submit_attcore(La, hs, k * 128) if fused else g.v8_submit_pair(La, hy0[0], hy0[1], hx1[0], hx1[1], k * 128)
        assert t > 0, t
        assert g.v8_wait(t) >= 0; t3 = time.perf_counter()
        (qB if fused else pB).run(); t4 = time.perf_counter()
        yy = y2 if fused else y1
        outs.append(np.stack([np.array(yy["ya"].np[0, :, 0, :]), np.array(yy["yb"].np[0, :, 0, :])]).astype(np.float32))
        ts.append((t2 - t1, t3 - t2, t4 - t3))
    return outs, np.array(ts) * 1e3
for rnd in range(2):
    oc, tc = run(False); of, tf = run(True)
print("pair  rel diff fused vs current (y after layer La)")
for k in range(NP):
    print(f"{k:4d}  {float(np.abs(of[k] - oc[k]).mean() / np.abs(oc[k]).mean()):.4f}")
md = lambda t: " / ".join(f"{float(np.median(t[1:, j])):.2f}" for j in range(3))
print(f"median ms ANE(La-1) / GPU(La) / ANE(La): current {md(tc)}   fused {md(tf)}")
