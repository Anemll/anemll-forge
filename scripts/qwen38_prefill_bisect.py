"""Which part of the prefill graph fails to load on the ANE? One layer, variants of the DeltaNet recurrence:
    full      chunked_delta as is (outputs from read-derived intermediate states)
    norec     no recurrence: outputs = v, rec state written with read * 1
    retout    state math as is, but outputs computed from the update's returned value (wrong math, rule probe)
"""
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("EXPORT_DIR", str(Path("~/Models/vq27b/export/full_mix25_mixer4_head4").expanduser()))
import coremltools as ct  # noqa: E402
from coremltools.converters.mil import Builder as mb  # noqa: E402
import qwen38_ane_chunk as C  # noqa: E402
import qwen38_ane_model as M  # noqa: E402

LAYERS = [int(x) for x in os.environ.get("LAYERS", "60").split(",")]
T = int(os.environ.get("T", "32"))
orig = C.chunked_delta


def norec(qh, kh, vh, beta, g, s, rec_st, nv, T, dk, dv, scr_st=None):
    mb.coreml_update_state(state=rec_st, value=mb.mul(x=s, y=np.float16(1)))
    return [vh]


def retout(qh, kh, vh, beta, g, s, rec_st, nv, T, dk, dv, scr_st=None):
    real = mb.coreml_update_state
    box = {}

    def capture(state, value):
        box["ret"] = real(state=state, value=value)
        return box["ret"]
    C.mb.coreml_update_state = capture
    try:
        orig(qh, kh, vh, beta, g, s, rec_st, nv, T, dk, dv)
    finally:
        C.mb.coreml_update_state = real
    return [mb.matmul(x=qh, y=box["ret"])]


cfg, ck = M.cfg(), M.Checkpoint()
C.CTX, C.REPEAT, C.LAYERS = int(os.environ.get("CTX", "2048")), 1, LAYERS
weights = {i: ck.layer(i) for i in LAYERS}
quant = {i: M.layer_quant(ck, i, weights[i]) for i in LAYERS}
rng = np.random.default_rng(0)
inp = {"x": (rng.standard_normal((1, cfg["hidden_size"], 1, T)) * 2).astype(np.float16), **C.prefill_inputs(0, T, cfg)}
for var in os.environ.get("VARIANTS", "full,norec,retout").split(","):
    C.chunked_delta = {"full": orig, "norec": norec, "retout": retout}[var]
    C.OUT = Path(__file__).parent / "qwen38_prefill" / f"bisect_{var}"
    mlc = C.build(cfg, weights, quant, T=T, compile_model=True)
    for units in ("CPU_ONLY", "CPU_AND_NE"):
        t0 = time.time()
        try:
            m = ct.models.CompiledMLModel(str(mlc), compute_units=getattr(ct.ComputeUnit, units))
            y = m.predict(inp, state=m.make_state())["y"]
            print(f"{var:7s} {units:10s}: OK {time.time() - t0:.1f}s finite={np.isfinite(y).all()}", flush=True)
        except Exception as e:
            print(f"{var:7s} {units:10s}: FAIL {time.time() - t0:.1f}s {str(e)[-90:]!r}", flush=True)
