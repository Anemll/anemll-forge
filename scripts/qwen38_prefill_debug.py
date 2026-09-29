"""Build only the prefill function (single-function model) and try loading it on each compute unit."""
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("EXPORT_DIR", str(Path("~/Models/vq27b/export/full_mix25_mixer4_head4").expanduser()))
import coremltools as ct  # noqa: E402
import qwen38_ane_chunk as C  # noqa: E402
import qwen38_ane_model as M  # noqa: E402

LAYERS = [int(x) for x in os.environ.get("LAYERS", "60,61,62,63").split(",")]
T = int(os.environ.get("T", "32"))

cfg, ck = M.cfg(), M.Checkpoint()
C.CTX, C.REPEAT, C.LAYERS, C.OUT = int(os.environ.get("CTX", "2048")), 1, LAYERS, Path(__file__).parent / "qwen38_prefill" / "dbg"
weights = {i: ck.layer(i) for i in LAYERS}
quant = {i: M.layer_quant(ck, i, weights[i]) for i in LAYERS}
mlc = C.build(cfg, weights, quant, T=T, compile_model=True)
print("compiled", mlc, flush=True)
inp = {"x": (np.random.default_rng(0).standard_normal((1, cfg["hidden_size"], 1, T)) * 2).astype(np.float16), **C.prefill_inputs(0, T, cfg)}
for units in os.environ.get("UNITS", "CPU_ONLY,CPU_AND_NE").split(","):
    t0 = time.time()
    try:
        m = ct.models.CompiledMLModel(str(mlc), compute_units=getattr(ct.ComputeUnit, units))
        y = m.predict(inp, state=m.make_state())["y"]
        print(f"{units}: OK load+predict {time.time() - t0:.1f}s, y finite {np.isfinite(y).all()}", flush=True)
    except Exception as e:
        print(f"{units}: FAIL after {time.time() - t0:.1f}s: {str(e)[-260:]!r}", flush=True)
