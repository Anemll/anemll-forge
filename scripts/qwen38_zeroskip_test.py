"""Does the ANE skip work on zero activation blocks in attention? One layer group (read-only KV inputs, T=8 lazy
function) at CTX; 8-row call time at different positions, with the KV cache past the position either zero or
random (masked either way). If time grows with position only for the zero variant, KV data zero-skipping helps;
if it grows for both, the zero probabilities from the mask (exp(-1e4) = 0) are skipped.
    RANGE=60-63 CTX=16384 python qwen38_zeroskip_test.py"""
import os
import shutil
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("EXPORT_DIR", str(Path("~/Models/vq27b/export/full_mix25_mixer4_head4").expanduser()))
import coremltools as ct  # noqa: E402
import qwen38_ane_chunk as C  # noqa: E402
import qwen38_ane_model as M  # noqa: E402
from qwen38_lazy_test import Runner  # noqa: E402
import qwen38_lazy_test as L  # noqa: E402

a_, b_ = (int(x) for x in os.environ.get("RANGE", "60-63").split("-"))
LAYERS = list(range(a_, b_ + 1))
CTX, T = int(os.environ.get("CTX", "16384")), 8
OUT = Path(__file__).parent / "qwen38_prefill" / "lazy"


def main():
    cfg, ck = M.cfg(), M.Checkpoint()
    L.CTX, L.KV_IN, L.HOST_KV = CTX, True, False
    C.CTX, C.REPEAT, C.OUT, C.GDN_IO, C.KV_IO, C.KV_IN, C.LAZY, C.LAYERS, C.JOFF, C.TAPS = \
        CTX, 1, OUT / "tmp", True, False, True, True, LAYERS, 0, []
    mlc = OUT / f"lazy_kvin_L{a_}-{b_}_ctx{CTX}_T{T}.mlmodelc"
    if not mlc.exists():
        weights = {i: ck.layer(i) for i in LAYERS}
        quant = {i: M.layer_quant(ck, i, weights[i]) for i in LAYERS}
        pkg = C.build(cfg, weights, quant, T=T, compile_model=False)
        ct.models.utils.compile_model(str(pkg), str(mlc))
        shutil.rmtree(C.OUT, ignore_errors=True)
    m = ct.models.CompiledMLModel(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)
    gdn = [j for j, l in enumerate(LAYERS) if cfg["layer_types"][l] == "linear_attention"]
    att = [j for j in range(len(LAYERS)) if j not in gdn]
    r = Runner(m, cfg, gdn, att)
    rng = np.random.default_rng(0)
    xs = (rng.standard_normal((T, cfg["hidden_size"])) * 2).astype(np.float16)
    kv_rand = {n: (rng.standard_normal(buf.shape) * 0.5).astype(np.float16) for n, buf in r.kv.items()}
    for fill in ("zero", "random"):
        row = []
        for pos in (64, 4096, 8192, 12288, CTX - 64):
            for n, buf in r.kv.items():  # history rows < pos: random; rows >= pos: zero or random
                a = kv_rand[n].copy()
                if fill == "zero":
                    a[:, pos:] = 0
                buf.write(a)
            r.pos, r.pending = pos, 0
            for _ in range(3):
                r.call(xs)
            ts = []
            for _ in range(25):
                t0 = time.perf_counter()
                r.call(xs)
                ts.append(1e3 * (time.perf_counter() - t0))
            row.append(f"pos {pos:5d}: {np.median(ts):5.2f} ms")
        print(f"KV past position = {fill:6s} | " + " | ".join(row), flush=True)


if __name__ == "__main__":
    main()
