"""ANE cost of per-group scalar LUT4 (groups along Cout) vs one per-tensor LUT, at the DeltaNet in_proj_qkv shape
(10240 x 5120), + per-output-channel scale, 16 stacked 1x1 convs on 8 rows (the verify block). Groups of 2048 rows
= separate q / k / v(x3) tables; 1280 groups = anemll's per-group-8.
    python qwen38_group_lut_probe.py"""
import shutil
import time
from pathlib import Path

import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

COUT, CIN, T, NL = 10240, 5120, 8, 16
OUT = Path(__file__).parent / "qwen38_prefill" / "grouplut"
SA = ct.models.SharedArray


def build(groups):
    rng = np.random.default_rng(0)
    luts = [(np.sort(rng.standard_normal((groups, 16)), 1) * 0.02).astype(np.float16) for _ in range(NL)]
    idxs = [rng.integers(0, 16, (COUT, CIN)).astype(np.uint8).astype(types.np_uint4_dtype) for _ in range(NL)]
    scales = [(rng.random(COUT) + 0.5).astype(np.float16) for _ in range(NL)]
    # each layer maps CIN -> CIN through (COUT x CIN) then a cheap fixed slice back to CIN rows
    @mb.program(input_specs=[mb.TensorSpec((1, CIN, 1, T), types.fp16)], opset_version=ct.target.iOS18)
    def prog(x):
        for l in range(NL):
            w = mb.constexpr_lut_to_dense(indices=idxs[l].reshape(COUT, CIN, 1, 1),
                                          lut=luts[l].reshape(groups, 1, 1, 1, 16, 1))
            w = mb.constexpr_blockwise_shift_scale(data=w, scale=scales[l].reshape(-1, 1, 1, 1))
            y = mb.conv(x=x, weight=w)
            x = mb.slice_by_index(x=y, begin=[0, 0, 0, 0], end=[1, CIN, 1, T])
        return mb.identity(x=x, name="y")
    OUT.mkdir(parents=True, exist_ok=True)
    pkg, mlc = OUT / f"g{groups}.mlpackage", OUT / f"g{groups}.mlmodelc"
    for p in (pkg, mlc):
        shutil.rmtree(p, ignore_errors=True)
    ct.convert(prog, minimum_deployment_target=ct.target.iOS18, skip_model_load=True).save(str(pkg))
    ct.models.utils.compile_model(str(pkg), str(mlc))
    shutil.rmtree(pkg, ignore_errors=True)
    return mlc


def main():
    for g in (1, 5, 20, 80, 1280):
        t0 = time.time()
        mlc = build(g)
        m = ct.models.CompiledMLModel(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)
        x, y = SA((1, CIN, 1, T)), SA((1, CIN, 1, T))
        x.write(np.random.default_rng(1).standard_normal((1, CIN, 1, T)).astype(np.float16) * 0.1)
        for _ in range(5):
            m.predict({"x": x}, output_backings={"y": y})
        ts = []
        for _ in range(30):
            t = time.perf_counter()
            m.predict({"x": x}, output_backings={"y": y})
            ts.append(1e3 * (time.perf_counter() - t))
        gb = NL * COUT * CIN / 2 / 1e9
        print(f"groups {g:5d}: {np.median(ts):7.2f} ms for {NL} matrices ({np.median(ts) / NL:.3f} ms each, "
              f"{gb / (np.median(ts) / 1e3):.0f} GB/s), built+loaded {time.time() - t0:.0f}s", flush=True)
        shutil.rmtree(mlc, ignore_errors=True)


if __name__ == "__main__":
    main()
