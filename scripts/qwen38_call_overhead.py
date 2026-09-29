"""Core ML per-call overhead on the ANE with zero-copy SharedArray I/O (ct.models.SharedArray, output_backings):
a 1x1 conv + add (a bare elementwise graph runs on the CPU), optionally with a large KV-like input (4, S, 256) of which
only 8 values are read (the per-call cost of mapping a big buffer), timed alone and as a chain of 16 dependent calls.
Twin: coreai/probes/coreai_call_overhead.py (Core AI).
    python qwen38_call_overhead.py"""
import shutil
import time
from pathlib import Path

import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.models.compute_plan import MLComputePlan

OUT = Path(__file__).parent / "qwen38_prefill" / "overhead"
SA = ct.models.SharedArray
CASES = [((1, 64, 1, 8), 0), ((1, 512, 1, 8), 0), ((1, 512, 1, 8), 8192), ((1, 512, 1, 8), 65536)]


def build(shape, kv):
    c = shape[1]
    w = (np.random.default_rng(0).standard_normal((c, c, 1, 1)) * c ** -0.5).astype(np.float16)
    if kv:
        @mb.program(input_specs=[mb.TensorSpec(shape, types.fp16), mb.TensorSpec((4, kv, 256), types.fp16)],
                    opset_version=ct.target.iOS18)
        def prog(x, kvin):
            y = mb.add(x=mb.conv(x=x, weight=w),
                       y=mb.reshape(x=mb.slice_by_index(x=kvin, begin=[0, 0, 0], end=[1, 1, 8]), shape=(1, 1, 1, 8)))
            return mb.add(x=y, y=np.float16(1.0), name="y")
    else:
        @mb.program(input_specs=[mb.TensorSpec(shape, types.fp16)], opset_version=ct.target.iOS18)
        def prog(x):
            return mb.add(x=mb.conv(x=x, weight=w), y=np.float16(1.0), name="y")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "x".join(map(str, shape)) + (f"_kv{kv}" if kv else "")
    pkg, mlc = OUT / f"c_{tag}.mlpackage", OUT / f"c_{tag}.mlmodelc"
    for p in (pkg, mlc):
        shutil.rmtree(p, ignore_errors=True)
    ct.convert(prog, minimum_deployment_target=ct.target.iOS18, skip_model_load=True).save(str(pkg))
    ct.models.utils.compile_model(str(pkg), str(mlc))
    plan = MLComputePlan.load_from_path(path=str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)
    ops = plan.model_structure.program.functions["main"].block.operations
    devs = sorted({type(u.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
                   for u in (plan.get_compute_device_usage_for_mlprogram_operation(o) for o in ops) if u is not None})
    return ct.models.CompiledMLModel(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE), devs


def main():
    for shape, kv in CASES:
        m, devs = build(shape, kv)
        a, b = SA(shape), SA(shape)
        a.write(np.ones(shape, np.float16))
        kvbuf = SA((4, kv, 256)) if kv else None

        def feed(xx):
            return {"x": xx, **({"kvin": kvbuf} if kv else {})}
        for _ in range(20):
            m.predict(feed(a), output_backings={"y": b})
        ts = []
        for _ in range(300):
            t = time.perf_counter()
            m.predict(feed(a), output_backings={"y": b})
            ts.append(1e3 * (time.perf_counter() - t))
        chain = []
        for _ in range(50):
            t = time.perf_counter()
            x, y = a, b
            for _ in range(16):
                m.predict(feed(x), output_backings={"y": y})
                x, y = y, x
            chain.append(1e3 * (time.perf_counter() - t))
        mb_io = (np.prod(shape) + kv * 4 * 256) * 2 / 2 ** 20
        print(f"{str(shape) + (f' + kv {kv}' if kv else ''):26s} ({mb_io:7.2f} MB in) on {devs}: single call median "
              f"{np.median(ts):.3f} ms (p10 {np.percentile(ts, 10):.3f}, p90 {np.percentile(ts, 90):.3f}); chain of 16: "
              f"{np.median(chain):.2f} ms = {np.median(chain) / 16:.3f} ms/call", flush=True)


if __name__ == "__main__":
    main()
