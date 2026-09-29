"""Which matmul pattern fails to load on the ANE (error -14)? Tiny graphs, no weights.
    step1   rhs - N @ rhs                    (1 use of N)
    chain2  x1 = rhs - N@rhs; x2 = rhs - N@x1
    chain3  one more step
    nn      N @ N
    nn_t    N @ transpose(N) (different second operand)
    state   chain3 with a state written from the result
"""
import shutil
import sys
from pathlib import Path

import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

OUT = Path(__file__).parent / "qwen38_prefill" / "probe"
NV, T, D = 48, int(sys.argv[1]) if len(sys.argv) > 1 else 32, 256


def build(var):
    specs = {"n": mb.TensorSpec((NV, T, T), types.fp16), "rhs": mb.TensorSpec((NV, T, D), types.fp16)}

    @mb.program(input_specs=list(specs.values()), opset_version=ct.target.iOS18)
    def prog(n, rhs):
        if var == "step1":
            return mb.sub(x=rhs, y=mb.matmul(x=n, y=rhs))
        if var.startswith("chain"):
            xs = rhs
            for _ in range(int(var[-1])):
                xs = mb.sub(x=rhs, y=mb.matmul(x=n, y=xs))
            return xs
        if var == "nn":
            return mb.matmul(x=mb.matmul(x=n, y=n), y=rhs)
        if var == "nn_t":
            return mb.matmul(x=mb.matmul(x=n, y=n, transpose_y=True), y=rhs)
        raise ValueError(var)
    return prog


for var in sys.argv[2:] or ["step1", "chain2", "chain3", "nn", "nn_t"]:
    pkg = OUT / f"{var}_T{T}.mlpackage"
    shutil.rmtree(pkg, ignore_errors=True)
    m = ct.convert(build(var), minimum_deployment_target=ct.target.iOS18, compute_units=ct.ComputeUnit.CPU_AND_NE,
                   skip_model_load=True)
    m.save(str(pkg))
    mlc = ct.models.utils.compile_model(str(pkg))
    rng = np.random.default_rng(0)
    inp = {"n": np.tril(rng.standard_normal((NV, T, T)) * 0.1, -1).astype(np.float16),
           "rhs": rng.standard_normal((NV, T, D)).astype(np.float16)}
    try:
        cm = ct.models.CompiledMLModel(mlc, compute_units=ct.ComputeUnit.CPU_AND_NE)
        cm.predict(inp)
        print(f"{var:7s} T={T}: OK", flush=True)
    except Exception as e:
        print(f"{var:7s} T={T}: FAIL {str(e)[-60:]!r}", flush=True)
