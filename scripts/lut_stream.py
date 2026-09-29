"""Is a vector LUT decoded on the ANE (compressed in DRAM) or expanded at compile time?
B parallel branches from one small input, each with its own LUT weight, summed. Weight-bandwidth
bound, so time tracks stored bits/weight if decoding is native; dense-like time means expanded.
Writes .mlpackage per case; timing done with ane_mil_bench on the compiled .mlmodelc."""
import sys
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

OUT = Path(__file__).parent / "lut_stream"
OUT.mkdir(exist_ok=True)
IDX = {1: types.np_uint1_dtype, 2: types.np_uint2_dtype, 4: types.np_uint4_dtype, 8: np.uint8}

# name: (op, C, k, stride, dil, branches, fmt)  fmt: dense | (cd, nb) | ("int8", cd, nb)
CASES = {
    "c1_dense":    ("conv", 2048, 1, 1, 1, 32, "dense"),
    "c1_s4":       ("conv", 2048, 1, 1, 1, 32, (1, 4)),
    "c1_v4n4":     ("conv", 2048, 1, 1, 1, 32, (4, 4)),
    "c1_i8v4n4":   ("conv", 2048, 1, 1, 1, 32, ("int8", 4, 4)),
    "k3_dense":    ("conv", 1024, 3, 1, 1, 16, "dense"),
    "k3_s4":       ("conv", 1024, 3, 1, 1, 16, (1, 4)),
    "k3_v4n4":     ("conv", 1024, 3, 1, 1, 16, (4, 4)),
    "k3s2_dense":  ("conv", 1024, 3, 2, 1, 16, "dense"),
    "k3s2_v4n4":   ("conv", 1024, 3, 2, 1, 16, (4, 4)),
    "k3d2_dense":  ("conv", 1024, 3, 1, 2, 16, "dense"),
    "k3d2_v4n4":   ("conv", 1024, 3, 1, 2, 16, (4, 4)),
    "k3d2_s4":     ("conv", 1024, 3, 1, 2, 16, (1, 4)),
    "k3d2_s1":     ("conv", 1024, 3, 1, 2, 16, (1, 1)),
    "lin_dense":   ("linear", 4096, 1, 1, 1, 8, "dense"),
    "lin_s4":      ("linear", 4096, 1, 1, 1, 8, (1, 4)),
    "lin_v4n4":    ("linear", 4096, 1, 1, 1, 8, (4, 4)),
}


def build(name):
    op, c, k, stride, dil, nbr, fmt = CASES[name]
    rng = np.random.default_rng(0)
    xshape = (1, c, 4, 4) if op == "conv" else (4, c)
    wshape = (c, c, k, k) if op == "conv" else (c, c)

    @mb.program(input_specs=[mb.TensorSpec(shape=xshape, dtype=types.fp16)], opset_version=ct.target.iOS18)
    def prog(x):
        acc = None
        for _ in range(nbr):
            if fmt == "dense":
                wt = mb.const(val=(rng.standard_normal(wshape) * (np.prod(wshape[1:]) ** -0.5)).astype(np.float16))
            else:
                int8 = fmt[0] == "int8"
                cd, nb = fmt[-2], fmt[-1]
                lut = rng.standard_normal((1 << nb, cd)) * (np.prod(wshape[1:]) ** -0.5)
                ishape = list(wshape); ishape[0] //= cd
                idx = rng.integers(0, 1 << nb, size=ishape).astype(IDX[nb])
                lshape = [1] * len(wshape) + [1 << nb, cd]
                if int8:
                    sc = np.abs(lut).max() / 127
                    lut_c = mb.constexpr_blockwise_shift_scale(
                        data=np.round(lut / sc).astype(np.int8).reshape(lshape),
                        scale=np.full([1] * len(lshape), sc, np.float16))
                else:
                    lut_c = lut.astype(np.float16).reshape(lshape)
                wt = mb.constexpr_lut_to_dense(indices=idx, lut=lut_c, vector_axis=0 if cd > 1 else None)
            if op == "conv":
                pad = dil * (k - 1) // 2
                y = mb.conv(x=x, weight=wt, strides=[stride] * 2, dilations=[dil] * 2, pad_type="custom", pad=[pad] * 4)
            else:
                y = mb.linear(x=x, weight=wt)
            acc = y if acc is None else mb.add(x=acc, y=y)
        return acc

    path = OUT / f"{name}.mlpackage"
    m = ct.convert(prog, minimum_deployment_target=ct.target.iOS18, compute_units=ct.ComputeUnit.CPU_AND_NE)
    m.save(str(path))
    n_w = nbr * int(np.prod(wshape))
    bits = 16 if fmt == "dense" else fmt[-1] / fmt[-2]
    print(f"{name:12s} weights={n_w/1e6:6.1f}M  bits/w={bits:g}  stored~{n_w*bits/8/1e6:6.1f} MB", flush=True)


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(CASES)):
        build(n)
