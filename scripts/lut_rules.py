"""Vector-LUT rule checks on the ANE via MIL (iOS18 constexpr_lut_to_dense).
Each case: one or two layers; report MLComputePlan placement, cosine of CPU_AND_NE output
vs numpy reference, and writes the .mlpackage for a direct ane_mil_bench compile."""
import sys
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

OUT = Path(__file__).parent / "lut_rules"
OUT.mkdir(exist_ok=True)
IDX = {1: types.np_uint1_dtype, 2: types.np_uint2_dtype, 3: types.np_uint3_dtype,
       4: types.np_uint4_dtype, 6: types.np_uint6_dtype, 8: np.uint8}
C, HW = 512, 32


def make_lut_weight(shape, cd, nb, axis, groups, rng, lut_int8=False):
    """Weight of `shape` from `groups` LUTs along axis 0 (Cout); vectors of cd along `axis`."""
    nl = 1 << nb
    luts = rng.standard_normal((groups, nl, cd)) * (np.prod(shape[1:]) ** -0.5)
    idx_shape = list(shape)
    idx_shape[axis] //= cd
    idx = rng.integers(0, nl, size=idx_shape)
    scale = None
    if lut_int8:
        scale = np.abs(luts).max() / 127
        luts = np.round(luts / scale).clip(-127, 127)
    # dense weight
    w = np.zeros(shape)
    g_of_out = np.arange(shape[0]) // (shape[0] // groups)
    moved = np.moveaxis(idx, axis, -1)                      # [..., n_vec]
    vals = luts[None]  # placeholder
    idx_m = np.moveaxis(idx, axis, -1)
    out_m = np.zeros(list(idx_m.shape[:-1]) + [idx_m.shape[-1] * cd])
    # output-channel index for each position of idx_m (axis 0 unless axis==0)
    it = np.nditer(idx_m, flags=["multi_index"])
    grp = np.empty(idx_m.shape, dtype=int)
    if axis == 0:
        vec_pos = np.arange(idx_m.shape[-1])
        grp[...] = (vec_pos * cd // (shape[0] // groups))
    else:
        o = np.arange(shape[0]).reshape([-1] + [1] * (idx_m.ndim - 1))
        grp[...] = np.broadcast_to(o // (shape[0] // groups), idx_m.shape)
    gathered = luts[grp, idx_m]                                # [..., n_vec, cd]
    out_m = gathered.reshape(*idx_m.shape[:-1], idx_m.shape[-1] * cd)
    w = np.moveaxis(out_m, -1, axis)
    lut_shape = [1] * len(shape)
    lut_shape[0] = groups
    lut = luts.reshape(*lut_shape, nl, cd)
    return w, idx.astype(IDX[nb]), lut, scale


CASES = {
    # name: dict(cd, nb, axis, groups, k, stride, dil, op, lut_int8)
    "base_v4n4":        dict(cd=4, nb=4),
    "base_v2n4":        dict(cd=2, nb=4),
    "scalar_n4":        dict(cd=1, nb=4),
    "grp16_v4n4":       dict(cd=4, nb=4, groups=C // 16),
    "grp64_v4n6":       dict(cd=4, nb=6, groups=C // 64),
    "grp4_v4n4":        dict(cd=4, nb=4, groups=C // 4),
    "grp16_s4":         dict(cd=1, nb=4, groups=C // 16),
    "grp2x256_v4n4":    dict(cd=4, nb=4, groups=2),
    "grp4x128_v4n4":    dict(cd=4, nb=4, groups=4),
    "grp2x256_s4":      dict(cd=1, nb=4, groups=2),
    "int8lut_v4n4":     dict(cd=4, nb=4, lut_int8=True),
    "int8lut_v2n4":     dict(cd=2, nb=4, lut_int8=True),
    "int8lut_s4":       dict(cd=1, nb=4, lut_int8=True),
    "int8lut_v2n8":     dict(cd=2, nb=8, lut_int8=True),   # 512 values = 512 bytes int8
    "int8lut_v4n8":     dict(cd=4, nb=8, lut_int8=True),   # 1024 values
    "fp16lut_v2n8":     dict(cd=2, nb=8),                  # 512 values = 1 KB fp16
    "int8lut_v4n6":     dict(cd=4, nb=6, lut_int8=True),   # 256 values
    "cin_axis_v4n4":    dict(cd=4, nb=4, axis=1),
    "k3_v4n4":          dict(cd=4, nb=4, k=3),
    "k3_s2_v4n4":       dict(cd=4, nb=4, k=3, stride=2),
    "k3_d2_v4n4":       dict(cd=4, nb=4, k=3, dil=2),
    "k3_s2_scalar":     dict(cd=1, nb=4, k=3, stride=2),
    "linear_v4n4":      dict(cd=4, nb=4, op="linear"),
    "linear_scalar":    dict(cd=1, nb=4, op="linear"),
    "matmul_v4n4":      dict(cd=4, nb=4, op="matmul"),
}


def build(name, p):
    rng = np.random.default_rng(0)
    cd, nb = p["cd"], p["nb"]
    axis, groups = p.get("axis", 0), p.get("groups", 1)
    k, stride, dil, op = p.get("k", 1), p.get("stride", 1), p.get("dil", 1), p.get("op", "conv")
    lut_int8 = p.get("lut_int8", False)
    if op == "conv":
        shape = (C, C, k, k)
        xshape = (1, C, HW, HW)
    else:
        shape = (C, C)            # linear weight [Cout, Cin]; matmul uses W^T
        xshape = (HW * HW, C)
    w, idx, lut, scale = make_lut_weight(shape, cd, nb, axis, groups, rng, lut_int8)
    wd = w * (scale if scale is not None else 1.0)

    @mb.program(input_specs=[mb.TensorSpec(shape=xshape, dtype=types.fp16)], opset_version=ct.target.iOS18)
    def prog(x):
        if lut_int8:
            # joint pattern: int8 LUT -> blockwise_shift_scale -> lut_to_dense
            sc = np.full([1] * lut.ndim, scale, np.float16)
            lut_c = mb.constexpr_blockwise_shift_scale(data=lut.astype(np.int8), scale=sc)
        else:
            lut_c = lut.astype(np.float16)
        wt = mb.constexpr_lut_to_dense(indices=idx, lut=lut_c, vector_axis=axis if cd > 1 else None)
        if op == "conv":
            pad = dil * (k - 1) // 2
            y = mb.conv(x=x, weight=wt, strides=[stride, stride], dilations=[dil, dil],
                        pad_type="custom", pad=[pad] * 4)
        elif op == "linear":
            y = mb.linear(x=x, weight=wt)
        else:
            y = mb.matmul(x=x, y=wt, transpose_y=True)
        return y

    path = OUT / f"{name}.mlpackage"
    m = ct.convert(prog, minimum_deployment_target=ct.target.iOS18, compute_units=ct.ComputeUnit.CPU_AND_NE)
    m.save(str(path))
    return path, wd, xshape, (k, stride, dil, op)


def reference(x, wd, k, stride, dil, op):
    import torch
    xt = torch.from_numpy(x.astype(np.float32))
    wt = torch.from_numpy(wd.astype(np.float32))
    if op == "conv":
        pad = dil * (k - 1) // 2
        return torch.nn.functional.conv2d(xt, wt, stride=stride, padding=pad, dilation=dil).numpy()
    return (xt @ wt.T).numpy()


def placement(model):
    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(model.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    fn = plan.model_structure.program.functions["main"]
    res = []
    for op in fn.block.operations:
        if any(t in op.operator_name for t in ("conv", "linear", "matmul")):
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            res.append(f"{op.operator_name.split('.')[-1]}:"
                       + ("?" if u is None else type(u.preferred_compute_device).__name__
                          .replace("ML", "").replace("ComputeDevice", "")))
    return ",".join(res)


def main():
    names = sys.argv[1:] or list(CASES)
    for name in names:
        try:
            path, wd, xshape, (k, s, d, op) = build(name, CASES[name])
            m = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
            x = np.random.default_rng(1).standard_normal(xshape).astype(np.float16)
            y = list(m.predict({"x": x}).values())[0].astype(np.float32).ravel()
            ref = reference(x, wd, k, s, d, op).ravel()
            cos = float(y @ ref / (np.linalg.norm(y) * np.linalg.norm(ref) + 1e-30))
            print(f"{name:16s} {placement(m):22s} cos={cos:.5f}", flush=True)
        except Exception as e:
            print(f"{name:16s} BUILD/RUN ERROR: {str(e).splitlines()[0][:150]}", flush=True)


if __name__ == "__main__":
    main()
