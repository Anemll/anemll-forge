"""Core ML (MIL) twin of bench_vector_lut: iOS18 constexpr_lut_to_dense with vector LUTs.
Conv chain C->C 1x1, (1,C,H,W) fp16, S layers; weights generated from exact LUTs.
Reports ANE placement (MLComputePlan), timing (CPU_AND_NE), cosine vs numpy reference."""
import statistics, sys, time
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

OUT = Path(__file__).parent / "vector_lut_coreml"
OUT.mkdir(exist_ok=True)
VARIANTS = {"dense": None, "s4": (1, 4), "s2": (1, 2), "v2": (2, 8), "v4": (4, 8), "v8": (8, 8), "v4n4": (4, 4)}
def variant(name):
    # "dense", named variants, or "cdXnbY" (cluster_dim X, n_bits Y)
    if name in VARIANTS:
        return VARIANTS[name]
    cd, nb = name[2:].split("nb")
    return int(cd), int(nb)


IDX_DTYPE = {3: types.np_uint3_dtype, 6: types.np_uint6_dtype, 1: types.np_uint1_dtype, 2: types.np_uint2_dtype, 4: types.np_uint4_dtype, 8: np.uint8}


def make_layers(c, s, cfg, seed=42):
    rng = np.random.default_rng(seed)
    cd, nb = cfg or (1, 8)
    layers = []
    for _ in range(s):
        lut = (rng.standard_normal((1 << nb, cd)) * c**-0.5).astype(np.float16)
        idx = rng.integers(0, 1 << nb, size=(c // cd, c))
        w = lut[idx].transpose(0, 2, 1).reshape(c, c)  # w[o*cd+j, i] = lut[idx[o,i], j]
        layers.append((lut, idx, w))
    return layers


def build(name, c, hw, s):
    cfg = variant(name)
    path = OUT / f"{name}_C{c}_HW{hw}_S{s}.mlpackage"
    layers = make_layers(c, s, cfg)
    if path.exists():
        return ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE), layers

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, c, hw, hw), dtype=types.fp16)],
                opset_version=ct.target.iOS18)
    def prog(x):
        for lut, idx, w in layers:
            if cfg is None:
                weight = mb.const(val=w.reshape(c, c, 1, 1))
            else:
                cd, nb = cfg
                weight = mb.constexpr_lut_to_dense(
                    indices=idx.reshape(c // cd, c, 1, 1).astype(IDX_DTYPE[nb]),
                    lut=lut.reshape(1, 1, 1, 1, 1 << nb, cd),
                    vector_axis=0 if cd > 1 else None,
                )
            x = mb.conv(x=x, weight=weight)
        return x

    model = ct.convert(prog, minimum_deployment_target=ct.target.iOS18,
                       compute_units=ct.ComputeUnit.CPU_AND_NE)
    model.save(str(path))
    return ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE), layers


def placement(model):
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.models.compute_device import MLNeuralEngineComputeDevice
    plan = MLComputePlan.load_from_path(model.get_compiled_model_path(),
                                        compute_units=ct.ComputeUnit.CPU_AND_NE)
    fn = plan.model_structure.program.functions["main"]
    counts = {}
    for op in fn.block.operations:
        if op.operator_name not in ("ios18.conv", "ios18.constexpr_lut_to_dense", "ios17.conv"):
            continue
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        dev = "?" if u is None else type(u.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
        key = f"{op.operator_name.split('.')[-1]}:{dev}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    c, hw, s = 2048, 4, 32
    for a in sys.argv[1:]:
        if a.startswith("--c="): c = int(a[4:])
        if a.startswith("--hw="): hw = int(a[5:])
        if a.startswith("--s="): s = int(a[4:])
    names = args or list(VARIANTS)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, c, hw, hw)).astype(np.float16)
    rows = []
    for name in names:
        cfg = variant(name)
        bits = 16 if cfg is None else cfg[1] / cfg[0]
        t0 = time.perf_counter()
        model, layers = build(name, c, hw, s)
        tb = time.perf_counter() - t0
        place = placement(model)
        if "--place-only" in sys.argv:
            print(f"{name:8s} {place}", flush=True)
            continue
        for _ in range(5):
            y = model.predict({"x": x})
        out_name = list(y)[0]
        ts = []
        for _ in range(50):
            t = time.perf_counter(); y = model.predict({"x": x}); ts.append(time.perf_counter() - t)
        med = statistics.median(ts)
        ref = x.astype(np.float32).reshape(c, -1)
        for _, _, w in layers:
            ref = w.astype(np.float32) @ ref
        yf = y[out_name].astype(np.float32).reshape(c, -1)
        cos = float((yf * ref).sum() / (np.linalg.norm(yf) * np.linalg.norm(ref) + 1e-30))
        rows.append((name, bits, med, cos, place))
        print(f"{name:6s} {bits:5g} b/w  build {tb:5.1f}s  {med*1e3:8.3f} ms  cos={cos:.4f}  {place}", flush=True)
    d = next((r[2] for r in rows if r[0] == "dense"), None)
    print(f"\n{'variant':7s} {'bits/w':>6s} {'ms':>8s} {'x dense':>7s} {'cos':>7s}  placement")
    for name, bits, med, cos, place in rows:
        print(f"{name:7s} {bits:6g} {med*1e3:8.3f} {(d/med if d else float('nan')):7.2f} {cos:7.4f}  {place}")


if __name__ == "__main__":
    main()
