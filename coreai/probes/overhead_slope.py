"""Per-call overhead, Core ML vs Core AI, by the slope method on the ANE: a chain of S 1x1 convs (C=1024, fp16, 8 rows)
for S = 1 and 8 in each framework; T(S) = T_call + S * T_conv -> T_call = T(1) - (T(8) - T(1)) / 7. Core ML placement
is checked with MLComputePlan (tiny graphs go to the CPU), Core AI's from the compiled manifest.
Run from the Forge root using its compatible Core ML and Core AI environments:
    .venv/bin/python coreai/probes/overhead_slope.py coreml
    coreai/.venv/bin/python coreai/probes/overhead_slope.py coreai"""
from __future__ import annotations

# Use only the helpers shipped in this repository.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import shutil
import sys
import time
from pathlib import Path

import numpy as np

C_, T_, REPS = 1024, 8, 300
ROOT = Path(__file__).resolve().parent / "artifacts_overhead_slope"


def weights(s):
    r = np.random.default_rng(0)
    return [(r.standard_normal((C_, C_, 1, 1)) * C_ ** -0.5).astype(np.float16) for _ in range(s)]


def stats(ts):
    return float(np.median(ts)), float(np.percentile(ts, 10))


def coreml():
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    from coremltools.models.compute_plan import MLComputePlan
    SA = ct.models.SharedArray
    res = {}
    for s in (1, 8):
        ws = weights(s)

        @mb.program(input_specs=[mb.TensorSpec((1, C_, 1, T_), types.fp16)], opset_version=ct.target.iOS18)
        def prog(x):
            for w in ws:
                x = mb.conv(x=x, weight=w)
            return mb.identity(x=x, name="y")
        ROOT.mkdir(parents=True, exist_ok=True)
        pkg, mlc = ROOT / f"cml_S{s}.mlpackage", ROOT / f"cml_S{s}.mlmodelc"
        for p in (pkg, mlc):
            shutil.rmtree(p, ignore_errors=True)
        ct.convert(prog, minimum_deployment_target=ct.target.iOS18, skip_model_load=True).save(str(pkg))
        ct.models.utils.compile_model(str(pkg), str(mlc))
        plan = MLComputePlan.load_from_path(path=str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)
        ops = plan.model_structure.program.functions["main"].block.operations
        devs = sorted({type(u.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
                       for u in (plan.get_compute_device_usage_for_mlprogram_operation(o) for o in ops) if u is not None})
        m = ct.models.CompiledMLModel(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)
        a, b = SA((1, C_, 1, T_)), SA((1, C_, 1, T_))
        a.write(np.ones((1, C_, 1, T_), np.float16) * 0.1)
        for _ in range(30):
            m.predict({"x": a}, output_backings={"y": b})
        ts = []
        for _ in range(REPS):
            t = time.perf_counter()
            m.predict({"x": a}, output_backings={"y": b})
            ts.append(1e3 * (time.perf_counter() - t))
        res[s] = stats(ts)
        print(f"Core ML S={s}: median {res[s][0]:.3f} ms (p10 {res[s][1]:.3f}) on {devs}", flush=True)
    per = (res[8][0] - res[1][0]) / 7
    print(f"Core ML: per conv {per:.3f} ms, per-call overhead {res[1][0] - per:.3f} ms (p10-based "
          f"{res[1][1] - (res[8][1] - res[1][1]) / 7:.3f})", flush=True)


def coreai():
    import asyncio
    import torch
    import torch.nn as nn
    import coreai_torch
    from coreai.runtime import AIModel, NDArray
    from coreai_opt.casting import cast_to_16_bit_precision
    from coreai_bench_helpers import specialization_for

    class Chain(nn.Module):
        def __init__(self, s):
            super().__init__()
            self.convs = nn.ModuleList(nn.Conv2d(C_, C_, 1, bias=False) for _ in range(s))
            with torch.no_grad():
                for c, w in zip(self.convs, weights(s)):
                    c.weight.copy_(torch.from_numpy(w.astype(np.float32)))

        def forward(self, x):
            for c in self.convs:
                x = c(x)
            return x

    async def run(s):
        out = ROOT / f"cai_S{s}.aimodel"
        if not out.exists():
            ep = torch.export.export(Chain(s).eval().to(torch.float16), (torch.randn(1, C_, 1, T_, dtype=torch.float16),),
                                     strict=False).run_decompositions(coreai_torch.get_decomp_table())
            cast_to_16_bit_precision(ep)
            conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
            conv.add_exported_program(ep, input_names=["x"], output_names=["y"])
            prog = conv.to_coreai()
            prog.optimize()
            ROOT.mkdir(parents=True, exist_ok=True)
            prog.save_asset(out)
        model = await AIModel.load(out, specialization_options=specialization_for("ane"))
        fn = model.load_function("main")
        x = NDArray(np.ones((1, C_, 1, T_), np.float16) * 0.1)
        for _ in range(30):
            await fn(inputs={"x": x})
        ts = []
        for _ in range(REPS):
            t = time.perf_counter()
            await fn(inputs={"x": x})
            ts.append(1e3 * (time.perf_counter() - t))
        h = (out / "main.hash").read_bytes().hex()
        mans = sorted(Path.home().glob(f"Library/Caches/coreai-cache/*/*/{h}/*/model.aimodelx/**/manifest.plist"))
        pl = "ANE" if mans and b"mps.fullyPlacedOnANE" in mans[-1].read_bytes() else "not fully ANE"
        return stats(ts), pl

    res = {}
    for s in (1, 8):
        res[s], pl = asyncio.run(run(s))
        print(f"Core AI S={s}: median {res[s][0]:.3f} ms (p10 {res[s][1]:.3f}) on {pl}", flush=True)
    per = (res[8][0] - res[1][0]) / 7
    print(f"Core AI: per conv {per:.3f} ms, per-call overhead {res[1][0] - per:.3f} ms (p10-based "
          f"{res[1][1] - (res[8][1] - res[1][1]) / 7:.3f})", flush=True)


if __name__ == "__main__":
    {"coreml": coreml, "coreai": coreai}[sys.argv[1]]()
