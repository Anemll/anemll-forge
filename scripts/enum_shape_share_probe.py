"""Can ONE Core ML function with enumerated input shapes serve several shapes on the ANE with one resident copy of
the weights? (multifunction models duplicate weights in ANE memory; see FEEDBACK_ANE_MULTIFUNCTION_MEMORY.md)
Toy: NL palettized (LUT4 per-tensor) 1x1 convs 5120 -> 10240 (sliced back) + attention over K / V inputs.
    A: rows T enumerated {8, 64}              (verify / prefill with one weight copy?)
    B: KV length S enumerated {2048, 8192}    (context ladder with one weight copy?)
For each: time per shape vs fixed-shape builds of the same weights (CPU fallback would be far slower) and wired
memory after load and after the first prediction at each shape.
    python enum_shape_share_probe.py"""
import gc
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct
import coremltools.optimize.coreml as cto

CIN, COUT, NL, D = 5120, 10240, 8, 256
OUT = Path(__file__).parent / "qwen38_prefill" / "enumprobe"


def wired_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * page / 2 ** 30


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lins = nn.ModuleList(nn.Conv2d(CIN, COUT, 1, bias=False) for _ in range(NL))
        for l in self.lins:
            nn.init.normal_(l.weight, std=CIN ** -0.5)

    def forward(self, x, k, v, mask):                 # x (1, CIN, 1, T); k, v (S, D); mask (1, S)
        for l in self.lins:
            x = l(x)[:, :CIN]
        q = x[0, :D, 0, :].transpose(0, 1)             # (T, D)
        o = torch.softmax(q @ k.transpose(0, 1) * D ** -0.5 + mask, -1) @ v
        return x, o


def convert(name, T, S):
    """T, S: an int (fixed) or a list (enumerated). Returns the compiled path."""
    mlc = OUT / f"{name}.mlmodelc"
    if mlc.exists():
        return mlc
    t0, s0 = (T if isinstance(T, int) else T[0]), (S if isinstance(S, int) else S[0])
    ex = (torch.randn(1, CIN, 1, t0), torch.randn(s0, D), torch.randn(s0, D), torch.zeros(1, s0))
    traced = torch.jit.trace(Toy().eval(), ex)
    shp = lambda fn, vals: ct.EnumeratedShapes(shapes=[fn(v) for v in vals], default=fn(vals[0])) if isinstance(vals, list) else fn(vals)  # noqa: E731
    inputs = [ct.TensorType("x", shape=shp(lambda t: (1, CIN, 1, t), T), dtype=np.float16),
              ct.TensorType("k", shape=shp(lambda s: (s, D), S), dtype=np.float16),
              ct.TensorType("v", shape=shp(lambda s: (s, D), S), dtype=np.float16),
              ct.TensorType("mask", shape=shp(lambda s: (1, s), S), dtype=np.float16)]
    m = ct.convert(traced, inputs=inputs, outputs=[ct.TensorType("y", dtype=np.float16), ct.TensorType("o", dtype=np.float16)],
                   minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16)
    cfg = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(nbits=4, mode="uniform", granularity="per_tensor"))
    m = cto.palettize_weights(m, cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    pkg = OUT / f"{name}.mlpackage"
    shutil.rmtree(pkg, ignore_errors=True)
    m.save(str(pkg))
    ct.models.utils.compile_model(str(pkg), str(mlc))
    shutil.rmtree(pkg, ignore_errors=True)
    return mlc


def feeds(T, S):
    r = np.random.default_rng(1)
    return {"x": (r.standard_normal((1, CIN, 1, T)) * 0.1).astype(np.float16), "k": r.standard_normal((S, D)).astype(np.float16),
            "v": r.standard_normal((S, D)).astype(np.float16), "mask": np.zeros((1, S), np.float16)}


def timed(m, f, n=20):
    for _ in range(3):
        m.predict(f)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        m.predict(f)
        ts.append(1e3 * (time.perf_counter() - t))
    return np.median(ts)


def run(label, mlc, shapes):
    gc.collect()
    w0 = wired_gb()
    t = time.time()
    m = ct.models.CompiledMLModel(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)
    msg = [f"load {time.time() - t:.1f}s +{wired_gb() - w0:.2f} GB"]
    for T, S in shapes:
        ms = timed(m, feeds(T, S))
        msg.append(f"T={T} S={S}: {ms:.2f} ms, wired +{wired_gb() - w0:.2f} GB")
    print(f"[{label}] " + " | ".join(msg), flush=True)
    del m
    gc.collect()
    time.sleep(1)


def main():
    print(f"toy: {NL} LUT4 convs {CIN}->{COUT} ({NL * CIN * COUT / 2 / 2**20:.0f} MB palettized) + attention D={D}", flush=True)
    run("A fixed T=8", convert("fixT8", 8, 2048), [(8, 2048)])
    run("A fixed T=64", convert("fixT64", 64, 2048), [(64, 2048)])
    run("A enum T{8,64}", convert("enumT", [8, 64], 2048), [(8, 2048), (64, 2048), (8, 2048)])
    run("B fixed S=2048", convert("fixS2k", 8, 2048), [(8, 2048)])
    run("B fixed S=8192", convert("fixS8k", 8, 8192), [(8, 8192)])
    run("B enum S{2048,8192}", convert("enumS", 8, [2048, 8192]), [(8, 2048), (8, 8192), (8, 2048)])


if __name__ == "__main__":
    main()
