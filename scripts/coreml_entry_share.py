"""Core ML counterpart of fp8-mlp-metal41-bench/coreai/coreai_entry_share.py "ladder": the same toy (16 x Conv2d(4096,
4096) fp16 + attention over K / V inputs) as a Core ML multifunction model with entries s2k / s8k / s16k (8 rows,
KV length 2048 / 8192 / 16384) and p64_s2k (64 rows): wired memory after loading each function and after its first
prediction, and the call times (ANE).
    python coreml_entry_share.py"""
import gc
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct

C, S, D = 4096, 16, 256
OUT = Path(__file__).parent / "qwen38_prefill" / "coreml_entry_share"
ENTRIES = {"s2k": (8, 2048), "s8k": (8, 8192), "s16k": (8, 16384), "p64_s2k": (64, 2048)}


def wired_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * page / 2 ** 30


class AttnChain(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.convs = nn.ModuleList(nn.Conv2d(C, C, 1, bias=False) for _ in range(S))
        for c in self.convs:
            nn.init.normal_(c.weight, std=C ** -0.5)

    def forward(self, x, k, v, mask):
        for c in self.convs:
            x = c(x)
        q = x[0, :D, 0, :].transpose(0, 1)
        o = torch.softmax(q @ k.transpose(0, 1) * 0.0625 + mask, -1) @ v
        return x, o


def build():
    mlc = OUT / "ladder_fp16.mlmodelc"
    if mlc.exists():
        return mlc
    OUT.mkdir(parents=True, exist_ok=True)
    model = AttnChain().eval()
    desc = ct.utils.MultiFunctionDescriptor()
    for name, (rows, ctx) in ENTRIES.items():
        ex = (torch.randn(1, C, 1, rows), torch.randn(ctx, D), torch.randn(ctx, D), torch.zeros(1, ctx))
        traced = torch.jit.trace(model, ex)
        m = ct.convert(traced, inputs=[ct.TensorType("x", shape=ex[0].shape, dtype=np.float16),
                                       ct.TensorType("k", shape=ex[1].shape, dtype=np.float16),
                                       ct.TensorType("v", shape=ex[2].shape, dtype=np.float16),
                                       ct.TensorType("mask", shape=ex[3].shape, dtype=np.float16)],
                       outputs=[ct.TensorType("y", dtype=np.float16), ct.TensorType("o", dtype=np.float16)],
                       minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16)
        p = OUT / f"{name}.mlpackage"
        shutil.rmtree(p, ignore_errors=True)
        m.save(str(p))
        desc.add_function(str(p), "main", name)
    desc.default_function_name = "s2k"
    pkg = OUT / "ladder_fp16.mlpackage"
    shutil.rmtree(pkg, ignore_errors=True)
    ct.utils.save_multifunction(desc, str(pkg))
    ct.models.utils.compile_model(str(pkg), str(mlc))
    for name in ENTRIES:
        shutil.rmtree(OUT / f"{name}.mlpackage", ignore_errors=True)
    shutil.rmtree(pkg, ignore_errors=True)
    return mlc


def main():
    mlc = build()
    size = sum(f.stat().st_size for f in mlc.rglob("*") if f.is_file()) / 1e6
    gc.collect()
    w0 = wired_gb()
    fns, log = {}, []
    for name in ENTRIES:
        fns[name] = ct.models.CompiledMLModel(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE, function_name=name)
        log.append(f"fn {name} +{wired_gb() - w0:.2f} GB")
    r = np.random.default_rng(1)
    for name, (rows, ctx) in ENTRIES.items():
        feed = {"x": (r.standard_normal((1, C, 1, rows)) * 0.1).astype(np.float16),
                "k": r.standard_normal((ctx, D)).astype(np.float16), "v": r.standard_normal((ctx, D)).astype(np.float16),
                "mask": np.zeros((1, ctx), np.float16)}
        fns[name].predict(feed)
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            fns[name].predict(feed)
            ts.append(1e3 * (time.perf_counter() - t))
        log.append(f"{name} {np.median(ts):.2f} ms +{wired_gb() - w0:.2f} GB")
    print(f"[coreml ladder_fp16] ({size:.0f} MB) " + " | ".join(log), flush=True)


if __name__ == "__main__":
    main()
