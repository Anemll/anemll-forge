"""In-place host updates of a Core AI input NDArray, seen by the ANE? The NDArray constructor copies its source and
numpy() returns fresh buffers, but the buffer-protocol pointer of the NDArray's own storage is stable: a writable
numpy view over it (ctypes) lets the host rewrite KV rows in place (qwen38_coreai_model.py relies on this). Test: an
attention-over-KV graph on the ANE (exact-length KV input), rows rewritten in place between calls, each output vs
numpy on the current contents; plus the time of a rows write vs a fresh NDArray of the whole cache.
    .venv/bin/python coreai_inplace_probe.py"""
from __future__ import annotations

import asyncio
import ctypes
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import coreai_torch
from coreai.runtime import AIModel, NDArray
from coreai.runtime._ndarray import StorageKind
from coreai_opt.casting import cast_to_16_bit_precision

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # coreai (builder, coreai_util)
from coreai_util import specialization_for

C, D, NKV, T, CTX = 512, 256, 4, 8, 16384
ROOT = Path(__file__).resolve().parent / "artifacts_inplace"
CACHE = Path.home() / "Library/Caches/coreai-cache"


def writable(nd: NDArray, dtype=np.float16) -> np.ndarray:
    """Writable zero-copy numpy view of an NDArray's own storage (its buffer-protocol pointer is stable)."""
    mv = memoryview(nd._tensor)  # noqa: SLF001
    ptr = np.frombuffer(mv, dtype=np.uint8).__array_interface__["data"][0]
    n = mv.nbytes
    return np.ctypeslib.as_array((ctypes.c_uint8 * n).from_address(ptr)).view(dtype).reshape(nd.shape)


class InputKV(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(C, C, 1, bias=False)

    def forward(self, x, k, v):
        y = self.conv(x)
        q = y[0, :D, 0, :].transpose(0, 1).unsqueeze(0).expand(NKV, T, D)
        o = torch.softmax(q @ k.transpose(1, 2) * D ** -0.5, -1) @ v
        return y, o


def build() -> Path:
    out = ROOT / "inplace.aimodel"
    if out.exists():
        return out
    m = InputKV().eval().to(torch.float16)
    ex = (torch.randn(1, C, 1, T, dtype=torch.float16), torch.randn(NKV, CTX, D, dtype=torch.float16),
          torch.randn(NKV, CTX, D, dtype=torch.float16))
    ep = torch.export.export(m, ex, strict=False).run_decompositions(coreai_torch.get_decomp_table())
    cast_to_16_bit_precision(ep)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    conv.add_exported_program(ep, input_names=["x", "k", "v"], output_names=["y", "o"])
    prog = conv.to_coreai()
    prog.optimize()
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)
    return out


def placement(since: float) -> str:
    mans = [p for p in CACHE.glob("*/*/*/*/model.aimodelx/**/manifest.plist") if p.stat().st_mtime >= since]
    if not mans:
        return "cached"
    text = max(mans, key=lambda p: p.stat().st_mtime).read_bytes()
    return ("fully on ANE" if b"mps.fullyPlacedOnANE" in text else "NOT fully on ANE") + f", {text.count(b'_ANE_region_')} regions"


async def main() -> None:
    t0 = time.time()
    model = await AIModel.load(build(), specialization_options=specialization_for("ane"))
    print(f"load; {placement(t0)}", flush=True)
    fn = model.load_function("main")
    r = np.random.default_rng(0)
    for kind in (StorageKind.IO_SURFACE, StorageKind.BYTES):
        k_nd = NDArray(np.zeros((NKV, CTX, D), np.float16), kind)
        v_nd = NDArray(np.zeros((NKV, CTX, D), np.float16), kind)
        kw, vw = writable(k_nd), writable(v_nd)
        x_nd = NDArray((r.standard_normal((1, C, 1, T)) * 0.1).astype(np.float16), kind)
        worst = 1.0
        for it in range(6):
            pos = int(r.integers(0, CTX - 64))
            kw[:, pos:pos + 64] = (r.standard_normal((NKV, 64, D)) * 2).astype(np.float16)   # in-place row writes
            vw[:, pos:pos + 64] = r.standard_normal((NKV, 64, D)).astype(np.float16)
            out = await fn(inputs={"x": x_nd, "k": k_nd, "v": v_nd})
            y = out["y"].numpy()[0, :D, 0, :].T.astype(np.float32)
            kk, vv = kw.astype(np.float32), vw.astype(np.float32)
            s = y[None] @ kk.transpose(0, 2, 1) * D ** -0.5
            s = np.exp(s - s.max(-1, keepdims=True))
            ref = (s / s.sum(-1, keepdims=True)) @ vv
            o = out["o"].numpy().astype(np.float64)
            cos = float(o.ravel() @ ref.ravel() / (np.linalg.norm(o) * np.linalg.norm(ref) + 1e-30))
            worst = min(worst, cos)
        t = time.perf_counter()
        for _ in range(10):
            kw[:, 100:108] = 1
        t_rows = (time.perf_counter() - t) / 10 * 1e3
        t = time.perf_counter()
        for _ in range(3):
            NDArray(kw, kind)
        t_new = (time.perf_counter() - t) / 3 * 1e3
        feed = {"x": x_nd, "k": k_nd, "v": v_nd}
        ts = []
        for _ in range(15):
            tt = time.perf_counter()
            await fn(inputs=feed)
            ts.append(1e3 * (time.perf_counter() - tt))
        print(f"{kind.value}: ANE sees in-place writes: worst cos {worst:.5f} over 6 rewrites; 8-row write {t_rows:.3f} ms vs "
              f"fresh NDArray of the {NKV * CTX * D * 2 / 2**20:.0f} MB cache {t_new:.1f} ms; call {np.median(ts):.2f} ms", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
