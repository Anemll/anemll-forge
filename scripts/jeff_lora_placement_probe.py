#!/usr/bin/env python3
"""Which dynamic LoRA layout stays on the ANE?

Compiles tiny graphs of one gate-sized projection (x [1,1024,1,256], rank 16,
out 3584): a constant 1x1 base conv plus y += (x @ A) @ (sB), with A and sB as
model inputs. Layouts: conv weights, plain matmul, conv-style [1,C,1,W], rank
padded to 32/64, and a flat pack. Prints the segmented-cache placement
(mps.fullyPlacedOnANE, ANE vs GPU regions) and the MPSGraph placement analysis.

    COREAI_PYTHON=.../coreai_gemm_bench/.venv/bin/python
    "$COREAI_PYTHON" scripts/jeff_lora_placement_probe.py [--only conv_r16,matmul_r16]
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ane_compile_mode  # noqa: E402

IN_F, OUT_F, T = 1024, 3584, 256
ART = Path(os.environ.get("LORA_PROBE_DIR", "/tmp/jeff-lora-probe"))


def _base(cin, cout):
    conv = nn.Conv2d(cin, cout, 1, bias=False)
    conv.weight = nn.Parameter(torch.randn(cout, cin, 1, 1).to(torch.float16) * 0.02, requires_grad=False)
    return conv


class ConstLoRA(nn.Module):
    """Control: A and sB are constants, the form the 27B low-rank path already uses."""

    def __init__(self, rank: int):
        super().__init__()
        self.base = _base(IN_F, OUT_F)
        self.a = nn.Conv2d(IN_F, rank, 1, bias=False)
        self.b = nn.Conv2d(rank, OUT_F, 1, bias=False)
        self.a.weight = nn.Parameter(torch.randn(rank, IN_F, 1, 1).to(torch.float16) * 0.02, requires_grad=False)
        self.b.weight = nn.Parameter(torch.randn(OUT_F, rank, 1, 1).to(torch.float16) * 0.02, requires_grad=False)

    def forward(self, x):
        return self.base(x) + self.b(self.a(x))

    def example(self):
        return (torch.zeros(1, IN_F, 1, T, dtype=torch.float16),)

    def names(self):
        return ["x"], ["y"]


class DynConv(nn.Module):
    """A [rank,in,1,1] and sB [out,rank,1,1] are conv-weight inputs."""

    def __init__(self, rank: int, n: int = 1, cin: int = IN_F, cout: int = OUT_F, rows: int = T):
        super().__init__()
        self.rank, self.n, self.cin, self.cout, self.rows = rank, n, cin, cout, rows
        self.base = nn.ModuleList(_base(cin, cout) for _ in range(n))

    def forward(self, x, *factors):
        acc = None
        for i, base in enumerate(self.base):
            term = base(x) + F.conv2d(F.conv2d(x, factors[2 * i]), factors[2 * i + 1])
            acc = term if acc is None else acc + term
        return acc

    def example(self):
        f = torch.float16
        ex = [torch.zeros(1, self.cin, 1, self.rows, dtype=f)]
        for _ in range(self.n):
            ex.append(torch.zeros(self.rank, self.cin, 1, 1, dtype=f))
            ex.append(torch.zeros(self.cout, self.rank, 1, 1, dtype=f))
        return tuple(ex)

    def names(self):
        ins = ["x"]
        for i in range(self.n):
            ins += [f"a{i}", f"b{i}"]
        return ins, ["y"]


class DynMatmul(nn.Module):
    """A [in,rank], sB [rank,out] as plain matmul inputs. x stays [1,C,1,T]."""

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank
        self.base = _base(IN_F, OUT_F)

    def forward(self, x, a, b):
        rows = x.reshape(IN_F, T).transpose(0, 1)
        delta = (rows @ a) @ b
        return self.base(x) + delta.transpose(0, 1).reshape(1, OUT_F, 1, T)

    def example(self):
        f = torch.float16
        return (torch.zeros(1, IN_F, 1, T, dtype=f), torch.zeros(IN_F, self.rank, dtype=f),
                torch.zeros(self.rank, OUT_F, dtype=f))

    def names(self):
        return ["x", "a", "b"], ["y"]


class DynNCHW(nn.Module):
    """A [1,in,1,rank] and sB [1,rank,1,out], then either matmul or a permute into conv weights."""

    def __init__(self, rank: int, as_conv: bool):
        super().__init__()
        self.rank, self.as_conv = rank, as_conv
        self.base = _base(IN_F, OUT_F)

    def forward(self, x, a, b):
        if self.as_conv:
            aw = a.permute(3, 1, 0, 2)
            bw = b.permute(3, 1, 0, 2)
            delta = F.conv2d(F.conv2d(x, aw), bw)
        else:
            rows = x.reshape(IN_F, T).transpose(0, 1)
            delta = (rows @ a.reshape(IN_F, self.rank)) @ b.reshape(self.rank, OUT_F)
            delta = delta.transpose(0, 1).reshape(1, OUT_F, 1, T)
        return self.base(x) + delta

    def example(self):
        f = torch.float16
        return (torch.zeros(1, IN_F, 1, T, dtype=f), torch.zeros(1, IN_F, 1, self.rank, dtype=f),
                torch.zeros(1, self.rank, 1, OUT_F, dtype=f))

    def names(self):
        return ["x", "a", "b"], ["y"]


class DynPack(nn.Module):
    """One flat buffer sliced into conv weights. qkv-B is 98304 elems, over the 65536 axis cap if 1-D."""

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank
        self.base = _base(IN_F, OUT_F)
        self.a_n = rank * IN_F
        self.b_n = OUT_F * rank

    def forward(self, x, packed):
        flat = packed.reshape(-1)
        a = flat[:self.a_n].reshape(self.rank, IN_F, 1, 1)
        b = flat[self.a_n:self.a_n + self.b_n].reshape(OUT_F, self.rank, 1, 1)
        return self.base(x) + F.conv2d(F.conv2d(x, a), b)

    def example(self):
        f = torch.float16
        n = self.a_n + self.b_n
        return (torch.zeros(1, IN_F, 1, T, dtype=f), torch.zeros(1, 1, 1, n, dtype=f))

    def names(self):
        return ["x", "packed"], ["y"]


def variants():
    return {
        "const_lora": lambda: ConstLoRA(16),
        "conv_r16": lambda: DynConv(16),
        "conv_r32": lambda: DynConv(32),
        "conv_r64": lambda: DynConv(64),
        "matmul_r16": lambda: DynMatmul(16),
        "matmul_r32": lambda: DynMatmul(32),
        "matmul_r64": lambda: DynMatmul(64),
        "nchw_matmul_r16": lambda: DynNCHW(16, as_conv=False),
        "nchw_conv_r16": lambda: DynNCHW(16, as_conv=True),
        "nchw_conv_r32": lambda: DynNCHW(32, as_conv=True),
        "nchw_conv_r64": lambda: DynNCHW(64, as_conv=True),
        "pack_flat_r16": lambda: DynPack(16),
        "conv_r16_x6": lambda: DynConv(16, n=6),
        "conv_r16_x25_small": lambda: DynConv(16, n=25, cin=64, cout=64, rows=64),
    }


def export(mod, dest: Path) -> None:
    import coreai_torch
    from coreai_opt.casting import cast_to_16_bit_precision

    ins, outs = mod.names()
    ep = torch.export.export(mod, mod.example(), strict=False).run_decompositions(coreai_torch.get_decomp_table())
    cast_to_16_bit_precision(ep)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    conv.add_exported_program(ep, input_names=ins, output_names=outs, entrypoint_name="main")
    prog = conv.to_coreai()
    prog.optimize()
    shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    prog.save_asset(dest)


def placement(package: Path) -> dict:
    digest = (package / "main.hash").read_bytes().hex()
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    root = Path.home() / "Library/Caches" / "coreai-cache" / build
    mans = list(root.glob(f"*/{digest}/**/manifest.plist"))
    info = {"digest": digest, "manifests": len(mans), "fully_ane": False, "no_gpu": False,
            "ane_regions": 0, "gpu_regions": 0, "mlir": []}
    if not mans:
        info["note"] = "no cached manifest"
        return info
    mf = max(mans, key=lambda p: p.stat().st_mtime)
    blob = mf.read_bytes()
    info["manifest"] = str(mf)
    info["fully_ane"] = b"mps.fullyPlacedOnANE" in blob
    info["no_gpu"] = b"mps.noGPUActivity" in blob
    info["ane_regions"] = len(set(re.findall(rb"[A-Za-z0-9_-]+_ANE_region_[A-Za-z0-9_]+", blob)))
    info["gpu_regions"] = len(set(re.findall(rb"[A-Za-z0-9_-]+_GPU_region_[A-Za-z0-9_]+", blob)))
    try:
        versions = plistlib.loads(blob).get("Package Version", {})
    except plistlib.InvalidFileException:
        versions = {}
    for fields in versions.values():
        for module in fields.get("Optimized Modules", {}).values():
            filename = module.get("File Name")
            if not filename:
                continue
            graph = mf.parent / filename
            if graph.is_file():
                gblob = graph.read_bytes()
                info["ane_regions"] = max(info["ane_regions"], len(set(re.findall(rb"[A-Za-z0-9_-]+_ANE_region_[A-Za-z0-9_]+", gblob))))
                info["gpu_regions"] = max(info["gpu_regions"], len(set(re.findall(rb"[A-Za-z0-9_-]+_GPU_region_[A-Za-z0-9_]+", gblob))))
            for mlir in graph.parent.rglob("*.mlir*"):
                info["mlir"].append(str(mlir.relative_to(mf.parent)) if mlir.is_relative_to(mf.parent) else mlir.name)
    info["mlir"] = sorted(set(info["mlir"]))[:12]
    # HWX path from ANERegionsHash, same lookup as scripts/m6_hwx_inspect.py (M5 is h17c).
    hwx = []
    for fields in versions.values():
        h = fields.get("ANERegionsHash", {}).get("h17c")
        if h and "_" in h:
            a, b = h.split("_", 1)
            path = Path("/Library/Caches/com.apple.aned") / build / "ModelAssetsCache" / "-_unsigned" / a / b / "model.hwx"
            try:
                exists = path.is_file()
                size = path.stat().st_size if exists else 0
                note = ""
            except OSError as exc:
                exists, size, note = False, 0, f"{type(exc).__name__}: {exc}"
            hwx.append({"path": str(path), "exists": exists, "bytes": size, "note": note})
    info["hwx"] = hwx
    return info


def unplaced_lines(text: str) -> list[str]:
    keep = []
    for line in text.splitlines():
        low = line.lower()
        if any(s in low for s in ("could not", "unsupported", "gpu", "not placed", "fallback", "cpu")):
            if "fully placed" in low:
                continue
            keep.append(line.strip()[:240])
    return keep[:40]


async def _load(dest: Path):
    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine())
    return await AIModel.load(dest, specialization_options=options)


async def run_one(name: str, dump: bool) -> dict:
    dest = ART / f"{name}.aimodel"
    mod = variants()[name]().eval().to(torch.float16)
    t0 = time.perf_counter()
    try:
        export(mod, dest)
    except Exception as exc:  # noqa: BLE001 — one bad layout should not stop the sweep
        return {"name": name, "export_error": f"{type(exc).__name__}: {exc}"[:500]}
    export_s = time.perf_counter() - t0
    os.environ["MPSGRAPH_PRINT_ANE_PLACEMENT_ANALYSIS"] = "1"
    cwd = None
    if dump:
        os.environ["MPSGRAPH_DUMP_MODULE"] = "1"
        dump_dir = ART / "dump" / name
        dump_dir.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(dump_dir)
    else:
        os.environ.pop("MPSGRAPH_DUMP_MODULE", None)
    t1 = time.perf_counter()
    load_error = None
    model = None
    try:
        model = await _load(dest)
    except Exception as exc:  # noqa: BLE001
        load_error = f"{type(exc).__name__}: {exc}"[:800]
    finally:
        if cwd:
            os.chdir(cwd)
    load_s = time.perf_counter() - t1
    row = {"name": name, "export_s": round(export_s, 2), "load_s": round(load_s, 2),
           "inputs": list(mod.names()[0]), "n_inputs": len(mod.names()[0])}
    if load_error:
        row["load_error"] = load_error
    else:
        row["placement"] = placement(dest)
        del model
    if dump:
        dumped = sorted(p.name for p in (ART / "dump" / name).glob("*"))
        row["dump_files"] = dumped[:30]
    gc.collect()
    print(json.dumps({k: row[k] for k in row if k != "unplaced"}), flush=True)
    if row.get("unplaced"):
        print("  unplaced:", *row["unplaced"][:8], sep="\n    ", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated variant names")
    ap.add_argument("--dump", default="conv_r16,matmul_r16,nchw_conv_r16,pack_flat_r16",
                    help="variants that also set MPSGRAPH_DUMP_MODULE")
    args = ap.parse_args()
    os.environ.pop("USE_LOCAL_COREAI", None)
    ane_compile_mode.apply(log=print)
    ART.mkdir(parents=True, exist_ok=True)
    names = [n for n in args.only.split(",") if n] or list(variants())
    unknown = [n for n in names if n not in variants()]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {list(variants())}")
    dump = set(args.dump.split(","))

    async def go():
        rows = []
        for name in names:
            print(f"\n== {name}", flush=True)
            rows.append(await run_one(name, name in dump))
        return rows

    rows = asyncio.run(go())
    out = ART / "placement.json"
    out.write_text(json.dumps(rows, indent=2))
    print("\nlayout                         on ANE  gpu_regions  ane_regions  load_s")
    for row in rows:
        if "placement" not in row:
            print(f"{row['name']:30} FAIL {row.get('export_error') or row.get('load_error')}")
            continue
        p = row["placement"]
        flag = "YES" if p.get("fully_ane") and not p.get("gpu_regions") else "NO"
        print(f"{row['name']:30} {flag:6}  {p.get('gpu_regions', '-'):11}  {p.get('ane_regions', '-'):11}  {row['load_s']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
