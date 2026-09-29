"""Which entry point keeps a Core AI package off the ANE? Builds small packages (layers x entries), loads each on the ANE
and reports placement + call time; then deletes the package and its cache entry.
    .venv/bin/python qwen38_coreai_bisect.py "0:v8_2k" "0:p64_2k" "3:v8_32k" "3:v8_64k" ...
spec = <layers, e.g. 0 or 0-3>:<entry,entry,...> with entry v8_<ctx>k | p64_<ctx>k"""
from __future__ import annotations

import asyncio
import gc
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ane-vector-lut/coreai (builder, coreai_util)
import qwen38_coreai_build as B
from coreai.runtime import AIModel
from coreai_util import specialization_for

CACHE = Path.home() / "Library/Caches/coreai-cache"


def placement(since: float) -> str:
    mans = [p for p in CACHE.glob("*/*/*/*/model.aimodelx/**/manifest.plist") if p.stat().st_mtime >= since]
    if not mans:
        return "no new manifest"
    text = max(mans, key=lambda p: p.stat().st_mtime).read_bytes()
    return ("fully on ANE" if b"mps.fullyPlacedOnANE" in text else "NOT fully on ANE") + f", {text.count(b'_ANE_region_')} region refs"


async def one(ck, spec: str) -> str:
    lay, ents = spec.split(":")
    a, _, b = lay.partition("-")
    layers = list(range(int(a), int(b or a) + 1))
    ctxs = [int(e.split("_")[1][:-1]) * 1024 for e in ents.split(",") if e.startswith("v8_")]
    pctxs = [int(e.split("_")[1][:-1]) * 1024 for e in ents.split(",") if e.startswith("p64_")]
    name = "bisect_" + spec.replace(":", "_").replace(",", "_")
    B.build_chunk(ck, layers, ctxs, pctxs, name=name)
    path = B.OUT / f"{name}.aimodel"
    t0 = time.time()
    try:
        model = await AIModel.load(path, specialization_options=specialization_for("ane"))
        res = f"{spec}: load {time.time() - t0:.0f}s, {placement(t0)}"
        del model
    except Exception as e:  # noqa: BLE001
        res = f"{spec}: LOAD FAILED {str(e)[:120]}"
    digest = (path / "main.hash").read_bytes().hex() if (path / "main.hash").exists() else None
    shutil.rmtree(path, ignore_errors=True)
    if digest:
        for d in CACHE.glob(f"*/*/{digest}"):
            shutil.rmtree(d, ignore_errors=True)
    gc.collect()
    print("RESULT " + res, flush=True)
    return res


async def main():
    ck = B.M.Checkpoint()
    for spec in sys.argv[1:]:
        await one(ck, spec)


if __name__ == "__main__":
    asyncio.run(main())
