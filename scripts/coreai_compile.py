"""Compile (specialize) a Core AI target build for this Mac's ANE ahead of serving, with the guided progress of
coreai_compile_guide: what is compiling, time left, safe to stop (finished packages stay cached; rerun to resume).
Run it with the same Python the server uses: the compile cache is keyed by macOS build and Python executable name.

    python forge.py compile --build <target build> [--draft <dflash2 .aimodel>]
    python scripts/coreai_compile.py --build <target build> [--draft <dflash2 .aimodel>]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPSGRAPH_ANE_BONDED_COMPILE_MODE", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "coreai" / "swift_bridge"))
import coreai_bridge as B  # noqa: E402
import coreai_compile_guide as G  # noqa: E402
from qwen38_coreai_model import MODE_ENV, graph_line, pick_package  # noqa: E402


def stamp(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, required=True, help="target build directory (manifest.json, packages)")
    ap.add_argument("--draft", type=Path, help="also compile this DFlash2 drafter package")
    a = ap.parse_args(argv)
    man = json.loads((a.build / "manifest.json").read_text())
    stamp(graph_line(man, a.build))
    extra = [(f"drafter {a.draft.name}", a.draft, G.DRAFTER_S)] if a.draft else []
    guide = G.target_guide(man, a.build, log=stamp, extra=extra, mode=int(os.environ.get(MODE_ENV, "0")))
    guide.hint_lines = [h for h in guide.hint_lines if "forge.py compile" not in h]  # this is that command
    guide.announce()
    t0 = time.time()
    entries = [(c["file"], c.get("compiled")) for c in man["chunks"]] + [(man["head"]["file"], man["head"].get("compiled"))]

    def load(file, compiled):
        target, _ = pick_package(a.build, file, compiled, stamp)
        B.Model(target, compute="ane")  # the first load specializes the package and caches it; released right away

    for file, compiled in entries:
        guide.load(file, lambda f=file, c=compiled: load(f, c))
    if a.draft:
        guide.load(f"drafter {a.draft.name}", lambda: B.Model(a.draft, compute="ane"))
    stamp(f"{G.TAG} done in {G.fmt(time.time() - t0)}: {len(entries) + bool(a.draft)} packages compiled and cached for "
          f"macOS {G.os_build()} / {Path(sys.executable).name}; the server now loads them in seconds")


if __name__ == "__main__":
    main()
