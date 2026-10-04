"""Compile (specialize) a Core AI target build for this Mac's ANE ahead of serving, with the guided progress of
coreai_compile_guide: what is compiling, time left, safe to stop (finished packages stay cached; rerun to resume).
Run it with the same Python the server uses: the compile cache is keyed by macOS build and Python (its bundle
identifier, e.g. org.python.python for a framework Python, else its executable name). --force drops this Python's
cached specializations of the build first, so every package recompiles.

    python forge.py compile --build <target build> [--draft <dflash2 .aimodel>] [--force]
    python scripts/coreai_compile.py --build <target build> [--draft <dflash2 .aimodel>] [--force]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "coreai" / "swift_bridge"))
import ane_compile_mode as SOC  # noqa: E402
import coreai_bridge as B  # noqa: E402
import coreai_compile_guide as G  # noqa: E402
from qwen38_coreai_model import MODE_ENV, graph_line, pick_package  # noqa: E402


def stamp(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, required=True, help="target build directory (manifest.json, packages)")
    ap.add_argument("--draft", type=Path, help="also compile this DFlash2 drafter package")
    ap.add_argument("--force", action="store_true", help="drop this Python's cached specializations first and recompile")
    a = ap.parse_args(argv)
    try:
        SOC.apply(strict=True)
    except SOC.UnsupportedSocError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    man = json.loads((a.build / "manifest.json").read_text())
    stamp(graph_line(man, a.build))
    if a.force:
        pkgs = [a.build / c["file"] for c in man["chunks"]] + [a.build / man["head"]["file"]] + ([a.draft] if a.draft else [])
        n = sum(G.purge(p) for p in pkgs)
        stamp(f"{G.TAG} --force: purged {n} cached specializations of {len(pkgs)} packages "
              f"(macOS {G.os_build()} / {G.process_key()})")
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
          f"macOS {G.os_build()} / {G.process_key()}; the server now loads them in seconds")


if __name__ == "__main__":
    raise SystemExit(main())
