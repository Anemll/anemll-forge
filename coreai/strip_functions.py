"""Keep only some functions of a Core AI package: load the .aimodel, erase every other function (GraphOp) from its
program and save a new package. The weights are untouched (the kept functions reference the same resources).

Use: a package exported with two function sets (qwen38_coreai_build.py ATT_INT8MM_M5: <entry> for M6 with FP8 and
<entry>_m5 without) cannot compile on an M5, whose ANE compiler rejects the FP8 functions and fails the whole package;
stripping the FP8 functions on the M5 gives an M5-only package without a second download.

    python coreai/strip_functions.py SRC.aimodel DST.aimodel --keep-suffix _m5      # the M5 set
    python coreai/strip_functions.py SRC.aimodel DST.aimodel --drop-suffix _m5      # the M6 set
Needs the Core AI authoring package (coreai, as the builder)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from coreai._compiler.dialects import coreai, udml
from coreai.authoring.asset import AIModelAsset


def graphs(region):
    """Every GraphOp in a region, recursing into udml namespaces."""
    for op in region.blocks[0].operations:
        if isinstance(op, coreai.GraphOp):
            yield op
        elif isinstance(op, udml.NamespaceOp):
            yield from graphs(op.regions[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--keep-suffix", help="keep only functions whose name ends with this")
    g.add_argument("--drop-suffix", help="drop the functions whose name ends with this")
    a = ap.parse_args()
    t0 = time.time()
    prog = AIModelAsset.load(a.src).program
    keep, drop = [], []
    for op in list(graphs(prog._mlir_module.body.region)):  # noqa: SLF001
        name = op.sym_name.value
        kept = name.endswith(a.keep_suffix) if a.keep_suffix else not name.endswith(a.drop_suffix)
        (keep if kept else drop).append(name)
        if not kept:
            op.operation.erase()
    if not keep:
        raise SystemExit("no function left")
    prog.save_asset(a.dst)
    mb = sum(f.stat().st_size for f in a.dst.rglob("*") if f.is_file()) / 1e6
    print(f"kept {keep}\ndropped {drop}\n{a.dst}: {mb:.0f} MB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
