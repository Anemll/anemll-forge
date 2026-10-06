"""Keep only some functions of a Core AI package: load the .aimodel, erase every other function (GraphOp) from its
program, optionally rename the kept ones, and save a new package. The weights are untouched (the kept functions
reference the same resources).

Use: a package exported with two function sets (qwen38_coreai_build.py ATT_INT8MM_M5: <entry> for M6 with FP8 and
<entry>_m5 without) cannot compile on an M5, whose ANE compiler rejects the FP8 functions and fails the whole package;
keeping the _m5 functions (renamed to <entry>) gives an M5 package without a second download. The runtime does this
on first start (scripts/soc_variant.py); this CLI is the manual form.

    python coreai/strip_functions.py SRC.aimodel DST.aimodel --keep-suffix _m5 [--rename]   # the M5 set
    python coreai/strip_functions.py SRC.aimodel DST.aimodel --drop-suffix _m5              # the M6 set
Needs the Core AI authoring package (coreai-core on PyPI)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def mlir_module(prog):
    """The program's MLIR module: AIProgram._mlir_module in coreai-core 1.0.0b2, ._module._mlir_module in 1.0.0b3."""
    m = getattr(prog, "_mlir_module", None)
    return m if m is not None else prog._module._mlir_module  # noqa: SLF001


def graphs(region):
    """Every function (coreai.graph op) in a region, recursing into udml namespaces; by op name, which is stable
    across coreai-core versions (the Python op classes moved)."""
    for op in region.blocks[0].operations:
        name = op.operation.name
        if name == "coreai.graph":
            yield op
        elif name == "udml.namespace":
            yield from graphs(op.regions[0])


def strip(src: Path, dst: Path, keep: dict[str, str]) -> tuple[list[str], list[str]]:
    """Save src as dst with only the functions in keep (physical name -> name in dst). Returns (kept, dropped)."""
    from coreai._compiler.ir import StringAttr
    from coreai.authoring.asset import AIModelAsset
    prog = AIModelAsset.load(src).program
    module = mlir_module(prog)
    kept, dropped = [], []
    with module.context:
        for op in list(graphs(module.body.region)):
            name = op.sym_name.value
            if name in keep:
                if keep[name] != name:
                    op.attributes["sym_name"] = StringAttr.get(keep[name])
                kept.append(name)
            else:
                op.operation.erase()
                dropped.append(name)
    missing = set(keep) - set(kept)
    if missing:
        raise ValueError(f"{src.name}: no function {sorted(missing)}")
    prog.save_asset(dst)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--keep-suffix", help="keep only functions whose name ends with this")
    g.add_argument("--drop-suffix", help="drop the functions whose name ends with this")
    ap.add_argument("--rename", action="store_true", help="with --keep-suffix: remove the suffix from the kept names")
    a = ap.parse_args()
    t0 = time.time()
    from coreai.authoring.asset import AIModelAsset
    names = AIModelAsset.load(a.src).summary(include_statistics=False).function_names
    if a.keep_suffix:
        keep = {n: (n[:-len(a.keep_suffix)] if a.rename else n) for n in names if n.endswith(a.keep_suffix)}
    else:
        keep = {n: n for n in names if not n.endswith(a.drop_suffix)}
    if not keep:
        raise SystemExit("no function left")
    kept, dropped = strip(a.src, a.dst, keep)
    mb = sum(f.stat().st_size for f in a.dst.rglob("*") if f.is_file()) / 1e6
    print(f"kept {[keep[k] for k in kept]}\ndropped {dropped}\n{a.dst}: {mb:.0f} MB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
