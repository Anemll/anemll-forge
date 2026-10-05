"""Hybrid Core AI target build: chunks from two builds of the same model, for bisecting a numerical problem by chunk.

Creates OUT/<name>/<build dir name>/ with the manifest and head of build A and symlinks to every chunk package: the
chunks listed in --from-b come from build B, the rest from A. The packages are the compiled ones of A and B, so a
hybrid loads without a new compile. Evaluate it like any build (scripts/m6_long_ctx_eval.py, m6_kl512_eval.py) and
halve the chunk set until one chunk carries the difference.

    python scripts/m6_hybrid_build.py --a <baseline build dir> --b <candidate build dir> --from-b 12,13,14,15 \\
        --out DIR --name hi
Both builds must have the same chunk plan, context entries and KV format (compare their manifests)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="baseline build dir (manifest.json, head, chunks)")
    ap.add_argument("--b", type=Path, required=True, help="candidate build dir")
    ap.add_argument("--from-b", default="", help="comma-separated chunk indices (0 = first chunk) taken from B")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    ma, mb = (json.loads((d / "manifest.json").read_text()) for d in (a.a, a.b))
    fa, fb = [c["file"] for c in ma["chunks"]], [c["file"] for c in mb["chunks"]]
    if fa != fb:
        raise SystemExit(f"chunk plans differ: {fa} vs {fb}")
    if ma.get("ctxs") != mb.get("ctxs") or ma.get("pctxs") != mb.get("pctxs"):
        raise SystemExit("context entries differ")
    use_b = {int(x) for x in a.from_b.split(",") if x}
    out = a.out / a.name / a.a.name
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(fa):
        dst = out / f
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to((a.b if i in use_b else a.a) / f)
    for f in a.a.iterdir():  # manifest, head, anything else of A
        if f.name not in fa and not (out / f.name).exists():
            (out / f.name).symlink_to(f)
    print(out, "| chunks from B:", sorted(use_b))


if __name__ == "__main__":
    main()
