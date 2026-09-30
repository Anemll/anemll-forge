"""Merge Core AI chunk builds made in separate OUT directories (parallel builds must not share one manifest.json) into
the main build directory: move the chunk packages, union the manifests' chunk lists.
    .venv/bin/python coreai_merge_builds.py <main dir> <other dir> [<other dir> ...]"""
import json
import shutil
import sys
from pathlib import Path


def main():
    main_dir, others = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    man = json.loads((main_dir / "manifest.json").read_text())
    chunks = {c["file"]: c for c in man["chunks"]}
    for d in others:
        m2 = json.loads((d / "manifest.json").read_text())
        for key in ("ctxs", "pctxs", "kv_len", "pkv_len", "T", "TP", "pend", "taps"):
            assert m2.get(key) == man.get(key), f"{d}: {key} differs ({m2.get(key)} vs {man.get(key)})"
        for c in m2["chunks"]:
            src = d / c["file"]
            if not src.exists():
                raise SystemExit(f"missing {src}")
            dst = main_dir / c["file"]
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            chunks[c["file"]] = c
            print(f"moved {c['file']}")
    man["chunks"] = sorted(chunks.values(), key=lambda c: c["layers"][0])
    (main_dir / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"{len(man['chunks'])} chunks: {[c['layers'] for c in man['chunks']]}")


if __name__ == "__main__":
    main()
