"""Context-capped builds: a target build with only the context entries up to a cap.

Core AI compiles a package as a whole, so a chunk with every entry from 8K to 192K compiles all of them, whatever
context the server will use: about 8 minutes per chunk of the 192K build (2 h 11 min in all on the M6), mostly the
long entries' prefill tiles. The Neural Engine also reserves each chunk's scratch for its largest function. A capped
build keeps, in each chunk, only the functions of the contexts up to the cap (same weights, coreai/strip_functions.py,
about a second per chunk) and the manifest entries to match; it is written once to $ANEMLL_FORGE_STATE/builds/ (the
download stays read-only; one more copy of the chunks on disk), the head linked, the manifest written last.
Measured on the M6 from the 192K build: a 16K cap compiled in 8 min 37 s and prefilled 267 to 280 tok/s (the full
build 241 to 260).

`forge.py compile --ctx 16K` creates and compiles the capped build; the runtime then uses it for any start whose
context cap fits in it (CTX=16K), and the full build otherwise. Needs coreai-core on the Mac that derives."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIDECARS = ("basis.json", "R.npy")  # rot1 builds keep these beside the manifest


def parse_cap(value) -> int | None:
    """16384, "16384" or "16K" -> 16384; None or "" -> None."""
    if value in (None, ""):
        return None
    text = str(value).strip().upper()
    return int(text[:-1]) * 1024 if text.endswith("K") else int(text)


def tag(ctx: int) -> str:
    return f"_{ctx // 1024}k"


def capped_manifest(man: dict, cap: int, source: Path | None = None) -> dict:
    """The manifest of the build capped at `cap`: contexts, rows, KV lengths and chunk entries up to it."""
    ctxs = [c for c in man["ctxs"] if c <= cap]
    if not ctxs:
        raise ValueError(f"no context entry at or below {cap}; this build has {man['ctxs']}")
    m = copy.deepcopy(man)
    keep = {tag(c) for c in ctxs}
    m["ctxs"] = ctxs
    m["pctxs"] = [c for c in man.get("pctxs", []) if c <= cap]
    for key in ("kv_len", "pkv_len", "ptp"):
        if key in m:
            m[key] = {c: v for c, v in m[key].items() if int(c) <= cap}
    if m.get("ptp"):
        m["TP"] = max(m["ptp"].values())
    for c in m["chunks"]:
        c["entries"] = [e for e in c["entries"] if any(e.endswith(t) for t in keep)]
        c.pop("compiled", None)
        for soc, names in (c.get("entries_by_soc") or {}).items():
            c["entries_by_soc"][soc] = {k: v for k, v in names.items() if any(k.endswith(t) for t in keep)}
    m["derived_ctx"] = {"cap": cap, **({"from": str(source)} if source else {})}
    return m


def path(root: Path, cap: int, state: Path | None = None) -> Path:
    """Where the build at `root` capped at `cap` lives (keyed by the source manifest, so a new download derives anew)."""
    root = Path(root)
    raw = (root / "manifest.json").read_bytes()
    state = Path(state or os.environ.get("ANEMLL_FORGE_STATE") or Path.home() / ".anemll-forge")
    return state / "builds" / f"{root.name}-ctx{cap // 1024}k-{hashlib.sha256(raw).hexdigest()[:12]}"


def capped_for(root: Path, ctx: int | None) -> int | None:
    """The cap to use for a server context `ctx`: the build's largest entry at or below it, None for the full build."""
    if ctx is None:
        return None
    ctxs = json.loads((Path(root) / "manifest.json").read_text())["ctxs"]
    fit = [c for c in ctxs if c <= ctx]
    return fit[-1] if fit and fit[-1] < max(ctxs) else None


def find(root: Path, ctx: int | None, state: Path | None = None) -> Path | None:
    """An already derived capped build that serves context `ctx`: the smallest cap at or above the entries it needs."""
    cap = capped_for(root, ctx)
    if cap is None:
        return None
    ctxs = json.loads((Path(root) / "manifest.json").read_text())["ctxs"]
    for c in ctxs:
        if c >= cap and c < max(ctxs) and (path(root, c, state) / "manifest.json").exists():
            return path(root, c, state)
    return None


def prepare(root: Path, ctx, log=print, state: Path | None = None, create: bool = True) -> Path:
    """root itself, or its build capped for context `ctx`. create=False only reuses a capped build already derived
    (the runtime: deriving is a decision taken by `forge.py compile --ctx`)."""
    root, ctx = Path(root), parse_cap(ctx)
    cap = capped_for(root, ctx)
    if cap is None:
        return root
    found = find(root, cap, state)
    if found is not None:
        log(f"[ctx] using the build capped at {json.loads((found / 'manifest.json').read_text())['ctxs'][-1] // 1024}K "
            f"derived from {root}: {found}")
        return found
    if not create:
        return root
    out = path(root, cap, state)
    man = json.loads((root / "manifest.json").read_text())
    capped = capped_manifest(man, cap, root)
    sys.path.insert(0, str(ROOT / "coreai"))
    import strip_functions
    try:
        strip_functions.require()
    except ImportError as e:
        raise RuntimeError(f"a build capped at {cap // 1024}K is derived on this Mac, which needs the Core AI authoring "
                           f"package: python -m pip install coreai-core ({e})") from None
    need = sum(f.stat().st_size for c in man["chunks"] for f in (root / c["file"]).rglob("*") if f.is_file())
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free
    if free < need * 1.05:
        raise RuntimeError(f"deriving the {cap // 1024}K build needs {need / 1e9:.1f} GB in {out}; "
                           f"{free / 1e9:.1f} GB free")
    log(f"[ctx] deriving the build capped at {cap // 1024}K (contexts {[c // 1024 for c in capped['ctxs']]}K, same "
        f"weights) into {out}: {len(man['chunks'])} chunks, up to {need / 1e9:.1f} GB")
    t0 = time.time()
    for c, cc in zip(man["chunks"], capped["chunks"]):
        dst = out / c["file"]
        if dst.exists():
            continue
        tmp = dst.with_name(dst.stem + ".partial.aimodel")
        shutil.rmtree(tmp, ignore_errors=True)
        strip_functions.strip(root / c["file"], tmp, {e: e for e in cc["entries"]})
        tmp.rename(dst)
    head = man["head"]["file"]
    if not (out / head).exists():
        (out / head).symlink_to((root / head).resolve(), target_is_directory=True)
    for name in SIDECARS:
        if (root / name).is_file():
            shutil.copy2(root / name, out / name)
    (out / "manifest.json").write_text(json.dumps(capped, indent=1))
    log(f"[ctx] {cap // 1024}K build ready in {time.time() - t0:.0f}s")
    return out
