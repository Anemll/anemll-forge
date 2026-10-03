"""Point a Pi profile at the context the local Qwen3.8 ANE server runs with.

Updates only the target `provider/model` entry of `<pi-dir>/models.json` (contextWindow, maxTokens, name) and
`<pi-dir>/settings.json` (compaction.modelOverrides: reserveTokens, keepRecentTokens), so Pi sends at most what
fits and compacts before the window fills. Pi itself clamps every request to
    max_tokens = min(maxTokens, contextWindow - estimated_prompt - 4096)
(CONTEXT_SAFETY_TOKENS in pi-ai, floored at 1), so the compaction point must leave room for the output cap:
    maxTokens         = clamp(ctx / 4, 2048, 16384)     (16K -> 4096, 32K -> 8192, 64K -> 16384)
    reserveTokens     = maxTokens + 4096                 (16K: compacts above ~8K; 64K: above ~45K)
    keepRecentTokens  = min(ctx / 8, 8192)               (16K -> 2048, 64K -> 8192)

The first run keeps the original files as `*.orig-qwen38`; every change keeps the previous version as
`*.prev-qwen38`. Pi's `/model` reloads models.json; restart Pi after changing compaction settings.

Examples:
    python scripts/qwen38_pi_config.py --ctx 65536
    python scripts/qwen38_pi_config.py --ctx 65536 --pi-dir "$HOME/.pi-forge" --build coreai_mixr12
    python scripts/qwen38_pi_config.py --ctx 16384 --dry-run
"""
import argparse
import json
import os
import shutil
import stat
import tempfile

from pathlib import Path
from qwen38_hardware_profile import require_m5pro_24gb

PROVIDER, MODEL = "ane-qwen38", "qwen38-27b-ane"
PI_SAFETY = 4096  # pi-ai CONTEXT_SAFETY_TOKENS reserved below the window for the response
CONTEXTS = (8192, 16384, 24576, 31744, 32768, 49152, 65536)


def budget(ctx: int) -> tuple[int, int, int]:
    """Return (maxTokens, compaction reserveTokens, keepRecentTokens) for a context window."""
    if ctx not in CONTEXTS:
        raise ValueError(f"unsupported context {ctx}; use one of {CONTEXTS}")
    max_tokens = max(2048, min(16384, ctx // 4))
    return max_tokens, max_tokens + PI_SAFETY, min(ctx // 8, 8192)


def model_name(ctx: int, build: str, draft: bool) -> str:
    base = (f"Qwen3.8 27B VQ {build} (ANE, {ctx // 1024}K"
            f"{', DFlash' if draft else ''})").replace("VQ  (", "VQ (")
    return base


def _stage(path: Path, content: bytes) -> Path:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.qwen38-", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), stat.S_IMODE(path.stat().st_mode))
            stream.write(content)
        return tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _save(updates: list[tuple[Path, dict]]) -> None:
    staged, originals, committed = {}, {}, []
    try:
        for path, data in updates:
            originals[path] = path.read_bytes()
            staged[path] = _stage(path, (json.dumps(data, indent=2) + "\n").encode())
            orig, prev = path.with_name(path.name + ".orig-qwen38"), path.with_name(path.name + ".prev-qwen38")
            if not orig.exists():
                shutil.copy2(path, orig)
            shutil.copy2(path, prev)
        for path, tmp in staged.items():
            os.replace(tmp, path)
            committed.append(path)
    except OSError:
        for path in reversed(committed):
            tmp = _stage(path, originals[path])
            try:
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        raise
    finally:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)


def sync(pi_dir: Path, ctx: int, build: str, draft: bool, dry_run: bool = False) -> list[str]:
    """Apply the context budget to one Pi profile; return a list of human-readable changes."""
    if ctx == 31744:
        require_m5pro_24gb()
    models_path, settings_path = pi_dir / "models.json", pi_dir / "settings.json"
    if not models_path.is_file():
        raise ValueError(f"no models.json in {pi_dir}")
    max_tokens, reserve, keep = budget(ctx)
    changes: list[str] = []
    updates: list[tuple[Path, dict]] = []

    models = json.loads(models_path.read_text())
    if not settings_path.is_file():
        raise ValueError(f"no settings.json in {pi_dir}")
    settings = json.loads(settings_path.read_text())
    if not isinstance(models, dict) or not isinstance(settings, dict):
        raise ValueError("models.json and settings.json must contain JSON objects")
    providers = models.get("providers", {})
    if not isinstance(providers, dict) or not isinstance(providers.get(PROVIDER, {}), dict):
        raise ValueError("invalid providers in models.json")
    entries = providers.get(PROVIDER, {}).get("models", [])
    if not isinstance(entries, list) or any(not isinstance(m, dict) for m in entries):
        raise ValueError("invalid model entries in models.json")
    entry = next((m for m in entries
                  if m.get("id") == MODEL), None)
    if entry is None:
        raise ValueError(f"no {PROVIDER}/{MODEL} entry in {models_path}")
    compaction = settings.setdefault("compaction", {})
    if not isinstance(compaction, dict):
        raise ValueError("invalid compaction in settings.json")
    over = compaction.setdefault("modelOverrides", {})
    if not isinstance(over, dict):
        raise ValueError("invalid compaction.modelOverrides in settings.json")
    current = over.get(f"{PROVIDER}/{MODEL}", {})
    if not isinstance(current, dict):
        raise ValueError("invalid per-model compaction override in settings.json")
    new = {"contextWindow": ctx, "maxTokens": max_tokens, "name": model_name(ctx, build, draft)}
    if any(entry.get(k) != v for k, v in new.items()):
        entry.update(new)
        changes.append(f"{models_path}: contextWindow {ctx}, maxTokens {max_tokens}, name '{new['name']}'")
        updates.append((models_path, models))
    want = {"reserveTokens": reserve, "keepRecentTokens": keep}
    if any(current.get(k) != v for k, v in want.items()):
        over[f"{PROVIDER}/{MODEL}"] = {**current, **want}
        changes.append(f"{settings_path}: compaction reserve {reserve} (compacts above ~{ctx - reserve}), keepRecent {keep}")
        updates.append((settings_path, settings))
    if not dry_run:
        _save(updates)
    return changes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ctx", type=int, required=True, help="server context window in tokens (e.g. 16384, 65536)")
    ap.add_argument("--pi-dir", type=Path, default=Path.home() / ".pi" / "agent",
                    help="Pi profile directory (default: ~/.pi/agent)")
    ap.add_argument("--build", default="", help="build directory name recorded in the model display name")
    ap.add_argument("--draft", choices=("on", "off"), default="on", help="whether the drafter is enabled (display only)")
    ap.add_argument("--dry-run", action="store_true", help="report changes without writing")
    a = ap.parse_args(argv)
    pi_dir = a.pi_dir.expanduser().resolve()
    try:
        changes = sync(pi_dir, a.ctx, a.build, a.draft == "on", a.dry_run)
    except (ValueError, OSError) as e:
        print(f"pi config: {e}")
        return 1
    if changes:
        print("pi config: " + ("would update" if a.dry_run else "updated") + " " + "; ".join(changes))
        print("Reopen /model in Pi (restart Pi after changing compaction settings).")
    else:
        print("pi config: already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
