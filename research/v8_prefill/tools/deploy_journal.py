"""Durable per-device Core AI deployment journal (JSONL, one file per device, append + fsync per record), written so the
EXISTING cleanup tools can later delete exactly what was deployed. Nothing here deletes anything.

Journals: C/deployment_journal/m6.jsonl and C/deployment_journal/iphone_<UDID>.jsonl. One line per event:
  {"schema": "coreai_deploy_journal/1", "ts": ISO local time, "event": "attempt" | "result" | "backfill",
   "attempt_id": "<run nonce>:<n>", "device": {...}, "run": {driver_sha256, plan_sha256, out, mode},
   "package": {"path": ..., "main_hash": hex | "UNKNOWN", "files_sha256": {...} | "UNKNOWN"}, "entry": ..., "functions": [...],
   "request": {require_cached, compile_mode, ...}, "outcome": "ok" | "failed" | "rejected_by_host" | "no_reply",
   "cache_entry_before": bool | "UNKNOWN", "cache_entry_after": bool | "UNKNOWN", "raw_reply": full reply | null,
   "error": str | null, "cleanup": {...}}
An "attempt" line is written BEFORE the native LOAD / compile is started, so a crash or timeout still leaves it; the
"result" line follows. Cleanup identity:
  iPhone: the package path is relative to the app's Documents; deletion is AIModelCache.deleteEntries(for:) from the
          SAME app (bundle com.anemll.coreaiphonetest), i.e. Runner "mode=purge model=<dir>", which reads
          <Documents>/<dir>/manifest.json {"chunks": [{"file": ...}]}. phone_purge_manifests() writes those manifests.
  M6:     scripts/coreai_cache_delete.py --delete <package paths>, run with the SAME python executable (the cache is split
          per executable name). m6_delete_args() lists the paths and that executable."""
import datetime, json, os, sys
from pathlib import Path

# Portable copy (22 Nov 2026, the HF V8-96K kit): the journal root is DEPLOY_JOURNAL_ROOT (default ~/.anemll-forge);
# the development machine's own copy (C/claude_full96_prep/freeze_eight_row_v4) writes to C/deployment_journal.
C = Path(os.environ.get("DEPLOY_JOURNAL_ROOT", str(Path.home() / ".anemll-forge")))
ROOT = C / "deployment_journal"
SCHEMA = "coreai_deploy_journal/1"
BUNDLE = "com.anemll.coreaiphonetest"
DELETE_TOOL = os.environ.get("COREAI_CACHE_DELETE", "scripts/coreai_cache_delete.py")


def journal_path(device):
    return ROOT / ("m6.jsonl" if device["kind"] == "m6" else f"iphone_{device['udid']}.jsonl")


def append(device, rec, root=None):
    p = journal_path(device) if root is None else Path(root) / journal_path(device).name
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"schema": SCHEMA, "ts": datetime.datetime.now().astimezone().isoformat(timespec="milliseconds"),
                       "device": device, **rec}, default=str, sort_keys=True)
    fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        mv = memoryview((line + "\n").encode())
        while mv:                                   # os.write may write fewer bytes than asked
            n = os.write(fd, mv)
            if n <= 0:
                raise OSError(f"journal write returned {n}")
            mv = mv[n:]
        os.fsync(fd)                                # durable BEFORE the caller starts the native LOAD
    finally:
        os.close(fd)
    return p


def phone_device(udid, machine, app_uuid, endpoint):
    return {"kind": "iphone", "udid": udid, "machine": machine, "bundle_id": BUNDLE, "executable_uuid": app_uuid, "endpoint": endpoint,
            "cleanup": {"how": "Runner mode=purge model=<package dir> (deleteEntries from the same app)", "tool": "CoreAIPhoneTest Runner.swift mode=purge"}}


def m6_device():
    import subprocess
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    return {"kind": "m6", "host": os.uname().nodename, "os_build": build, "python": sys.executable, "cache_client": Path(sys.executable).name,
            "cleanup": {"how": f"{sys.executable} {DELETE_TOOL} --delete <package>", "tool": DELETE_TOOL}}


class Journal:
    """one per run: numbered attempt ids, the run identity on every line"""

    def __init__(self, run, nonce, root=None):
        self.run, self.nonce, self.n, self.root = run, nonce, 0, root

    def attempt(self, device, package, **kw):
        self.n += 1
        aid = f"{self.nonce}:{self.n}"
        append(device, {"event": "attempt", "attempt_id": aid, "run": self.run, "package": package, **kw}, self.root)
        return aid

    def result(self, device, aid, package, outcome, raw_reply=None, error=None, **kw):
        append(device, {"event": "result", "attempt_id": aid, "run": self.run, "package": package, "outcome": outcome,
                        "raw_reply": raw_reply, "error": None if error is None else str(error)[:4000], **kw}, self.root)


def read(path):
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def phone_purge_manifests(journal_file, out_dir):
    """per package directory in Documents: a manifest.json {"chunks": [{"file": rel}]} listing EVERY package any attempt
    touched (successful or failed), for Runner mode=purge model=<dir>. Written under out_dir/<dir>/manifest.json (to be
    copied into the app's Documents/<dir>/ by devicectl). Returns {dir: [files]}."""
    by = {}
    for r in read(journal_file):
        p = (r.get("package") or {}).get("path")
        if r.get("device", {}).get("kind") == "iphone" and isinstance(p, str) and "/" in p:
            d, f = p.split("/", 1)
            if f not in by.setdefault(d, []):
                by[d].append(f)
    for d, files in by.items():
        q = Path(out_dir) / d
        q.mkdir(parents=True, exist_ok=True)
        (q / "manifest.json").write_text(json.dumps({"chunks": [{"file": f} for f in sorted(files)], "purpose": "cache purge only (from the deployment journal)"}, indent=1) + "\n")
    return by


def m6_delete_args(journal_file):
    """(python executable, [package paths]) for coreai_cache_delete.py --delete, from every M6 attempt"""
    recs = [r for r in read(journal_file) if r.get("device", {}).get("kind") == "m6"]
    pys = sorted({r["device"]["python"] for r in recs})
    paths = sorted({r["package"]["path"] for r in recs if isinstance((r.get("package") or {}).get("path"), str)})
    return pys, paths
