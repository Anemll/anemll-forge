"""Clean up the two ANE compile caches on macOS without breaking what is still in use.

Core AI keeps an entry per package and Python in ~/Library/Caches/coreai-cache/<macOS build>/<client>/<hash>/; its
manifest.plist names the compiled ANE program in aned's cache (ANERegionsHash {soc: "<a>_<b>"} ->
/Library/Caches/com.apple.aned/<build>/ModelAssetsCache/<client>/<a>/<b>/). The two drift apart:
  - orphaned programs: aned programs no Core AI entry names any more (research builds whose Core AI entries were
    purged or replaced). They only take disk space: 273 programs, 176 GB on the M6 on 17 Oct 2026.
  - dangling entries: Core AI entries whose aned program is gone (aned's cache was cleaned). Loading one fails with
    nilError instead of recompiling; removing the entry makes the next load compile again.
  - failed compiles: Core AI specializations with no compiled MPSGraph package in them (an aborted or failed
    compile), or compiled in another ANE bonded compile mode than this Mac's policy (a load that did not set
    MPSGRAPH_ANE_BONDED_COMPILE_MODE: mode 0 keeps a second copy of the weights). Removing one makes the next load
    compile it again; its aned program is then orphaned (--delete-orphans).
  - stale compile dirs: MPSGraph's working dir of each compile ($TMPDIR/com.apple.MetalPerformanceShadersGraph/
    mpsgraph-<pid>-<time>-*), left behind when the compiling process died (0.4 to 0.8 GB each for a 27B chunk).

    python scripts/ane_cache_cleanup.py                    report only (sizes, counts, other macOS builds)
    python scripts/ane_cache_cleanup.py --delete-dangling  also remove dangling Core AI entries (yours, no sudo)
    python scripts/ane_cache_cleanup.py --delete-failed    also remove failed and wrong-mode Core AI specializations
    python scripts/ane_cache_cleanup.py --delete-temp      also remove compile dirs whose process is gone (no sudo)
    python scripts/ane_cache_cleanup.py --delete-orphans   orphaned aned programs: aned's cache is root-owned, so
                                                           without root this writes the list and prints the sudo
                                                           command; as root (sudo) it deletes them itself
Reading aned's cache needs read access to /Library/Caches/com.apple.aned (an ACL or root). Only aned's client for
unsigned processes (-_unsigned: Python and other command-line tools) is checked by default; other clients (Apple's
*_platform, SHARED_SYSTEM_MODELS, signed apps) have their Core AI caches elsewhere and are never touched. Programs
written in the last --recent minutes are kept (a compile in progress writes its Core AI entry last). Stop servers
and compiles first anyway. Only program directories inside ModelAssetsCache/<client>/ are ever deleted.
"""
from __future__ import annotations

import argparse
import os
import platform
import plistlib
import pwd
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import ane_compile_mode as SOC

ANED = Path("/Library/Caches/com.apple.aned")


def user_home() -> Path:
    """The invoking user's home, also under sudo (whose ~ is root's)."""
    name = os.environ.get("SUDO_USER")
    return Path(pwd.getpwnam(name).pw_dir) if name else Path.home()


def os_build() -> str:
    return subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip() or platform.version()


def du(p: Path) -> int:
    out = subprocess.run(["du", "-sk", str(p)], capture_output=True, text=True).stdout.split()
    return int(out[0]) * 1024 if out else 0


def stale_compile_dirs() -> list[Path]:
    """MPSGraph compile dirs (mpsgraph-<pid>-...) in this user's temp dir whose process is no longer running."""
    tmp = Path(os.environ.get("TMPDIR") or subprocess.run(["getconf", "DARWIN_USER_TEMP_DIR"], capture_output=True,
                                                          text=True).stdout.strip())
    out = []
    for d in sorted((tmp / "com.apple.MetalPerformanceShadersGraph").glob("mpsgraph-*")):
        try:
            os.kill(int(d.name.split("-")[1]), 0)
        except ProcessLookupError:
            out.append(d)
        except (ValueError, IndexError, PermissionError):
            pass   # not a pid-named dir, or a live process of another user
    return out


def failed_specializations(entries, mode: int | None, cutoff: float) -> list[tuple[Path, str]]:
    """(specialization dir, reason) for Core AI specializations (<entry>/<id>/) last written before `cutoff` that hold
    no compiled MPSGraph package, or one compiled in another bonded mode than `mode` (None: not checked)."""
    out = []
    for e in entries:
        for spec in (d for d in e.iterdir() if d.is_dir()):
            if max([spec.stat().st_mtime] + [c.stat().st_mtime for c in spec.iterdir()]) > cutoff:
                continue   # being written: a compile in progress
            modes = set()
            found = False
            for mf in spec.rglob("manifest.plist"):
                if ".mpsgraphpackage" in str(mf):
                    found = True
                    modes.update(int(m) for m in re.findall(rb"aneBondedCompileMode\W+(\d+)", mf.read_bytes()))
            if not found:
                out.append((spec, "no compiled package (failed or aborted compile)"))
            elif mode is not None and modes and mode not in modes:
                out.append((spec, f"bonded compile mode {sorted(modes)}, this Mac uses {mode}"))
        if not any(d.is_dir() for d in e.iterdir()) and e.stat().st_mtime <= cutoff:
            out.append((e, "empty entry"))
    return out


def program_refs(entry: Path) -> set[str]:
    """The '<a>_<b>' aned program hashes a Core AI cache entry's manifests name."""
    refs: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "ANERegionsHash" and isinstance(v, dict):
                    refs.update(h for h in v.values() if isinstance(h, str))
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for m in entry.glob("*/model.aimodelx/**/manifest.plist"):
        try:
            walk(plistlib.load(open(m, "rb")))
        except Exception:  # noqa: BLE001  unreadable or partial manifest: names nothing
            pass
    return refs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=os_build(), help="macOS build whose caches to check (default: this one)")
    ap.add_argument("--delete-dangling", action="store_true", help="remove Core AI entries whose aned program is gone")
    ap.add_argument("--delete-orphans", action="store_true", help="delete aned programs no Core AI entry names")
    ap.add_argument("--delete-temp", action="store_true", help="remove MPSGraph compile dirs whose process is gone")
    ap.add_argument("--delete-failed", action="store_true",
                    help="remove Core AI specializations of failed compiles or of another compile mode")
    ap.add_argument("--list", type=Path, default=Path("/tmp/ane_orphans.txt"), help="orphan list for the sudo step")
    ap.add_argument("--client", default="-_unsigned", help="aned client directory to check (default: -_unsigned)")
    ap.add_argument("--recent", type=float, default=15, help="keep programs modified in the last N minutes")
    a = ap.parse_args()
    coreai = user_home() / "Library/Caches/coreai-cache" / a.build
    assets = ANED / a.build / "ModelAssetsCache"
    if not assets.is_dir():
        sys.exit(f"no {assets} (or no read access: grant it, or run with sudo)")

    client, cutoff = assets / a.client, time.time() - 60 * a.recent

    def scan():
        """Core AI entries of every client (one directory per process name, e.g. python, python3) with the aned
        programs they name; aned's programs of this client; orphans (none named, not recent) and recent ones."""
        entries = {e: program_refs(e) for c in sorted(coreai.iterdir()) if c.is_dir() for e in sorted(c.iterdir())
                   if e.is_dir()} if coreai.is_dir() else {}
        referenced = set().union(*entries.values()) if entries else set()
        present = {}
        for prog_a in (p for p in client.iterdir() if p.is_dir()) if client.is_dir() else ():
            for prog_b in (p for p in prog_a.iterdir() if p.is_dir()):
                present[f"{prog_a.name}_{prog_b.name}"] = prog_b
        recent = {h for h, p in present.items() if h not in referenced and p.stat().st_mtime > cutoff}
        orphans = {h: p for h, p in present.items() if h not in referenced and h not in recent}
        return entries, referenced, present, recent, orphans

    def report_orphans(orphans, recent):
        print(f"  orphaned aned programs in {a.client}: {len(orphans)}, {sum(du(p) for p in orphans.values()) / 1e9:.1f} GB"
              + (f" (and {len(recent)} written in the last {a.recent:.0f} min, kept: a compile may be running)"
                 if recent else ""))

    entries, referenced, present, recent, orphans = scan()
    dangling = [e for e, refs in entries.items() if refs and not refs & present.keys()]
    print(f"macOS {a.build}: {len(entries)} Core AI entries naming {len(referenced)} programs; aned holds {len(present)}")
    report_orphans(orphans, recent)
    print(f"  dangling Core AI entries (their program is gone, they would fail with nilError): {len(dangling)}, "
          f"{sum(du(e) for e in dangling) / 1e9:.1f} GB")
    mode = SOC.policy_mode(SOC.detect_soc())
    failed = failed_specializations(entries, mode, cutoff)
    print(f"  failed or wrong-mode Core AI specializations: {len(failed)}, {sum(du(d) for d, _ in failed) / 1e9:.1f} GB")
    for d, why in failed:
        print(f"    {d.parent.name[:16] if d.parent != coreai else d.name[:16]}... {why}")
    stale = [] if os.environ.get("SUDO_USER") else stale_compile_dirs()   # root's TMPDIR is not the user's
    print(f"  stale MPSGraph compile dirs (their process is gone): {len(stale)}, {sum(du(d) for d in stale) / 1e9:.1f} GB")
    others = [d.name for d in ANED.iterdir() if d.is_dir() and d.name != a.build and (d / "ModelAssetsCache").is_dir()]
    if others:
        print(f"  aned caches of other macOS builds (unused by this one): {', '.join(others)}")

    if a.delete_dangling:
        for e in dangling:
            shutil.rmtree(e)
        print(f"removed {len(dangling)} dangling Core AI entries")
    if a.delete_failed:
        for d, _ in failed:
            shutil.rmtree(d)
        print(f"removed {len(failed)} failed or wrong-mode specializations")
    if a.delete_temp:
        for d in stale:
            shutil.rmtree(d)
        print(f"removed {len(stale)} stale compile dirs")
    if (a.delete_dangling and dangling) or (a.delete_failed and failed):
        entries, referenced, present, recent, orphans = scan()   # the programs they named are orphans now
        print("after that:")
        report_orphans(orphans, recent)
    if a.delete_orphans and orphans:
        if os.geteuid() == 0:
            for p in orphans.values():
                if client in p.parents:   # only program directories inside ModelAssetsCache/<client>/
                    shutil.rmtree(p)
            print(f"deleted {len(orphans)} orphaned aned programs")
        else:
            a.list.write_text("\n".join(str(p) for p in orphans.values()) + "\n")
            root = str(client) + "/"
            print(f"aned's cache is root-owned. The list is in {a.list}; delete with:\n"
                  f"  sudo sh -c 'while IFS= read -r d; do case \"$d\" in {root}*) rm -rf -- \"$d\";; esac; done' "
                  f"< {a.list}\n  (or: sudo {sys.executable} {Path(__file__).resolve()} --delete-orphans)")


if __name__ == "__main__":
    main()
