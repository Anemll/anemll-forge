"""Portable GPU-table kit for a V8 build (22 Nov 2026). The GPU engine reads the palettized mixer weights zero-copy from
the Neural Engine's compiled programs (model.hwx in aned's cache): the table files `*.tab` hold offsets into those
programs and the manifest names each program's path and weight section (`hwx`: path, off, size). Everything else in a
table directory (norms, rotations, low-rank adapters, dense k / v, DeltaNet constants) comes from the release export
and does not depend on the machine.

  pack     W_DIR BUILD KIT      (on the build machine) copy W_DIR's table files into KIT with a sha256 per file, and
                                record per HWX the package it belongs to, its weight section (off, size) and the
                                section's sha256; the manifests are stored with the machine paths removed.
  install  KIT BUILD OUT        (on any Mac, after the build's packages were compiled there: forge.py compile) find
                                each package's local model.hwx, check that its weight section has the recorded offset,
                                size and sha256 (= the same compiled program, so the shipped .tab offsets are valid),
                                copy the files (sha256 checked) and write OUT/manifest.json and manifest_best.json with
                                the local paths. Exit 2 when a program differs (another OS build or compiler): then
                                rebuild the tables from the export with build_gpu_tables.sh.
  verify   OUT REF              every table file byte-identical and the manifests equal apart from "dir" and the HWX
                                paths (REF = a directory the full rebuild made, e.g. the served pipe/w_dec8ts).

Reading aned's cache needs read access to it (README: Requirements, root and SIP). The package -> HWX lookup is
m6_hwx_inspect.hwx_paths (Core AI cache manifest -> ANERegionsHash -> aned path) for the executable name "python",
the one the server and forge.py run as.
    python gpu_tables_kit.py pack|install|verify ..."""
import hashlib, json, os, shutil, struct, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for cand in (os.environ.get("V02", ""), str(HERE.parents[2]), "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02"):
    if cand and (Path(cand) / "scripts/m6_hwx_inspect.py").exists():
        sys.path.insert(0, str(Path(cand) / "scripts"))
        break
import m6_hwx_inspect as HI  # noqa: E402

KIT_VERSION = 1


def kern(p):
    """(file offset, size) of the HWX's __KERN segment: the compiled weights (as prep_engine.kern)."""
    b = open(p, "rb").read(65536)
    ncmds, off = struct.unpack_from("<I", b, 16)[0], 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<2I", b, off)
        if cmd == 0x19 and b[off + 8:off + 24].rstrip(b"\0").decode().startswith("__KERN"):
            return tuple(struct.unpack_from("<4Q", b, off + 24)[2:])
        off += size
    raise ValueError(f"{p}: no __KERN segment")


def sha_file(p, off=0, size=None, bs=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        f.seek(off)
        left = size
        while left is None or left > 0:
            b = f.read(bs if left is None else min(bs, left))
            if not b:
                break
            h.update(b)
            if left is not None:
                left -= len(b)
    return h.hexdigest()


def local_hwx(build: Path, pkg: str) -> str:
    ps = HI.hwx_paths(build / pkg)
    if len(ps) != 1:
        raise SystemExit(f"{pkg}: {len(ps)} compiled programs found for this OS build (compile the build first: "
                         f"forge.py compile; aned's cache must be readable)")
    return str(ps[0])


def pkg_order(build: Path):
    """The packages in the order prep_engine meets them (layer order): the HWX list order of its manifest."""
    chunks = sorted(json.loads((build / "manifest.json").read_text())["chunks"], key=lambda c: c["layers"][0])
    seen = []
    for c in chunks:
        if c["file"] not in seen:
            seen.append(c["file"])
    return seen


def strip(m):
    m = json.loads(json.dumps(m))
    m["dir"] = ""
    m["hwx"] = [{"off": h["off"], "size": h["size"]} for h in m["hwx"]]
    return m


def pack(wdir: Path, build: Path, kit: Path):
    best = json.loads((wdir / "manifest_best.json").read_text())
    base = json.loads((wdir / "manifest.json").read_text())
    pkgs = pkg_order(build)
    assert len(pkgs) == len(best["hwx"]), (len(pkgs), len(best["hwx"]))
    assert base["hwx"] == best["hwx"]
    hwx = []
    for pkg, h in zip(pkgs, best["hwx"]):
        p = local_hwx(build, pkg)
        assert p == h["path"], f"{pkg}: the manifest's HWX {h['path']} is not this package's ({p})"
        assert kern(p) == (h["off"], h["size"]), (pkg, kern(p), h)
        hwx.append({"package": pkg, "off": h["off"], "size": h["size"], "sha256": sha_file(p, h["off"], h["size"]),
                    "os_build": HI.os_build()})
        print(f"{pkg}: weight section {h['size'] / 2**30:.2f} GiB sha256 {hwx[-1]['sha256'][:16]}", flush=True)
    kit.mkdir(parents=True, exist_ok=True)
    files = {}
    for f in sorted(wdir.iterdir()):
        if f.name.startswith("manifest") or f.suffix in (".log",) or not f.is_file():
            continue
        shutil.copy2(f, kit / f.name)
        files[f.name] = sha_file(kit / f.name)
    (kit / "manifest.json").write_text(json.dumps(strip(base), indent=1))
    (kit / "manifest_best.json").write_text(json.dumps(strip(best), indent=1))
    info = {"kit_version": KIT_VERSION, "build": build.name, "source": wdir.name, "hwx": hwx, "files": files,
            "lmax": best["Lmax"], "layers": len(best["layers"])}
    (kit / "kit.json").write_text(json.dumps(info, indent=1))
    print(f"kit {kit}: {len(files)} files, {len(hwx)} programs")


def install(kit: Path, build: Path, out: Path):
    info = json.loads((kit / "kit.json").read_text())
    assert info["kit_version"] == KIT_VERSION
    pkgs = pkg_order(build)
    assert pkgs == [h["package"] for h in info["hwx"]], f"build {build} packages {pkgs} != the kit's"
    paths, bad = [], []
    for h in info["hwx"]:
        p = local_hwx(build, h["package"])
        ko = kern(p)
        s = sha_file(p, *ko) if ko == (h["off"], h["size"]) else None
        ok = s == h["sha256"]
        where = "" if ko == (h["off"], h["size"]) else f" (offset, size {ko} vs the kit's {(h['off'], h['size'])})"
        print(f"{h['package']}: {p} weight section {'MATCH' if ok else 'DIFFERS'}{where}", flush=True)
        paths.append(p)
        if not ok:
            bad.append(h["package"])
    if bad:
        print(f"{len(bad)} of {len(info['hwx'])} compiled programs differ from the kit's (kit OS build "
              f"{info['hwx'][0]['os_build']}, this Mac {HI.os_build()}): the shipped .tab offsets do not apply. "
              f"Rebuild the tables from the export: build_gpu_tables.sh (README).")
        sys.exit(2)
    out.mkdir(parents=True, exist_ok=True)
    for name, sha in info["files"].items():
        shutil.copy2(kit / name, out / name)
        assert sha_file(out / name) == sha, f"{name}: sha256 mismatch (incomplete download?)"
    for name in ("manifest.json", "manifest_best.json"):
        m = json.loads((kit / name).read_text())
        m["dir"] = str(out.resolve())
        m["hwx"] = [{"path": p, "off": h["off"], "size": h["size"]} for p, h in zip(paths, info["hwx"])]
        (out / name).write_text(json.dumps(m, indent=1))
    print(f"installed {len(info['files'])} table files into {out}; server: --v8-gpu-manifest {out}/manifest_best.json "
          f"(launcher: GM={out}/manifest_best.json)")


def verify(out: Path, ref: Path):
    names = sorted(f.name for f in ref.iterdir() if f.is_file() and not f.name.startswith("manifest") and f.suffix != ".log")
    mine = sorted(f.name for f in out.iterdir() if f.is_file() and not f.name.startswith("manifest") and f.suffix != ".log")
    assert names == mine, f"file sets differ: only in ref {sorted(set(names) - set(mine))[:5]}, only in out {sorted(set(mine) - set(names))[:5]}"
    diff = [n for n in names if sha_file(out / n) != sha_file(ref / n)]
    for name in ("manifest.json", "manifest_best.json"):
        a, b = json.loads((out / name).read_text()), json.loads((ref / name).read_text())
        if [h["path"] for h in a["hwx"]] != [h["path"] for h in b["hwx"]]:
            print(f"{name}: HWX paths differ (expected on another Mac)")
        if strip(a) != strip(b):
            diff.append(name)
    print(f"verify {out} vs {ref}: {len(names)} table files, {'all identical' if not diff else f'{len(diff)} differ: {diff[:8]}'}")
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    cmd, args = sys.argv[1], [Path(a) for a in sys.argv[2:]]
    {"pack": pack, "install": install, "verify": verify}[cmd](*args)
