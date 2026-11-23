"""Stage the tree8 + sliced build (v8_dec8ts, served since 22 Nov 2026) for the Hugging Face repo
anemll/anemll-forge-qwen3.8-27B-V8-96K, in the existing staging folder (real directories, a symlink per file into the
build, so nothing is copied; the upload follows the links). Local only: no upload.
  coreai/m6-96k-dec8ts/  the 8 packages (+ .info.json), head_T8.aimodel (from coreai_seg2), manifest.json and
                         manifest_files.json with every local path reduced to its name (export, head.file,
                         v8_shared.from, dec8.from), as hf_stage_dec8.py did for m6-96k-dec8
  gpu/w_dec8ts/          the GPU-table kit (pipe/gpu_tables_kit.py pack: real files, sha256 per file and per HWX)
  code/anemll-forge/     the git archive of COMMIT, extracted into an empty folder (the previous snapshot is moved to
                         OLD_CODE outside the staging folder, not deleted)
The earlier coreai/m6-96k-v8 and coreai/m6-96k-dec8 folders are left in place (removing them from the Hub is a
separate decision). The model card, MODIFICATIONS.md and DEPLOY.md are written by hand from research/v8_prefill/hf/.
    python3 hf_stage_dec8ts.py COMMIT [--no-kit]"""
import json, os, shutil, subprocess, sys
from pathlib import Path

COMMIT = sys.argv[1]
S = Path("/Volumes/SSD4TB/anemll-forge-research/hf_stage_v8_96k")
B = Path("/Volumes/SSD4TB/anemll-forge-research/v8_dec8ts")
HEAD = Path("/Volumes/SSD4TB/anemll-forge-research/coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8/head_T8.aimodel")
REPO = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8")
WT = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_dec8ts")
OLD_CODE = Path("/Volumes/SSD4TB/anemll-forge-research/hf_stage_v8_96k_old_code")
OUT = S / "coreai/m6-96k-dec8ts"


def mirror(src: Path, dst: Path):
    n = 0
    for root, _, files in os.walk(src):
        r = Path(root)
        d = dst / r.relative_to(src)
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            link = d / f
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(r / f)
            n += 1
    return n


OUT.mkdir(parents=True, exist_ok=True)
files = 0
for pkg in sorted(B.glob("chunk_L*.aimodel")):
    files += mirror(pkg, OUT / pkg.name)
    info = B / (pkg.name + ".info.json")
    if info.exists():
        link = OUT / info.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(info)
        files += 1
files += mirror(HEAD, OUT / HEAD.name)
for name in ("manifest.json", "manifest_files.json"):
    m = json.loads((B / name).read_text())
    m["export"] = Path(m["export"]).name
    m["head"]["file"] = HEAD.name
    for k in ("v8_shared", "dec8"):
        if isinstance(m.get(k), dict) and str(m[k].get("from", "")).startswith("/"):
            m[k]["from"] = Path(m[k]["from"]).name
    s = json.dumps(m, indent=1)
    assert '"/' not in s, f"absolute path left in {name}"
    (OUT / name).write_text(s)
    files += 1
print(f"build: {files} files in {OUT}", flush=True)

if "--no-kit" not in sys.argv:
    kit = S / "gpu/w_dec8ts"
    subprocess.run([sys.executable, str(REPO / "research/v8_prefill/pipe/gpu_tables_kit.py"), "pack", str(WT), str(B), str(kit)],
                   check=True)

code = S / "code/anemll-forge"
if code.exists():
    OLD_CODE.mkdir(parents=True, exist_ok=True)
    dst = OLD_CODE / f"anemll-forge.{len(list(OLD_CODE.iterdir()))}"
    shutil.move(str(code), str(dst))
    print(f"previous code snapshot moved to {dst}")
code.mkdir(parents=True)
arch = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", COMMIT], capture_output=True, check=True).stdout
subprocess.run(["tar", "-x", "-C", str(code)], input=arch, check=True)
print(f"code snapshot {COMMIT}: {sum(1 for _ in code.rglob('*') if _.is_file())} files")
