"""Stage the SPLIT+GPU build (v8_dec8) for the Hugging Face repo anemll/anemll-forge-qwen3.8-27B-V8-96K (17 Nov 2026), in the
existing staging folder (real directories, file symlinks into the build, so nothing is copied):
  coreai/m6-96k-dec8/   the 8 packages (+ .info.json), head_T8.aimodel (from coreai_seg2), manifest.json and
                        manifest_files.json with the local paths reduced to names (export, head.file, v8_shared.from,
                        dec8.from) as the 16 Nov staging did; the check outputs (dec8_*.json, *.npz) stay out
  code/anemll-forge/    the git archive of branch v8-gpu-ane-prefill at COMMIT (files extracted over the old snapshot)
  README.md             the model card (research/v8_prefill/hf/README.md with COMMIT filled in)
  MODIFICATIONS.md      + a SPLIT+GPU section
The old coreai/m6-96k-v8 folder is left in place (removing it from the Hub is a separate step).
    python3 hf_stage_dec8.py COMMIT"""
import json, os, subprocess, sys
from pathlib import Path

COMMIT = sys.argv[1]
S = Path("/Volumes/SSD4TB/anemll-forge-research/hf_stage_v8_96k")
B = Path("/Volumes/SSD4TB/anemll-forge-research/v8_dec8")
HEAD = Path("/Volumes/SSD4TB/anemll-forge-research/coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8/head_T8.aimodel")
REPO = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8")
OUT = S / "coreai/m6-96k-dec8"


def mirror(src: Path, dst: Path):
    """Real directories, a symlink per file."""
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

code = S / "code/anemll-forge"
code.mkdir(parents=True, exist_ok=True)
arch = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", COMMIT], capture_output=True, check=True).stdout
subprocess.run(["tar", "-x", "-C", str(code)], input=arch, check=True)

card = (REPO / "research/v8_prefill/hf/README.md").read_text()
assert "`COMMIT`" in card
(S / "README.md").write_text(card.replace("`COMMIT`", f"`{COMMIT}`"))

mod = S / "MODIFICATIONS.md"
t = mod.read_text()
sec = f"""
## SPLIT+GPU decode build (this repository, 17 November 2026)

`coreai/m6-96k-dec8/` replaces the 16 four-layer packages with 8 packages of 8 layers built from the same release 0.2
export (`v8/build_dec8_chunk.py` at `{COMMIT}`). Each package carries, on one copy of its weights, the target's verify
and prefill entries and the V8 prefill entries of its two four-layer groups (`g0_` / `g1_`), plus three decode entries
that split each verification step around its attention layers: `dva` (the layers up to the attention layer's q / k / v
projections), `dvba` (the rest of that layer fused with the next group's part A) and `dvb` (the rest). They bind no KV
cache and no mask; the GPU computes the attention over the KV history between them. The weights, quantization, cache
format and numerics of every other entry are unchanged.
"""
if "SPLIT+GPU decode build" not in t:
    i = t.find("## Retaining notices")
    t = t[:i] + sec.lstrip("\n") + "\n" + t[i:] if i >= 0 else t + sec
    mod.write_text(t)
print(f"staged {files} files in {OUT}; code snapshot {COMMIT}; README and MODIFICATIONS updated")
