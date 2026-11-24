"""Stage the super-adapter tree build (v8_dec8ts_sa, 24 Nov 2026) for the Hugging Face repo
anemll/anemll-forge-qwen3.8-27B-V8-96K-SuperAdapter (private first). Local only: no upload. Like v8/hf_stage_dec8ts.py:
real directories with a symlink per file (the upload follows the links), path-stripped manifests, the GPU-table kit
packed (gpu_tables_kit.py pack), the code snapshot = git archive of COMMIT. Plus the V8-96K repo's drafter, model files,
LICENSE / QWEN_SOURCE.json (unchanged), NOTICE and MODIFICATIONS.md with an adapter section, adapter/ (the upstream
adapter_config.json, ADAPTER_SOURCE.json, adapter_export.json), and the card (hf/README_superadapter.md, COMMIT and
ADAPTER_SHA filled in), DEPLOY.md and AGENTS.md from the snapshot.
    python3 hf_stage_sa.py COMMIT ADAPTER_SHA [--no-kit]"""
import json, os, shutil, subprocess, sys
from pathlib import Path

COMMIT, ADAPTER_SHA = sys.argv[1], sys.argv[2]
R = Path("/Volumes/SSD4TB/anemll-forge-research")
S = R / "hf_stage_v8_96k_sa"
BASE = R / "hf_stage_v8_96k"
B = R / "v8_dec8ts_sa"
WT = R / "v8_dec8ts_sa_w"
EXPORT = R / "exports/release_vq3pA_sa_p64"
AD = R / "adapters/qwen38-27b-super-adapter"
HEAD = R / "coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8/head_T8.aimodel"
REPO = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8")
OUT = S / "coreai/m6-96k-dec8ts-sa"


def link(dst: Path, src: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def mirror(src: Path, dst: Path):
    n = 0
    for root, _, files in os.walk(src, followlinks=True):
        r = Path(root)
        for f in files:
            link(dst / r.relative_to(src) / f, r / f)
            n += 1
    return n


S.mkdir(parents=True, exist_ok=True)
files = 0
for pkg in sorted(B.glob("chunk_L*.aimodel")):
    files += mirror(pkg, OUT / pkg.name)
    info = B / (pkg.name + ".info.json")
    if info.exists():
        link(OUT / info.name, info); files += 1
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
    (OUT / name).write_text(s); files += 1
print(f"build: {files} files in {OUT}", flush=True)

if "--no-kit" not in sys.argv:
    kit = S / "gpu/w_dec8ts_sa"
    if kit.exists():
        shutil.rmtree(kit)
    subprocess.run([sys.executable, str(REPO / "research/v8_prefill/pipe/gpu_tables_kit.py"), "pack", str(WT), str(B), str(kit)],
                   check=True)

for sub in ("drafter", "model"):
    print(sub, mirror(BASE / sub, S / sub), "files")
for name in ("LICENSE", "QWEN_SOURCE.json", "config.json"):
    link(S / name, BASE / name)

ad = S / "adapter"
ad.mkdir(exist_ok=True)
shutil.copy(AD / "adapter_config.json", ad / "adapter_config.json")
exp = json.loads((EXPORT / "adapter_export.json").read_text())
exp["src"], exp["adapter"] = Path(exp["src"]).name, "harness-hub/Qwen3.8-27B-super-adapter"
(ad / "adapter_export.json").write_text(json.dumps(exp, indent=1))
cfg = json.loads((AD / "adapter_config.json").read_text())
(ad / "ADAPTER_SOURCE.json").write_text(json.dumps({
    "upstream_adapter": {"id": "harness-hub/Qwen3.8-27B-super-adapter", "revision": ADAPTER_SHA,
                         "url": f"https://huggingface.co/harness-hub/Qwen3.8-27B-super-adapter/tree/{ADAPTER_SHA}",
                         "license": "apache-2.0", "base_model": cfg.get("base_model_name_or_path"),
                         "peft": {k: cfg.get(k) for k in ("r", "lora_alpha", "target_modules", "use_rslora", "use_dora")},
                         "description": "pi, Hermes Agent and OpenCode RL adapters concatenated along the rank at weight 1/2 "
                                        "(blog: https://huggingface.co/blog/harness-adapters-at-scale)"},
    "conversion": {"tool": "research/v8_prefill/adapter/adapter_export.py", "code_commit": COMMIT,
                   "form": "fp16 low-rank side paths lr_a = (alpha / r) B, lr_b = A next to the unchanged 3-bit weights; "
                           "attention appended to the release's rank-64 corrections and zero-padded to rank 192",
                   "export": EXPORT.name, "base_export": "release_vq3pA_mixh_s600_k1_mat"},
    "redistributed": "converted factors only (inside the Core AI packages and the GPU tables); the upstream adapter "
                     "weights are not included"}, indent=1))

code = S / "code/anemll-forge"
if code.exists():
    shutil.rmtree(code)
code.mkdir(parents=True)
arch = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", COMMIT], capture_output=True, check=True).stdout
subprocess.run(["tar", "-x", "-C", str(code)], input=arch, check=True)
H = code / "research/v8_prefill/hf"
card = (H / "README_superadapter.md").read_text().replace("COMMIT", COMMIT[:7]).replace("ADAPTER_SHA", ADAPTER_SHA[:10])
assert "COMMIT" not in card and "ADAPTER_SHA" not in card
(S / "README.md").write_text(card)
for name in ("DEPLOY.md", "AGENTS.md"):
    shutil.copy(H / name, S / name)
(S / "NOTICE").write_text((BASE / "NOTICE").read_text() + "\n" + (code / "research/v8_prefill/adapter/NOTICE_ADAPTER.txt").read_text())
(S / "MODIFICATIONS.md").write_text((BASE / "MODIFICATIONS.md").read_text().rstrip() + "\n\n"
                                    + (code / "research/v8_prefill/adapter/MODIFICATIONS_ADAPTER.md").read_text())
print(f"code snapshot {COMMIT[:7]}: {sum(1 for _ in code.rglob('*') if _.is_file())} files; card, DEPLOY, AGENTS, NOTICE, "
      f"MODIFICATIONS written -> {S}")
