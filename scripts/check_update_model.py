#!/usr/bin/env python3
"""Check an installed bundle against a pinned Hub inventory; never change model files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys

from hf_release import DEFAULT_REPO, MANIFEST, load_manifest, validate_manifest


def remote_manifest(repo, revision):
    from huggingface_hub import HfApi, hf_hub_download
    commit = HfApi().repo_info(repo_id=repo, repo_type="model", revision=revision).sha
    if not commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Hub did not return a resolved commit revision")
    path = hf_hub_download(repo_id=repo, filename=MANIFEST, revision=commit)
    return commit, validate_manifest(json.loads(Path(path).read_text()))


def compare(installed, latest, runtime):
    """Compare advertised hashes, without reading gigabytes of installed weights."""
    if runtime not in installed["runtimes"]:
        raise ValueError(f"Installed bundle has no {runtime} runtime")
    if runtime not in latest["runtimes"]:
        raise ValueError(f"Latest bundle has no {runtime} runtime")
    if installed["upstream_model"]["id"] != latest["upstream_model"]["id"]:
        raise ValueError("Remote repository describes a different upstream model")
    groups = {"model", runtime} | ({"drafter"} if runtime == "coreai" else set())
    def inventory(man, components):
        return {f["path"]: (f["bytes"], f["sha256"]) for f in man["files"] if f["component"] in components}
    before, after = inventory(installed, groups), inventory(latest, groups)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    metadata_changed = installed != latest
    status = "model_update_available" if changed else "metadata_update_available" if metadata_changed else "up_to_date"
    return dict(status=status, model_update_available=bool(changed), changed_model_files=changed,
                installed_export=(installed.get("drafter") or {}).get("target_export"),
                latest_export=(latest.get("drafter") or {}).get("target_export"),
                latest_contexts=latest["runtimes"][runtime]["contexts"],
                comparison="release inventories; local file integrity is not checked")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path(os.environ.get("FORGE_BUNDLE", "~/Models/anemll-forge-qwen3.8-27B")),
                   help="installed bundle (default: FORGE_BUNDLE or ~/Models/anemll-forge-qwen3.8-27B)")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--revision", default="main")
    p.add_argument("--runtime", choices=("coreai", "coreml"), default="coreai")
    p.add_argument("--json", action="store_true", help="machine-readable result; errors exit 1, completed checks exit 0")
    a = p.parse_args(argv)
    try:
        bundle = a.bundle.expanduser().resolve()
        installed = load_manifest(bundle)
        commit, latest = remote_manifest(a.repo, a.revision)
        result = compare(installed, latest, a.runtime)
        result.update(bundle=str(bundle), repo=a.repo, revision=commit)
        # Select a fresh destination, including when a previous update is still on disk.
        output = bundle.with_name(f"{bundle.name}-{commit[:8]}")
        suffix = 2
        while output.exists():
            output = bundle.with_name(f"{bundle.name}-{commit[:8]}-{suffix}")
            suffix += 1
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / "forge.py"), "download",
                   "--repo", a.repo, "--revision", commit, "--runtime", a.runtime, "--output", str(output)]
        if result["status"] != "up_to_date":
            result["download_command"] = shlex.join(command)
        if a.json:
            print(json.dumps(result, indent=2))
        else:
            labels = {"up_to_date": "Your model bundle is up to date.",
                      "model_update_available": "A model update is available.",
                      "metadata_update_available": "Model files are current; release metadata or documentation changed."}
            print(labels[result["status"]])
            print(f"Repository: {a.repo}@{commit}")
            if result["model_update_available"]:
                print(f"Weights: {result['installed_export'] or 'not recorded'} -> {result['latest_export'] or 'not recorded'}")
                print(f"Changed model files: {len(result['changed_model_files'])}")
            if "download_command" in result:
                print("Download the complete matching bundle into a new directory:")
                print(result["download_command"])
                print("Stop the server before switching FORGE_BUNDLE; run forge.py quick-test on the new bundle first.")
            print("This compares release inventories. Use forge.py quick-test --check-only to verify local files.")
        return 0
    except Exception as exc:
        if a.json:
            print(json.dumps({"status": "error", "error": str(exc)}))
        else:
            print(f"Model update check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
