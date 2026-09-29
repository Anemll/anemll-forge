#!/usr/bin/env python3
"""Build release manifests, download pinned HF bundles, and run a short smoke test."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import time

COMMANDS = {"release-manifest", "download", "quick-test"}
MANIFEST = "release.json"
DEFAULT_REPO = "anemll/anemll-forge-qwen3.8-27B"
DEFAULT_DRAFTER = "dflash2_lut4_gptq.aimodel"
SELECTOR_FILE = "selector.safetensors"
SELECTOR_KEYS = ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook")
ROOT_DOCUMENTS = {"README.md", "LICENSE", "NOTICE", "MODIFICATIONS.md", "QWEN_SOURCE.json"}
MODIFICATION_NOTICE = "Converted/quantized by ANEMLL; see MODIFICATIONS.md for details."
TEXT_SUFFIXES = {".json", ".md", ".txt", ".yaml", ".yml", ".toml", ".xml", ".plist"}

MODEL_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "generation_config.json", "chat_template.jinja", "chat_template.json", "vocab.json",
               "merges.txt", "tokenizer.model", "added_tokens.json", "embed_tokens_fp16.npy"}


def requires_modification_notice(name, component):
    if Path(name).name in {"LICENSE", "NOTICE"}:
        return False
    return (name == "model/embed_tokens_fp16.npy" or
            (component in {"coreai", "coreml", "export", "drafter"} and Path(name).suffix.lower() not in TEXT_SUFFIXES))

def add_commands(sub):
    p = sub.add_parser("release-manifest", help="hash a prepared HF upload directory")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--model-id", required=True, help="canonical upstream checkpoint ID")
    p.add_argument("--model-revision", required=True, help="upstream checkpoint revision used")
    p.add_argument("--plain", action="store_true", help="diagnostic target-only bundle; release requires DFlash2")
    p = sub.add_parser("download", help="download and verify one runtime from Hugging Face")
    p.add_argument("--repo", default=DEFAULT_REPO, help="HF owner/repository (default: %(default)s)")
    p.add_argument("--revision", default="main", help="tag/commit/branch, resolved to a commit before download")
    p.add_argument("--runtime", choices=("coreai", "coreml"), default="coreai")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include-export", action="store_true", help="also download optional quantized conversion weights")
    p.add_argument("--plain", action="store_true", help="diagnostic target-only download, omitting DFlash2")
    p = sub.add_parser("quick-test", help="verify a bundle and test short DFlash2 speculative generation")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--runtime", choices=("coreai", "coreml"), default="coreai")
    p.add_argument("--ctx", type=int, help="default: smallest supported context")
    p.add_argument("--check-only", action="store_true", help="integrity/layout check without loading the model")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--report", type=Path, help="optional JSON test report")
    p.add_argument("--plain", action="store_true", help="diagnostic plain target generation instead of DFlash2")


def relative(value):
    if not isinstance(value, str) or not value or "\\" in value or any(c in value for c in "*?[]"):
        raise ValueError(f"Invalid bundle path: {value!r}")
    p = PurePosixPath(value)
    if not p.parts or p.is_absolute() or any(x in ("", ".", "..") for x in p.parts) or p.as_posix() != value:
        raise ValueError(f"Bundle paths must be canonical relative paths: {value!r}")
    return p


def local(root, name):
    p = root.joinpath(*relative(name).parts)
    if not p.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes bundle directory: {name}")
    return p


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_manifest(m):
    if not isinstance(m, dict) or type(m.get("schema_version")) is not int or m["schema_version"] != 1:
        raise ValueError("Unsupported release.json schema; expected schema_version 1")
    upstream = m.get("upstream_model")
    if not isinstance(upstream, dict) or any(not isinstance(upstream.get(k), str) or not upstream[k].strip()
                                            for k in ("id", "revision")):
        raise ValueError("Record the upstream checkpoint ID and revision in release.json")
    runtimes = m.get("runtimes")
    if not isinstance(runtimes, dict) or not runtimes or set(runtimes) - {"coreml", "coreai"}:
        raise ValueError("release.json must describe coreml and/or coreai runtimes")
    roots = {"model": str(relative(m.get("model_path")))}
    for name, entry in runtimes.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("contexts"), list):
            raise ValueError(f"Invalid runtime metadata: {name}")
        contexts = entry["contexts"]
        if not contexts or any(type(c) is not int or c <= 0 for c in contexts) or len(set(contexts)) != len(contexts):
            raise ValueError(f"Invalid runtime contexts: {name}")
        roots[name] = str(relative(entry.get("path")))
    if m.get("export_path"):
        roots["export"] = str(relative(m["export_path"]))
    if m.get("drafter") is not None:
        d = m["drafter"]
        if not isinstance(d, dict) or d.get("runtime") != "coreai" or "coreai" not in runtimes:
            raise ValueError("DFlash2 requires a Core AI target and drafter runtime")
        roots["drafter"] = str(relative(d.get("path")))
        for key in ("model", "metadata", "config", "codebooks"):
            relative(d.get(key))
        if d["model"] != DEFAULT_DRAFTER or d["metadata"] != Path(DEFAULT_DRAFTER).with_suffix(".json").as_posix():
            raise ValueError("Release must use the tested dflash2_lut4_gptq.aimodel and matching sidecar")
        if d["config"] != "config.json" or d["codebooks"] != SELECTOR_FILE:
            raise ValueError("Drafter requires config.json and compact selector.safetensors")
        for key in ("target_export", "head_export"):
            if len(relative(d.get(key)).parts) != 1:
                raise ValueError("Drafter target/head identities must be export names")
        taps = d.get("taps")
        if not isinstance(taps, list) or not taps or any(type(i) is not int or not 0 <= i < 64 for i in taps):
            raise ValueError("Invalid drafter target taps")
    paths = list(roots.values())
    if any(a == b or a.startswith(b + "/") or b.startswith(a + "/")
           for i, a in enumerate(paths) for b in paths[i+1:]):
        raise ValueError("Bundle component paths must be distinct and must not overlap")
    files, seen = m.get("files"), set()
    if not isinstance(files, list) or not files:
        raise ValueError("release.json has no file inventory")
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Invalid file entry")
        name = str(relative(entry.get("path")))
        group = entry.get("component")
        if group == "documentation":
            if name not in ROOT_DOCUMENTS:
                raise ValueError(f"Unexpected root documentation file: {name}")
        elif group not in roots or not name.startswith(roots[group] + "/"):
            raise ValueError(f"File is outside its component: {name}")
        if (name == roots["model"] + "/embed_tokens_fp16.npy" or requires_modification_notice(name, group)):
            if not isinstance(entry.get("modification_notice"), str) or not entry["modification_notice"].strip():
                raise ValueError(f"Missing modification notice: {name}")
        if name in seen:
            raise ValueError(f"Duplicate file entry: {name}")
        seen.add(name)
        if type(entry.get("bytes")) is not int or entry["bytes"] < 0:
            raise ValueError(f"Invalid byte count: {name}")
        if not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError(f"Invalid SHA256: {name}")
    for required in sorted(ROOT_DOCUMENTS):
        if not any(f["path"] == required and f["component"] == "documentation" for f in files):
            raise ValueError(f"Missing release documentation: {required}")
    for required in ("config.json", "tokenizer.json", "tokenizer_config.json", "embed_tokens_fp16.npy"):
        if roots["model"] + "/" + required not in seen:
            raise ValueError(f"Missing model file in inventory: {required}")
    for runtime in runtimes:
        if not any(f["component"] == runtime for f in files):
            raise ValueError(f"No runtime files: {runtime}")
    if "drafter" in roots:
        d = m["drafter"]
        for key in ("metadata", "config", "codebooks"):
            if roots["drafter"] + "/" + d[key] not in seen:
                raise ValueError(f"Missing drafter file in inventory: {d[key]}")
        prefix = roots["drafter"] + "/" + d["model"] + "/"
        if not any(name.startswith(prefix) for name in seen):
            raise ValueError("Drafter package is absent from inventory")
    return m


def load_manifest(root):
    return validate_manifest(json.loads((root / MANIFEST).read_text()))


def model_config(root, m):
    c = json.loads(local(root, m["model_path"] + "/config.json").read_text()).get("text_config", {})
    if (c.get("num_hidden_layers") != 64 or c.get("hidden_size") != 5120
            or type(c.get("vocab_size")) is not int or c["vocab_size"] <= 0):
        raise ValueError("Expected Qwen research config: 64 layers, hidden_size 5120, explicit vocab_size")
    return c


def check_embedding(p, config):
    # Read only the .npy header; a multi-GB embedding never needs to enter RAM here.
    with p.open("rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError("Embedding is not a NumPy .npy file")
        version = f.read(2)
        if version == b"\x01\x00":
            size, fmt = 2, "<H"
        elif version in (b"\x02\x00", b"\x03\x00"):
            size, fmt = 4, "<I"
        else:
            raise ValueError(f"Unsupported NumPy header version: {version!r}")
        length = f.read(size)
        if len(length) != size:
            raise ValueError("Truncated embedding header")
        header_size = struct.unpack(fmt, length)[0]
        if header_size > 65536:
            raise ValueError("Embedding header is too large")
        try:
            header = ast.literal_eval(f.read(header_size).decode("utf-8" if version[0] == 3 else "latin1"))
        except (ValueError, SyntaxError, UnicodeError) as e:
            raise ValueError("Invalid embedding header") from e
        offset = f.tell()
    shape = (config["vocab_size"], config["hidden_size"])
    if not isinstance(header, dict) or header.get("shape") != shape or header.get("descr") not in ("<f2", "=f2"):
        raise ValueError(f"Embedding must be float16 with shape {shape}")
    if header.get("fortran_order") is not False or p.stat().st_size != offset + shape[0] * shape[1] * 2:
        raise ValueError("Embedding layout or file length is invalid")


def referenced_asset(root, m, runtime, name):
    relative(name)
    prefix = m["runtimes"][runtime]["path"] + "/" + name
    entries = [f for f in m["files"] if f["path"].startswith(prefix + "/") or f["path"] == prefix]
    if not entries or not local(root, prefix).exists():
        raise ValueError(f"Runtime asset missing from inventory/bundle: {prefix}")


def check_chunks(manifest):
    chunks = manifest.get("chunks")
    if manifest.get("T") != 8 or not isinstance(chunks, list) or not chunks:
        raise ValueError("Expected a T=8 runtime with chunks covering all 64 layers")
    next_layer = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Invalid chunk metadata")
        layers = chunk.get("layers")
        if (not isinstance(layers, list) or len(layers) != 2 or any(type(x) is not int for x in layers)
                or layers[0] != next_layer or not layers[0] <= layers[1] < 64):
            raise ValueError("Chunk ranges must cover layers 0–63 in order without gaps or overlaps")
        relative(chunk.get("file"))
        next_layer = layers[1] + 1
    if next_layer != 64:
        raise ValueError("Chunk ranges must cover all 64 layers")
    return chunks


def check_layout(root, m, runtime):
    c = model_config(root, m)
    check_embedding(local(root, m["model_path"] + "/embed_tokens_fp16.npy"), c)
    info = m["runtimes"][runtime]
    build = local(root, info["path"])
    contexts = info["contexts"]
    inventory = {f["path"] for f in m["files"] if f["component"] == runtime}
    # The runtime may prefer a sibling .aimodelc. Never load added, unhashed packages.
    for p in build.rglob("*"):
        if p.is_symlink():
            raise ValueError(f"Runtime bundle must not contain symlinks: {p}")
        if p.is_file() and p.relative_to(root).as_posix() not in inventory:
            raise ValueError(f"Runtime file is not inventoried: {p.relative_to(root)}")
    if runtime == "coreai":
        name = info["path"] + "/manifest.json"
        if name not in {f["path"] for f in m["files"]}:
            raise ValueError("Core AI runtime manifest is absent from inventory")
        manifest = json.loads((build / "manifest.json").read_text())
        if sorted(manifest.get("ctxs", [])) != sorted(contexts):
            raise ValueError("Core AI context metadata disagrees with runtime manifest")
        chunks = check_chunks(manifest)
        head = manifest.get("head")
        if not isinstance(head, dict):
            raise ValueError("Invalid Core AI head metadata")
        assets = [x["file"] for x in chunks] + [head.get("file", "")]
        for chunk in chunks:
            entries = chunk.get("entries", [])
            if any(f"v8_{ctx // 1024}k" not in entries for ctx in contexts):
                raise ValueError("Core AI chunk lacks a verify entry for a declared context")
        for package in [*chunks, head]:
            compiled = package.get("compiled")
            if compiled is not None:
                assets.append(str(relative(compiled)))
            else:
                sibling = relative(package.get("file")).with_suffix(".aimodelc").as_posix()
                if local(build, sibling).exists():
                    assets.append(sibling)
    else:
        assets = []
        for ctx in contexts:
            name = f"manifest_ctx{ctx}_v4.json"
            if info["path"] + "/" + name not in {f["path"] for f in m["files"]}:
                raise ValueError(f"Missing Core ML runtime manifest: {name}")
            if (build / f"manifest_ctx{ctx}_v5.json").exists():
                raise ValueError("Competing v5 manifest: first release path expects v4")
            manifest = json.loads((build / name).read_text())
            if manifest.get("ctx") != ctx or manifest.get("version") != 4:
                raise ValueError(f"Core ML context/version mismatch: {name}")
            chunks = check_chunks(manifest)
            assets.extend([x["file"] for x in chunks] + [manifest.get("head", "")])
    for name in assets:
        referenced_asset(root, m, runtime, name)
    return c


def check_selector(p, vocab, rank):
    """Validate the compact safetensors layout without loading its large codebooks."""
    with p.open("rb") as f:
        length = f.read(8)
        if len(length) != 8:
            raise ValueError("Truncated selector safetensors header")
        size = struct.unpack("<Q", length)[0]
        if not 1 <= size <= 1024 * 1024:
            raise ValueError("Invalid selector safetensors header size")
        try:
            header = json.loads(f.read(size))
        except (ValueError, UnicodeError) as e:
            raise ValueError("Invalid selector safetensors header") from e
    if not isinstance(header, dict) or set(header) - {"__metadata__"} != set(SELECTOR_KEYS):
        raise ValueError("Compact selector must contain only predecessor and successor codebooks")
    spans = []
    for key in SELECTOR_KEYS:
        info = header[key]
        if (not isinstance(info, dict) or info.get("shape") != [vocab, rank]
                or info.get("dtype") not in ("BF16", "F16", "F32")):
            raise ValueError(f"Selector {key} must be floating point with shape [{vocab}, {rank}]")
        offsets = info.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(i) is not int or i < 0 for i in offsets)
                or offsets[1] - offsets[0] != vocab * rank * (4 if info["dtype"] == "F32" else 2)):
            raise ValueError(f"Invalid selector data offsets: {key}")
        spans.append(offsets)
    spans.sort()
    if spans[0][0] != 0 or spans[0][1] != spans[1][0] or p.stat().st_size != 8 + size + spans[1][1]:
        raise ValueError("Selector data must be complete, contiguous and nonoverlapping")


def drafter_paths(build, draft=None, directory=None):
    """Resolve the tested release drafter; never substitute a legacy Core ML/RTN package."""
    pkg = Path(draft).expanduser().resolve() if draft else build.parent / "drafter" / DEFAULT_DRAFTER
    directory = Path(directory).expanduser().resolve() if directory else pkg.parent
    if pkg.suffix != ".aimodel" or not pkg.is_dir():
        raise ValueError(f"Missing Core AI DFlash2 drafter: {pkg}. Supply --draft/--drafter, or --plain for diagnostics.")
    return pkg, directory


def check_drafter_pair(pkg, directory, target, config):
    """Check target/head/taps and runtime support before either full model is loaded."""
    meta = json.loads(pkg.with_suffix(".json").read_text())
    cfg = json.loads((directory / "config.json").read_text())
    if not isinstance(meta, dict) or not isinstance(cfg, dict) or not isinstance(cfg.get("dflash_config"), dict):
        raise ValueError("Invalid DFlash2 metadata/config object")
    dc = cfg.get("dflash_config", {})
    taps = dc.get("target_layer_ids")
    if (target.get("T") != 8 or not isinstance(taps, list) or not taps
            or taps != target.get("taps") or dc.get("block_size") != 8
            or cfg.get("num_target_layers") != 64 or cfg.get("hidden_size") != config.get("hidden_size")
            or cfg.get("vocab_size") != config.get("vocab_size")):
        raise ValueError("DFlash2 target shape, T=8 or feature taps do not match the target manifest/config")
    if (meta.get("T") != 8 or meta.get("R") != 8 or meta.get("RP") != 64
            or meta.get("W") != cfg.get("sliding_window") or meta.get("W") != 2048
            or cfg.get("num_hidden_layers") != 5 or cfg.get("head_dim") != 128
            or cfg.get("num_key_value_heads") != 8):
        raise ValueError("Expected the tested five-layer DFlash2 drafter (T=8/R=8/RP=64/W=2048)")
    target_id = Path(target.get("export", "")).name
    draft_target = Path(meta.get("target_export", "")).name
    head_id = Path(meta.get("head_export", "")).parent.name
    if not target_id or draft_target != target_id or head_id != target_id:
        raise ValueError("DFlash2 target/head export mismatch; use the drafter built with this target's head")
    rank = dc.get("selector_rank")
    if type(rank) is not int or rank != 256:
        raise ValueError("Expected DFlash2 selector_rank 256")
    check_selector(directory / SELECTOR_FILE, cfg["vocab_size"], rank)
    entries = meta.get("entries", {})
    if not {"draft", "ctx64"} <= set(entries) or "logits" not in entries.get("draft", {}).get("outputs", []):
        raise ValueError("Drafter needs draft/ctx64 entries and the matching target head")
    return cfg, meta


def require_drafter(m, runtime, plain=False):
    if not plain and (runtime != "coreai" or not m.get("drafter")):
        raise ValueError("Default release path requires Core AI target + DFlash2 drafter; use --plain only for diagnostics")


def check_drafter(root, m):
    d = m["drafter"]
    directory = local(root, d["path"])
    pkg = local(directory, d["model"])
    inventory = {f["path"] for f in m["files"] if f["component"] == "drafter"}
    for p in directory.rglob("*"):
        if p.is_symlink() or (p.is_file() and p.relative_to(root).as_posix() not in inventory):
            raise ValueError(f"Drafter file is a symlink or not inventoried: {p.relative_to(root)}")
    if not pkg.is_dir():
        raise ValueError("Missing Core AI DFlash2 package")
    target = json.loads(local(root, m["runtimes"]["coreai"]["path"] + "/manifest.json").read_text())
    cfg, meta = check_drafter_pair(pkg, directory, target, model_config(root, m))
    if (d["taps"] != cfg["dflash_config"]["target_layer_ids"]
            or d["target_export"] != Path(meta["target_export"]).name
            or d["head_export"] != Path(meta["head_export"]).parent.name):
        raise ValueError("Release drafter association disagrees with its runtime metadata")
    return cfg, meta


def verify(root, m, runtime, include_export=False, plain=False):
    if runtime not in m["runtimes"]:
        raise ValueError(f"Release does not contain runtime: {runtime}")
    require_drafter(m, runtime, plain)
    groups = {"documentation", "model", runtime} | ({"export"} if include_export else set()) | ({"drafter"} if not plain else set())
    files = [f for f in m["files"] if f["component"] in groups]
    for f in files:
        p = local(root, f["path"])
        if not p.is_file() or p.stat().st_size != f["bytes"]:
            raise ValueError(f"Missing file or size mismatch: {f['path']}")
        if digest(p) != f["sha256"]:
            raise ValueError(f"SHA256 mismatch: {f['path']}")
    check_layout(root, m, runtime)
    if not plain:
        check_drafter(root, m)
    return {"verified_files": len(files), "verified_bytes": sum(f["bytes"] for f in files)}


def make_manifest(a):
    root = a.bundle.expanduser().resolve()
    m = dict(schema_version=1, upstream_model=dict(id=a.model_id, revision=a.model_revision),
             model_path="model", runtimes={}, files=[])
    for name in sorted(ROOT_DOCUMENTS):
        p = root / name
        if p.is_symlink():
            raise ValueError(f"Release documentation must not be a symlink: {name}")
        if not p.is_file():
            raise ValueError(f"Missing release documentation: {name}")
        m["files"].append(dict(path=name, component="documentation", bytes=p.stat().st_size, sha256=digest(p)))
    # Notices describe derived binary artifacts; authors must also put accurate change
    # notices into any modified source files or package metadata they distribute.
    groups = {"model": root / "model"}
    for runtime in ("coreai", "coreml"):
        build = root / runtime
        if not build.is_dir():
            continue
        if runtime == "coreai":
            contexts = json.loads((build / "manifest.json").read_text())["ctxs"]
        else:
            contexts = [int(p.name.split("_")[1][3:]) for p in build.glob("manifest_ctx*_v4.json")]
        m["runtimes"][runtime] = dict(path=runtime, contexts=sorted(contexts))
        groups[runtime] = build
    if (root / "export").is_dir():
        m["export_path"] = "export"
        groups["export"] = root / "export"
    if (root / "drafter").is_dir():
        if "coreai" not in m["runtimes"]:
            raise ValueError("A release drafter requires a Core AI target")
        target = json.loads((root / "coreai/manifest.json").read_text())
        cfg, meta = check_drafter_pair(root / "drafter" / DEFAULT_DRAFTER, root / "drafter", target,
                                       model_config(root, m))
        m["drafter"] = dict(path="drafter", runtime="coreai", model=DEFAULT_DRAFTER,
                            metadata=Path(DEFAULT_DRAFTER).with_suffix(".json").as_posix(),
                            config="config.json", codebooks=SELECTOR_FILE,
                            target_export=Path(meta["target_export"]).name,
                            head_export=Path(meta["head_export"]).parent.name,
                            taps=cfg["dflash_config"]["target_layer_ids"])
        groups["drafter"] = root / "drafter"
    if "coreai" in m["runtimes"]:
        require_drafter(m, "coreai", getattr(a, "plain", False))
    for group, directory in groups.items():
        if directory.is_symlink():
            raise ValueError(f"Upload bundle must not use a symlink component directory: {directory}")
        for p in sorted(directory.rglob("*")):
            if p.is_symlink():
                raise ValueError(f"Upload bundle must be self-contained; copy symlink contents: {p}")
            if not p.is_file() or p.name == ".DS_Store":
                continue
            name = p.relative_to(root).as_posix()
            relative(name)
            if group == "model" and p.relative_to(directory).as_posix() not in MODEL_FILES:
                raise ValueError(f"Unexpected model file; stage only tokenizer/config/embedding assets: {name}")
            entry = dict(path=name, component=group, bytes=p.stat().st_size, sha256=digest(p))
            if requires_modification_notice(name, group):
                entry["modification_notice"] = MODIFICATION_NOTICE
            m["files"].append(entry)
    validate_manifest(m)
    for runtime in m["runtimes"]:
        check_layout(root, m, runtime)
    if m.get("drafter"):
        check_drafter(root, m)
    # Publish the inventory only after all components have passed layout checks.
    tmp = root / "release.json.tmp"
    tmp.write_text(json.dumps(m, indent=2) + "\n")
    tmp.replace(root / MANIFEST)
    print(json.dumps(dict(manifest=str(root / MANIFEST), files=len(m["files"]),
                          bytes=sum(f["bytes"] for f in m["files"])), indent=2))
    return 0


def download(a):
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError as e:
        raise RuntimeError("Install the download dependency: python -m pip install huggingface_hub") from e
    revision = HfApi().repo_info(repo_id=a.repo, repo_type="model", revision=a.revision).sha
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Hub did not return a resolved commit revision")
    manifest_path = hf_hub_download(repo_id=a.repo, filename=MANIFEST, revision=revision)
    m = validate_manifest(json.loads(Path(manifest_path).read_text()))
    if a.runtime not in m["runtimes"]:
        raise ValueError(f"Release does not contain runtime: {a.runtime}")
    plain = getattr(a, "plain", False)
    require_drafter(m, a.runtime, plain)
    if a.include_export and not m.get("export_path"):
        raise ValueError("This release has no optional export component")
    root = a.output.expanduser().resolve()
    old = root / MANIFEST
    if old.exists() and json.loads(old.read_text()) != m:
        raise ValueError("Output contains a different release; use a new --output directory")
    groups = {"documentation", "model", a.runtime} | ({"export"} if a.include_export else set()) | ({"drafter"} if not plain else set())
    names = [f["path"] for f in m["files"] if f["component"] in groups]
    for name in [MANIFEST, *names]:
        local(root, name)  # Reject existing symlink escapes before any download writes.
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {a.repo}@{revision}: {a.runtime}, {len(names)} files", flush=True)
    snapshot_download(repo_id=a.repo, revision=revision, allow_patterns=[MANIFEST, *names], local_dir=root)
    # Verify the copied manifest, not only the cached one used to plan the download.
    if load_manifest(root) != m:
        raise ValueError("Downloaded release manifest changed")
    result = verify(root, m, a.runtime, a.include_export, plain=plain)
    print(json.dumps(dict(status="PASS", repo=a.repo, revision=revision, runtime=a.runtime,
                          bundle=str(root), speculative=not plain, **result), indent=2))
    return 0


def finite_logits(value, shape, where):
    import numpy as np
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError(f"Invalid logits at {where}: expected finite shape {shape}")
    return value


def speculative_greedy(model, drafter, ids, vocab, tokens, stops, gap=0.003):
    """Exercise DFlash2 proposal, T=8 verification and accepted-state commit."""
    import numpy as np
    logits = finite_logits(model.feed(ids, on_features=drafter.add_context), (vocab,), "prompt")
    anchor = int(np.argmax(logits))
    output, cycles, accepted, verify_end = [anchor], 0, 0, 0.0
    while len(output) < tokens and anchor not in stops:
        if verify_end:
            remaining = gap - (time.perf_counter() - verify_end)
            if remaining > 0:
                time.sleep(remaining)
        position = model.pos
        drafts, info = drafter.propose(anchor, position)
        if len(drafts) != 7 or any(type(i) not in (int, np.int32, np.int64) or not 0 <= i < vocab for i in drafts):
            raise ValueError("DFlash2 must propose seven valid token IDs")
        if not isinstance(info, dict):
            raise ValueError("DFlash2 returned invalid output metadata")
        finite_logits(info.get("logits"), (7, vocab), "drafter")
        for key, shape in (("hp", (7, 256)), ("hidden", (8, 5120))):
            value = np.asarray(info.get(key))
            if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
                raise ValueError(f"Invalid DFlash2 {key}: expected finite shape {shape}")
        logits = finite_logits(model.call([anchor, *drafts]), (8, vocab), "verify")
        verify_end = time.perf_counter()
        predicted = np.argmax(logits, axis=1)
        k = 0
        while k < 7 and drafts[k] == int(predicted[k]):
            k += 1
        new = [int(i) for i in drafts[:k]] + [int(predicted[k])]
        new = new[:tokens - len(output)]
        for i, token in enumerate(new):
            if token in stops:
                new = new[:i + 1]
                break
        # Last emitted token remains the next anchor. Rejected/unused rows never enter state.
        committed = len(new)
        features = np.asarray(model.features(committed))
        if features.shape != (committed, 25600) or features.dtype.kind != "f" or not np.isfinite(features).all():
            raise ValueError("Target did not return finite DFlash2 features from all five taps")
        drafter.add_context(features, np.arange(position, position + committed))
        model.accept(committed)
        output.extend(new)
        anchor = new[-1]
        cycles += 1
        accepted += min(k, len(new))
        if anchor in stops:
            break
    if not cycles:
        raise ValueError("DFlash2 was not exercised; use a longer prompt/completion smoke test")
    return output, dict(draft_calls=cycles, verify_calls=cycles, accepted_draft_tokens=accepted)


def smoke(root, m, runtime, ctx, prompt, tokens, plain=False):
    if sys.platform != "darwin":
        raise ValueError("Inference smoke test requires macOS; --check-only is portable")
    if not prompt.strip() or not 1 <= tokens <= 64:
        raise ValueError("Use a nonempty --prompt and --tokens between 1 and 64")
    require_drafter(m, runtime, plain)
    if not plain and tokens < 2:
        raise ValueError("Speculative quick-test needs --tokens >= 2 to exercise the drafter")
    import numpy as np
    from tokenizers import Tokenizer
    model_path = local(root, m["model_path"])
    build = local(root, m["runtimes"][runtime]["path"])
    tok = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    ids = tok.encode(prompt, add_special_tokens=False).ids
    cap = ctx
    if runtime == "coreai":
        runtime_manifest = json.loads((build / "manifest.json").read_text())
        cap = runtime_manifest.get("kv_len", {}).get(str(ctx), ctx)
    if not ids or len(ids) + tokens + 8 > cap:
        raise ValueError("Prompt and completion exceed smoke-test context capacity")
    vocab = model_config(root, m)["vocab_size"]
    if any(i < 0 or i >= vocab for i in ids):
        raise ValueError("Tokenizer produced IDs outside the configured vocabulary")
    os.environ.update(MODEL=str(model_path), EXPORT_DIR=build.name, ANE_OUT=str(build.parent),
                      CTX=str(ctx), EMBED_NPY=str(model_path / "embed_tokens_fp16.npy"))
    os.environ.pop("CTX_LADDER", None)
    for key in ("ONLY", "NLAYERS", "DBG_MIXER_IN", "DBG_GDN", "DBG_TAPS"):
        os.environ.pop(key, None)
    dcfg, ddir, dpkg = None, None, None
    if not plain:
        dcfg, _ = check_drafter(root, m)
        ddir = local(root, m["drafter"]["path"])
        dpkg = local(ddir, m["drafter"]["model"])
        os.environ.update(DRAFTER=str(ddir), COREAI_DRAFTER_COMPUTE="ane")
    start = time.perf_counter()
    if runtime == "coreai":
        os.environ["COREAI_DIR"] = str(build)
        bridge_dir = Path(os.path.expanduser(os.environ.get("COREAI_BRIDGE_DIR", str(
            Path(__file__).resolve().parents[1] / "coreai/swift_bridge")))).resolve()
        bridge = Path(os.path.expanduser(os.environ.get("COREAI_BRIDGE_LIB", str(
            bridge_dir / "libcoreai_bridge.dylib")))).resolve()
        if not bridge.is_file():
            raise ValueError("Build the Swift bridge first: bash coreai/swift_bridge/build.sh")
        os.environ["COREAI_BRIDGE"] = "1"
        os.environ["COREAI_BRIDGE_DIR"] = str(bridge_dir)
        os.environ["COREAI_BRIDGE_LIB"] = str(bridge)
        from qwen38_coreai_model import CoreAIQwen
        model = CoreAIQwen(ctx=ctx, ladder=[ctx])
    else:
        from qwen38_ane_model import load_model
        model = load_model()
    drafter = None
    if not plain:
        from dflash2_ane_drafter import load_codebooks
        from dflash2_coreai_drafter import CoreAIDrafter
        drafter = CoreAIDrafter(dpkg, dcfg, load_codebooks(ddir), model.emb)
    load_seconds = time.perf_counter() - start
    output, start = [], time.perf_counter()
    stops = {i for i in (tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")) if i is not None}
    stats = {}
    if drafter is not None:
        output, stats = speculative_greedy(model, drafter, ids, vocab, tokens, stops)
    else:
        logits = model.feed(ids)
        for i in range(tokens):
            logits = finite_logits(logits, (vocab,), f"step {i}")
            token = int(np.argmax(logits))
            output.append(token)
            if token in stops or i == tokens - 1:
                break
            logits = model.step(token)
    text = tok.decode(output, skip_special_tokens=True)
    if not text.strip():
        raise ValueError("Generation produced no visible text")
    return dict(load_seconds=round(load_seconds, 3), generation_seconds=round(time.perf_counter()-start, 3),
                prompt_tokens=len(ids), generated_tokens=len(output), text=text, speculative=not plain, **stats,
                note="Smoke test checks execution and finite logits; it does not certify ANE placement or model quality.")


def run(a):
    if a.command == "release-manifest":
        return make_manifest(a)
    if a.command == "download":
        return download(a)
    root = a.bundle.expanduser().resolve()
    if a.report and a.report.expanduser().resolve().is_relative_to(root):
        raise ValueError("Write --report outside the weight bundle to preserve its inventory")
    m = load_manifest(root)
    if a.runtime not in m["runtimes"]:
        raise ValueError(f"Release does not contain runtime: {a.runtime}")
    contexts = m["runtimes"][a.runtime]["contexts"]
    ctx = a.ctx if a.ctx is not None else min(contexts)
    if ctx not in contexts:
        raise ValueError(f"Unsupported --ctx {ctx}; available: {contexts}")
    plain = getattr(a, "plain", False)
    result = dict(runtime=a.runtime, ctx=ctx, speculative=not plain, **verify(root, m, a.runtime, plain=plain))
    if not a.check_only:
        result.update(smoke(root, m, a.runtime, ctx, a.prompt, a.tokens, plain=plain))
    result.update(status="PASS", inference_run=not a.check_only)
    if a.report:
        dest = a.report.expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_commands(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args()
    try:
        raise SystemExit(run(args))
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
