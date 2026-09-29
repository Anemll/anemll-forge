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
MODEL_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "generation_config.json", "chat_template.jinja", "chat_template.json", "vocab.json",
               "merges.txt", "tokenizer.model", "added_tokens.json", "embed_tokens_fp16.npy"}


def add_commands(sub):
    p = sub.add_parser("release-manifest", help="hash a prepared HF upload directory")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--model-id", required=True, help="canonical upstream checkpoint ID")
    p.add_argument("--model-revision", required=True, help="upstream checkpoint revision used")
    p = sub.add_parser("download", help="download and verify one runtime from Hugging Face")
    p.add_argument("--repo", required=True, help="HF owner/repository")
    p.add_argument("--revision", default="main", help="tag/commit/branch, resolved to a commit before download")
    p.add_argument("--runtime", choices=("coreai", "coreml"), default="coreai")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include-export", action="store_true", help="also download optional quantized conversion weights")
    p = sub.add_parser("quick-test", help="verify a bundle and test short greedy inference")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--runtime", choices=("coreai", "coreml"), default="coreai")
    p.add_argument("--ctx", type=int, help="default: smallest supported context")
    p.add_argument("--check-only", action="store_true", help="integrity/layout check without loading the model")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--report", type=Path, help="optional JSON test report")


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
        if group not in roots or not name.startswith(roots[group] + "/"):
            raise ValueError(f"File is outside its component: {name}")
        if name in seen:
            raise ValueError(f"Duplicate file entry: {name}")
        seen.add(name)
        if type(entry.get("bytes")) is not int or entry["bytes"] < 0:
            raise ValueError(f"Invalid byte count: {name}")
        if not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError(f"Invalid SHA256: {name}")
    for required in ("config.json", "tokenizer.json", "tokenizer_config.json", "embed_tokens_fp16.npy"):
        if roots["model"] + "/" + required not in seen:
            raise ValueError(f"Missing model file in inventory: {required}")
    for runtime in runtimes:
        if not any(f["component"] == runtime for f in files):
            raise ValueError(f"No runtime files: {runtime}")
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


def verify(root, m, runtime, include_export=False):
    if runtime not in m["runtimes"]:
        raise ValueError(f"Release does not contain runtime: {runtime}")
    groups = {"model", runtime} | ({"export"} if include_export else set())
    files = [f for f in m["files"] if f["component"] in groups]
    for f in files:
        p = local(root, f["path"])
        if not p.is_file() or p.stat().st_size != f["bytes"]:
            raise ValueError(f"Missing file or size mismatch: {f['path']}")
        if digest(p) != f["sha256"]:
            raise ValueError(f"SHA256 mismatch: {f['path']}")
    check_layout(root, m, runtime)
    return {"verified_files": len(files), "verified_bytes": sum(f["bytes"] for f in files)}


def make_manifest(a):
    root = a.bundle.expanduser().resolve()
    m = dict(schema_version=1, upstream_model=dict(id=a.model_id, revision=a.model_revision),
             model_path="model", runtimes={}, files=[])
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
            m["files"].append(dict(path=name, component=group, bytes=p.stat().st_size, sha256=digest(p)))
    validate_manifest(m)
    for runtime in m["runtimes"]:
        check_layout(root, m, runtime)
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
    if a.include_export and not m.get("export_path"):
        raise ValueError("This release has no optional export component")
    root = a.output.expanduser().resolve()
    old = root / MANIFEST
    if old.exists() and json.loads(old.read_text()) != m:
        raise ValueError("Output contains a different release; use a new --output directory")
    groups = {"model", a.runtime} | ({"export"} if a.include_export else set())
    names = [f["path"] for f in m["files"] if f["component"] in groups]
    for name in [MANIFEST, *names]:
        local(root, name)  # Reject existing symlink escapes before any download writes.
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {a.repo}@{revision}: {a.runtime}, {len(names)} files", flush=True)
    snapshot_download(repo_id=a.repo, revision=revision, allow_patterns=[MANIFEST, *names], local_dir=root)
    # Verify the copied manifest, not only the cached one used to plan the download.
    if load_manifest(root) != m:
        raise ValueError("Downloaded release manifest changed")
    result = verify(root, m, a.runtime, a.include_export)
    print(json.dumps(dict(status="PASS", repo=a.repo, revision=revision, runtime=a.runtime,
                          bundle=str(root), **result), indent=2))
    return 0


def smoke(root, m, runtime, ctx, prompt, tokens):
    if sys.platform != "darwin":
        raise ValueError("Inference smoke test requires macOS; --check-only is portable")
    if not prompt.strip() or not 1 <= tokens <= 64:
        raise ValueError("Use a nonempty --prompt and --tokens between 1 and 64")
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
    start = time.perf_counter()
    if runtime == "coreai":
        os.environ["COREAI_DIR"] = str(build)
        bridge = Path(__file__).resolve().parents[1] / "coreai/swift_bridge/libcoreai_bridge.dylib"
        if not bridge.is_file():
            raise ValueError("Build the Swift bridge first: bash coreai/swift_bridge/build.sh")
        os.environ["COREAI_BRIDGE"] = "1"
        os.environ["COREAI_BRIDGE_DIR"] = str(bridge.parent)
        os.environ["COREAI_BRIDGE_LIB"] = str(bridge)
        from qwen38_coreai_model import CoreAIQwen
        model = CoreAIQwen(ctx=ctx, ladder=[ctx])
    else:
        from qwen38_ane_model import load_model
        model = load_model()
    load_seconds = time.perf_counter() - start
    output, start = [], time.perf_counter()
    logits = model.feed(ids)
    stops = {i for i in (tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")) if i is not None}
    for i in range(tokens):
        logits = np.asarray(logits)
        if logits.shape != (vocab,) or not np.isfinite(logits).all():
            raise ValueError(f"Invalid logits at step {i}: expected finite shape ({vocab},)")
        token = int(np.argmax(logits))
        output.append(token)
        if token in stops or i == tokens - 1:
            break
        logits = model.step(token)
    text = tok.decode(output, skip_special_tokens=True)
    if not text.strip():
        raise ValueError("Generation produced no visible text")
    return dict(load_seconds=round(load_seconds, 3), generation_seconds=round(time.perf_counter()-start, 3),
                prompt_tokens=len(ids), generated_tokens=len(output), text=text,
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
    result = dict(runtime=a.runtime, ctx=ctx, **verify(root, m, a.runtime))
    if not a.check_only:
        result.update(smoke(root, m, a.runtime, ctx, a.prompt, a.tokens))
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
