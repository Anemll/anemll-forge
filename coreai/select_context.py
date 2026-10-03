#!/usr/bin/env python3
"""Derive a context-ladder target build without re-exporting or changing weights.

Requires the Apple coreai-core authoring SDK (use the source asset's version).
This does not compile or load a model on ANE. Use inspect_coreai_cache.py after
isolated hardware probes; a successful load alone does not prove ANE placement.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qwen38_hardware_profile import validate_hardware_profile


def context_manifest(source, ctx, prefill_ctx=None):
    if source.get("kv_cache", {}).get("format") == "selectable":
        raise ValueError("Selectable-cache bundles require format-aware graph selection; use a legacy FP16 source")
    contexts = sorted(set([ctx] if isinstance(ctx, int) else ctx))
    if not contexts:
        raise ValueError("Select at least one context")
    for context in contexts:
        if context not in source.get("ctxs", []):
            raise ValueError(f"Unsupported verification context {context}")
        if context <= 0 or context % 1024:
            raise ValueError("Context must be a positive multiple of 1024")
    prefill = contexts.copy() if prefill_ctx is None else sorted(set(prefill_ctx))
    for context in prefill:
        if context not in contexts or context not in source.get("pctxs", []):
            raise ValueError(f"Context {context} must support both verification and prefill")
    result = deepcopy(source)
    entries = [f"v8_{context // 1024}k" for context in contexts]
    entries += [f"p64_{context // 1024}k" for context in prefill]
    result["ctxs"] = contexts.copy()
    result["pctxs"] = prefill.copy()
    result["kv_len"] = {str(context): source["kv_len"][str(context)] for context in contexts}
    result["pkv_len"] = {str(context): source["pkv_len"][str(context)] for context in prefill}
    if not prefill:
        result["TP"] = 0
    for chunk in result["chunks"]:
        if not set(entries) <= set(chunk["entries"]):
            raise ValueError(f"Missing required entries in {chunk['file']}")
        chunk["entries"] = entries.copy()
        chunk["entries_ctx"] = [contexts.copy(), prefill.copy()]
        chunk.pop("entries_by_kv", None)
        chunk.pop("compiled", None)
    result["head"].pop("compiled", None)
    return result


def artifact_path(root, filename):
    path = root / filename
    if Path(filename).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes build directory: {filename}")
    return path


def resource_digest(path):
    """Hash the MLIR bytecode resource section, including its exact weight bytes."""
    data = path.read_bytes()
    if data[:4] != b"ML\xefR":
        raise ValueError("Not MLIR bytecode")
    pos = 4

    def vint():
        nonlocal pos
        first = data[pos]
        pos += 1
        if first == 0:
            value = int.from_bytes(data[pos:pos + 8], "little")
            pos += 8
            return value
        extra = (first & -first).bit_length() - 1
        value = first >> (extra + 1)
        for i in range(extra):
            value |= data[pos] << (7 - extra + 8 * i)
            pos += 1
        return value

    version = vint()
    if version != 6:
        raise ValueError(f"Resource verification supports bytecode v6, not v{version}")
    pos = data.index(b"\x00", pos) + 1
    while pos < len(data):
        tag = data[pos]
        pos += 1
        size = vint()
        if tag & 128:
            alignment = vint()
            pos = (pos + alignment - 1) // alignment * alignment
        if tag & 127 == 5:
            if pos + size > len(data):
                raise ValueError("Truncated resource section")
            return hashlib.sha256(data[pos:pos + size]).hexdigest()
        pos += size
    raise ValueError("No resource section")


def named_resource_digests(module):
    """Verify named buffers independent of bytecode resource ordering/padding.

    The textual resource encoding includes the buffer's alignment and every
    byte. Do not print this assembly: it contains the full model weights.
    """
    asm = module.operation.get_asm(large_elements_limit=1_000_000_000,
                                   large_resource_limit=1_000_000_000)
    marker = asm.rfind('{-#')
    if marker < 0:
        raise ValueError("Missing MLIR resource table")
    referenced = set(re.findall(r'dense_resource<([^>]+)>', asm[:marker]))
    resources = {}
    for match in re.finditer(r'([\w.]+): "(0x[0-9A-Fa-f]+)"', asm[marker:]):
        name, encoded = match.groups()
        if name in resources:
            raise ValueError(f"Duplicate resource: {name}")
        resources[name] = (len(encoded), hashlib.sha256(encoded.upper().encode('ascii')).hexdigest())
    if not resources or not referenced <= resources.keys():
        raise ValueError("Incomplete named resource verification")
    return resources


def verify_resources(source, destination, derived_module):
    original = resource_digest(source / 'main.mlirb')
    derived = resource_digest(destination / 'main.mlirb')
    if original == derived:
        return dict(resources_sha256=original, resources_unchanged=True,
                    resource_section_unchanged=True)
    from coreai.authoring import AIModelAsset
    before = named_resource_digests(AIModelAsset.load(source).program._mlir_module)
    after = named_resource_digests(derived_module)
    if before != after:
        raise ValueError("Named resource bytes changed; refusing to publish manifest")
    return dict(resources_sha256=original, derived_resources_sha256=derived,
                resources_unchanged=True, resource_section_unchanged=False,
                named_resources_verified=len(before))


def select_asset(source, destination, entries):
    from coreai.authoring import AIModelAsset

    if destination.exists():
        raise ValueError(f"Refusing to overwrite {destination}")
    asset = AIModelAsset.load(source)
    program = asset.program
    original = {}
    for op in list(program._mlir_module.body.operations):
        if op.name != "coreai.graph":
            raise ValueError(f"Unexpected top-level operation: {op.name}")
        name = op.sym_name.value
        if name in entries:
            original[name] = op.operation.get_asm(large_elements_limit=16)
        else:
            op.operation.erase()
    if set(original) != set(entries):
        raise ValueError(f"Requested graph missing in {source}")
    program._mlir_module.operation.verify()
    # Deliberately do not optimize, palettize, or otherwise rewrite the graphs.
    program.save_asset(destination, metadata=asset.metadata)
    reloaded = AIModelAsset.load(destination).program
    reloaded._mlir_module.operation.verify()
    for name, text in original.items():
        if reloaded.get_graph(name).operation.get_asm(large_elements_limit=16) != text:
            raise ValueError(f"Graph changed during serialization: {name}")
    resources = verify_resources(source, destination, reloaded._mlir_module)
    return {"file": destination.name, "entries": entries,
            "graphs_unchanged": True, **resources,
            "source_main_hash": (source / "main.hash").read_bytes().hex(),
            "derived_main_hash": (destination / "main.hash").read_bytes().hex()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ctx", default="8192", help="comma-separated contexts, e.g. 8192,16384")
    parser.add_argument("--prefill-ctx", default=None,
                        help="optional subset of --ctx retaining p64; other contexts feed through v8")
    args = parser.parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"Output already exists; choose a new directory: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("Source and output must not contain each other")
    try:
        contexts = [int(value) for value in args.ctx.split(",")]
        prefill = None if args.prefill_ctx is None else [int(v) for v in args.prefill_ctx.split(',') if v]
        original_manifest = json.loads((source / "manifest.json").read_text())
        validate_hardware_profile(original_manifest)
        manifest = context_manifest(original_manifest, contexts, prefill)
        for record in [*manifest["chunks"], manifest["head"]]:
            if not artifact_path(source, record["file"]).is_dir():
                raise ValueError(f"Missing asset: {record['file']}")
        output.mkdir(parents=True)
        report = {"source": "local source path omitted", "contexts": manifest["ctxs"],
                  "prefill_contexts": manifest["pctxs"], "chunks": []}
        if len(manifest["ctxs"]) == 1:
            report["context"] = manifest["ctxs"][0]  # compatibility with single-context records
        for record in manifest["chunks"]:
            result = select_asset(artifact_path(source, record["file"]),
                                  artifact_path(output, record["file"]), record["entries"])
            report["chunks"].append(result)
            print(f"selected {record['file']}: {', '.join(record['entries'])}; exact resources verified", flush=True)
        shutil.copytree(artifact_path(source, manifest["head"]["file"]),
                        artifact_path(output, manifest["head"]["file"]))
        (output / "context_selection.json").write_text(json.dumps(report, indent=2) + "\n")
        # Publish only after all unchanged graphs and resources were verified.
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Context selection failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
