#!/usr/bin/env python3
"""Experimental direct-attention extension of a validated 8K/16K build.

Requires coreai-core==1.0.0b2, explicit --m5pro-24gb opt-in, and exactly
Apple M5 Pro with 24 GB Unified Memory. Retains the original graphs and
clones the 16K graphs, changing only context types and two slice-boundary
constants. The CLI supports 24K and lean 31K. This does not prove ANE
placement or model quality: probe and validate separately.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import sys

from select_context import artifact_path, verify_resources
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qwen38_hardware_profile import M5PRO_24GB_PROFILE, require_m5pro_24gb


def extension_contexts(contexts):
    contexts = sorted(set(contexts))
    if not contexts or any(not isinstance(c, int) or c <= 16384 or c % 1024
                           or c + 64 > 65536 for c in contexts):
        raise ValueError("Extensions must be whole Ki-token contexts above 16K and below the 64K axis cap")
    return contexts


def rewrite(text, tokens, context=24576):
    if tokens not in (8, 64):
        raise ValueError("Only the published v8 and p64 graphs are supported")
    extension_contexts([context])
    for old, new in ((16384 + tokens, context + tokens), (16384, context)):
        text = re.sub(r"(?<![\d.])" + str(old) + r"(?![\d.])", str(new), text)
    return text.replace("_16k", f"_{context // 1024}k")


def prefill_extensions(contexts, prefill_contexts):
    selected = contexts if prefill_contexts is None else sorted(set(prefill_contexts))
    if any(c not in contexts for c in selected):
        raise ValueError("Prefill extensions must be a subset of decode extensions")
    return selected


def extend_manifest(source, contexts=(24576,), prefill_contexts=None):
    contexts = extension_contexts(contexts)
    prefill_contexts = prefill_extensions(contexts, prefill_contexts)
    if source.get("kv_cache", {}).get("format", "fp16") != "fp16":
        raise ValueError("Direct extension requires the legacy FP16-cache build, not V8/selectable graphs")
    if source["ctxs"] != [8192, 16384] or source["pctxs"] != [8192, 16384]:
        raise ValueError("Source must be a validated 8K/16K ladder")
    if source["T"] != 8 or source["TP"] != 64:
        raise ValueError("Unexpected verification/prefill block size")
    for key in ("kv_len", "pkv_len"):
        if source[key].get("16384") != 16384:
            raise ValueError("Source must have full-length 16K KV tensors")
    result = deepcopy(source)
    result["hardware_profile"] = deepcopy(M5PRO_24GB_PROFILE)
    result["ctxs"].extend(contexts)
    result["pctxs"].extend(prefill_contexts)
    result["kv_len"].update({str(c): c for c in contexts})
    result["pkv_len"].update({str(c): c for c in prefill_contexts})
    for chunk in result["chunks"]:
        if set(chunk["entries"]) != {"v8_8k", "v8_16k", "p64_8k", "p64_16k"}:
            raise ValueError("Unexpected source graphs")
        sizes = [8192, 16384, *contexts]
        psizes = [8192, 16384, *prefill_contexts]
        chunk["entries"] = ([f"v8_{c // 1024}k" for c in sizes] +
                            [f"p64_{c // 1024}k" for c in psizes])
        chunk["entries_ctx"] = [sizes.copy(), psizes.copy()]
        chunk.pop("compiled", None)
    result["head"].pop("compiled", None)
    return result


def extend_asset(source, destination, contexts=(24576,), prefill_contexts=None):
    require_m5pro_24gb()
    from coreai.authoring import AIModelAsset
    from coreai._compiler.ir import Attribute, Type

    contexts = extension_contexts(contexts)
    prefill_contexts = prefill_extensions(contexts, prefill_contexts)
    if destination.exists():
        raise ValueError(f"Refusing to overwrite {destination}")
    asset = AIModelAsset.load(source)
    program = asset.program
    module = program._mlir_module
    expected = {op.sym_name.value: op.operation.get_asm(large_elements_limit=16)
                for op in module.body.operations}
    if set(expected) != {"v8_8k", "v8_16k", "p64_8k", "p64_16k"}:
        raise ValueError("Unexpected source graphs")
    changes = []

    def walk(op, tokens, context):
        op = op.operation
        for named in list(op.attributes):
            old = str(named.attr)
            new = rewrite(old, tokens, context)
            if new != old:
                if "dense_resource" in old:
                    raise ValueError("A resource attribute would change")
                if op.name == "coreai.constant" and old not in (
                    "dense<[0, 0, 16384]> : tensor<3xsi32>",
                    "dense<[2147483647, 2147483647, 16384]> : tensor<3xsi32>",
                ):
                    raise ValueError(f"Unexpected constant rewrite: {old}")
                op.attributes[named.name] = Attribute.parse(new)
                changes.append(dict(op=op.name, attribute=named.name, before=old, after=new))
        for value in op.results:
            old = str(value.type)
            new = rewrite(old, tokens, context)
            if new != old:
                if op.name == "coreai.constant":
                    raise ValueError("A weight/constant shape would change")
                value.set_type(Type.parse(new))
                changes.append(dict(op=op.name, result_type=old, after=new))
        for region in op.regions:
            for block in region.blocks:
                for arg in block.arguments:
                    old = str(arg.type)
                    new = rewrite(old, tokens, context)
                    if new != old:
                        arg.set_type(Type.parse(new))
                        changes.append(dict(block_argument=old, after=new))
                for child in block.operations:
                    walk(child, tokens, context)

    with module.context:
        for context in contexts:
            for prefix, tokens in (("v8", 8), ("p64", 64)):
                if prefix == "p64" and context not in prefill_contexts:
                    continue
                name = f"{prefix}_16k"
                cloned = program.get_graph(name).operation.clone()
                walk(cloned, tokens, context)
                module.body.append(cloned)
                expected[f"{prefix}_{context // 1024}k"] = rewrite(expected[name], tokens, context)
        module.operation.verify()
        for name, text in expected.items():
            if program.get_graph(name).operation.get_asm(large_elements_limit=16) != text:
                raise ValueError(f"Unexpected graph change: {name}")
    program.save_asset(destination, metadata=asset.metadata)
    reloaded = AIModelAsset.load(destination).program
    reloaded._mlir_module.operation.verify()
    for name, text in expected.items():
        if reloaded.get_graph(name).operation.get_asm(large_elements_limit=16) != text:
            raise ValueError(f"Graph changed during serialization: {name}")
    resources = verify_resources(source, destination, reloaded._mlir_module)
    return dict(file=destination.name, changes=changes, **resources,
                retained_graphs_unchanged=True,
                source_main_hash=(source / "main.hash").read_bytes().hex(),
                derived_main_hash=(destination / "main.hash").read_bytes().hex())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ctx", default="24576", help="comma-separated experimental extensions: 24576 or 24576,31744")
    parser.add_argument("--prefill-ctx", help="subset of extensions with batched prefill; defaults to all")
    parser.add_argument("--m5pro-24gb", action="store_true", help="opt into the experimental M5 Pro / 24 GB Unified Memory workaround")
    args = parser.parse_args(argv)
    if not args.m5pro_24gb:
        parser.error("Experimental extension requires explicit --m5pro-24gb opt-in")
    try:
        require_m5pro_24gb()
    except ValueError as exc:
        parser.error(str(exc))
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or source.is_relative_to(output) or output.is_relative_to(source):
        parser.error("Choose a new output directory outside the source")
    contexts = extension_contexts([int(c) for c in args.ctx.split(',')])
    pcontexts = None if args.prefill_ctx is None else [int(c) for c in args.prefill_ctx.split(',') if c]
    pcontexts = prefill_extensions(contexts, pcontexts)
    if not set(contexts) <= {24576, 31744} or 31744 in pcontexts:
        parser.error("This 24 GB profile supports 24K/31K decode and batched prefill only through 24K; "
                     "use --ctx 24576,31744 --prefill-ctx 24576 for the lean 31K ladder")
    manifest = extend_manifest(json.loads((source / "manifest.json").read_text()), contexts, pcontexts)
    for record in [*manifest["chunks"], manifest["head"]]:
        if not artifact_path(source, record["file"]).is_dir():
            parser.error(f"Missing asset: {record['file']}")
    output.mkdir(parents=True)
    report = dict(source="local source path omitted", contexts=manifest["ctxs"],
                  attention="direct, cloned from 16K", extensions=contexts,
                  prefill_extensions=pcontexts, chunks=[])
    for record in manifest["chunks"]:
        report["chunks"].append(extend_asset(artifact_path(source, record["file"]),
                                             artifact_path(output, record["file"]), contexts, pcontexts))
        print(f"extended {record['file']}; original graphs and exact resources verified", flush=True)
    shutil.copytree(artifact_path(source, manifest["head"]["file"]),
                    artifact_path(output, manifest["head"]["file"]))
    (output / "context_extension.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
