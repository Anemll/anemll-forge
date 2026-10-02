"""KV representation metadata and accepted-row writes (no model imports)."""


def cache_formats(manifest):
    """Validate cache layouts without importing model or array dependencies."""
    metadata = manifest.get("kv_cache", {})
    if not isinstance(metadata, dict):
        raise ValueError("Invalid kv_cache metadata")
    actual = metadata.get("format", "fp16")
    if actual == "selectable":
        layouts = metadata.get("formats")
        if not isinstance(layouts, dict) or set(layouts) != {"fp16", "v8"}:
            raise ValueError("Selectable KV exports must contain fp16 and v8 layouts")
        for name, layout in layouts.items():
            if not isinstance(layout, dict) or layout.get("format") != name:
                raise ValueError("Invalid selectable KV layout")
            cache_formats({"kv_cache": layout})
            if name == "fp16" and (layout.get("keys") != "float16" or layout.get("values") != "float16"):
                raise ValueError("FP16 layout requires FP16 keys and values")
        if metadata.get("default") not in layouts:
            raise ValueError("Invalid default KV format")
        for chunk in manifest.get("chunks", []):
            for name in layouts:
                cache_entries(manifest, chunk, name)
        return ("fp16", "v8")
    if actual not in ("fp16", "v8"):
        raise ValueError("KV cache format must be auto, fp16 or v8")
    if actual == "v8" and (metadata.get("keys") != "float16" or metadata.get("values") != "int8"
                           or metadata.get("scales") != "float16" or metadata.get("scale_granularity") != "token_head"):
        raise ValueError("V8 requires FP16 keys, INT8 values and FP16 scales per token/head")
    return (actual,)


def cache_entries(manifest, chunk, mode):
    """Map runtime's canonical names onto a selected format's physical functions."""
    if manifest.get("kv_cache", {}).get("format") != "selectable":
        return {name: name for name in chunk["entries"]}
    formats = chunk.get("entries_by_kv", {})
    aliases = formats.get(mode) if isinstance(formats, dict) else None
    if not isinstance(aliases, dict) or not aliases:
        raise ValueError(f"Missing {mode} KV entry map for {chunk.get('file', 'chunk')}")
    required = {f"v8_{c // 1024}k" for c in manifest.get("ctxs", [])}
    if manifest.get("TP", 0):
        required.update(f"p64_{c // 1024}k" for c in manifest.get("pctxs", []))
    if not required.issubset(aliases):
        raise ValueError(f"Incomplete {mode} KV entry map for {chunk.get('file', 'chunk')}")
    if any(not isinstance(k, str) or not isinstance(v, str) or v not in chunk["entries"]
           for k, v in aliases.items()):
        raise ValueError(f"Invalid {mode} KV physical entry map")
    return dict(aliases)


def cache_format(manifest, requested="auto"):
    """Legacy manifests are FP16; selectable exports share weights for both modes."""
    formats = cache_formats(manifest)
    if requested not in ("auto", "fp16", "v8"):
        raise ValueError("KV cache format must be auto, fp16 or v8")
    actual = manifest.get("kv_cache", {}).get("default", formats[0]) if len(formats) > 1 else formats[0]
    if requested != "auto" and requested not in formats:
        raise ValueError(f"Requested {requested} KV cache, but this bundle contains {actual}. "
                         "Select a matching --build; cache compression requires different ANE entry points.")
    return actual if requested == "auto" else requested


def quantize_values(values):
    """Symmetric signed INT8; round using the scale that is actually stored."""
    import numpy as np
    values = np.asarray(values, dtype=np.float16).astype(np.float32)
    scales = np.maximum(np.max(np.abs(values), axis=-1) / 127, 1e-6).astype(np.float16)
    codes = np.clip(np.rint(values / scales.astype(np.float32)[..., None]), -127, 127).astype(np.int8)
    return codes, scales


def append_rows(kv, outputs, attention_indices, position, count, mode):
    """Write accepted rows only. Keys/local outputs always remain FP16."""
    if not count:
        return
    end = position + count
    for j in attention_indices:
        keys = outputs[f"k{j}_new"][:, :count]
        values = outputs[f"v{j}_new"][:, :count]
        kv[f"k{j}"][1][:, position:end] = keys
        if mode == "v8":
            codes, scales = quantize_values(values)
            kv[f"v{j}"][1][:, position:end] = codes
            kv[f"vs{j}"][1][:, position:end] = scales
        else:
            kv[f"v{j}"][1][:, position:end] = values
