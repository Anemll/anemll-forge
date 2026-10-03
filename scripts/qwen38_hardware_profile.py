"""Fail-closed gates for opt-in, hardware-specific research builds.

Unmarked upstream manifests retain their normal behavior on every chip.
No hardware detection here selects a workaround or changes context defaults.
"""
import platform
import subprocess


M5PRO_24GB_PROFILE = {
    "id": "m5pro-24gb-direct-fp16-v1",
    "chip": "Apple M5 Pro",
    "unified_memory_bytes": 24 * 1024**3,
    "experimental": True,
}


def require_m5pro_24gb():
    """Require the exact tested chip and physical RAM, not current free RAM."""
    if platform.system() != "Darwin":
        raise ValueError("Experimental direct-attention profile requires macOS on Apple M5 Pro with 24 GB Unified Memory")
    try:
        chip = subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                       text=True, timeout=5).strip()
        memory = int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"],
                                            text=True, timeout=5).strip())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ValueError("Cannot verify chip and Unified Memory; refusing the experimental workaround") from exc
    if chip != M5PRO_24GB_PROFILE["chip"] or memory != M5PRO_24GB_PROFILE["unified_memory_bytes"]:
        raise ValueError("Experimental direct-attention profile requires exactly Apple M5 Pro and 24 GB Unified Memory; "
                         "it is not enabled on M6, other M5 variants, or other RAM capacities")


def validate_hardware_profile(manifest):
    """Reject incompatible/unknown marked builds before allocating a model."""
    if "hardware_profile" not in manifest:
        return
    if manifest["hardware_profile"] != M5PRO_24GB_PROFILE:
        raise ValueError("Unknown or malformed hardware_profile; refusing hardware-specific build")
    if manifest.get("kv_cache", {}).get("format", "fp16") != "fp16":
        raise ValueError("M5 Pro / 24 GB direct-attention profile requires FP16 KV cache")
    if any(c not in (8192, 16384, 24576, 31744) for c in manifest.get("ctxs", [])) or \
            any(c not in (8192, 16384, 24576) for c in manifest.get("pctxs", [])):
        raise ValueError("M5 Pro / 24 GB profile supports decode through 31K and batched prefill only through 24K")
    require_m5pro_24gb()
