"""End-to-end leverage model for M6 ANE attention-compute changes.

Companion to scripts/m6_attn_compute.py. That harness explores the attention
kernel dtype (INT8-INT8 / FP8-FP8 versus the current V8 dequant-to-FP16). This
module answers a different question: if such a kernel really ran faster on the
M6 ANE, how much would end-to-end prefill/decode move?

Everything here is arithmetic on documented bytes and an Amdahl split. It is not
an ANE measurement. The attention time fraction is an on-device input, not a
constant. See docs/research/M6_COMPUTE_FEASIBILITY_2026-10-02.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

# Documented byte counts from docs/PERFORMANCE_SESSION.md and the V8 trace.
# Labeled measured-prior in the research doc; reproduced here as constants only.
WEIGHT_BYTES_PER_VERIFY = 10.61e9       # whole verifier forward (chunks + head)
CHUNK_BYTES = 9.98e9                     # chunk weights alone
HEAD_BYTES = 0.63e9                      # output head alone
KV_BYTES_PER_TOKEN_FP16 = 65536.0       # 16 layers x 4 KV heads x 256 x 2 (K and V) x 2 bytes
KV_BYTES_PER_TOKEN_V8 = 49280.0         # FP16 K + INT8 V + FP16 token/head scales (48.125 KiB)

# Qualitative kernel-speedup ceilings for the ranked options. These are the best
# case a kernel could reach IF the M6 executes the path natively. They are
# inferred ceilings for modeling, never measured results. speedup 1.0 == no gain.
OPTION_CEILINGS = {
    "int8_int8_attn": {"kernel_speedup": 2.0, "affects": "attention_compute",
                       "basis": "W8A8 2x convention; may collapse to 1.0 if ANE dequants to FP16"},
    "fp8_fp8_attn": {"kernel_speedup": 2.0, "affects": "attention_compute",
                     "basis": "E4M3 both operands; Forge FP8 LUT values were promoted to FP16 at compile"},
    "k8_with_v8_storage": {"kernel_speedup": 1.3, "affects": "kv_bandwidth",
                           "basis": "another ~25% logical KV payload on top of V8; bandwidth not compute"},
    "fused_or_blocked_attention": {"kernel_speedup": 1.3, "affects": "attention_compute",
                                   "basis": "less traffic / better fusion, not a 2x MAC"},
    "zero_skip_masked_kv": {"kernel_speedup": 1.2, "affects": "attention_compute",
                            "basis": "only if the ANE skips zero or masked tiles"},
    "lowprec_gdn_compute": {"kernel_speedup": 1.5, "affects": "mixer_compute",
                            "basis": "48 of 64 layers; high recurrent-quality risk"},
    "bonded_compile_flags": {"kernel_speedup": 1.0, "affects": "none",
                             "basis": "mode 2 already default, same ms as mode 0"},
    "winograd": {"kernel_speedup": 1.0, "affects": "none",
                 "basis": "1x1 convs and matmuls; no Winograd path or Forge hook"},
}


def amdahl_speedup(fraction, kernel_speedup):
    """End-to-end speedup when a kernel taking `fraction` of time is sped up.

    speedup = 1 / ((1 - fraction) + fraction / kernel_speedup).
    kernel_speedup may be math.inf (kernel reduced to zero time).
    """
    _check_fraction(fraction)
    if kernel_speedup <= 0:
        raise ValueError("kernel_speedup must be positive")
    if math.isinf(kernel_speedup):
        serial = 1.0 - fraction
    else:
        serial = (1.0 - fraction) + fraction / kernel_speedup
    if serial <= 0:
        return math.inf
    return 1.0 / serial


def required_fraction(target_speedup, kernel_speedup):
    """Minimum time fraction the kernel must occupy to reach `target_speedup`.

    From target = 1/((1-f)+f/s): f = (1 - 1/target) / (1 - 1/s).
    Returns a value in (0, 1], or None if even an infinitely fast kernel cannot
    reach the target (target > 1/(0) is impossible; target capped by s).
    """
    if target_speedup < 1.0:
        raise ValueError("target_speedup must be >= 1.0")
    if kernel_speedup <= 1.0 and not math.isinf(kernel_speedup):
        return None if target_speedup > 1.0 else 0.0
    numerator = 1.0 - 1.0 / target_speedup
    denom = 1.0 if math.isinf(kernel_speedup) else (1.0 - 1.0 / kernel_speedup)
    frac = numerator / denom
    if frac > 1.0 + 1e-12:
        return None
    return min(max(frac, 0.0), 1.0)


def required_kernel_speedup(target_speedup, fraction):
    """Minimum kernel speedup to reach `target_speedup` given a time fraction.

    From target = 1/((1-f)+f/s): s = f / (1/target - (1 - f)).
    Returns math.inf when not reachable at any finite speedup, i.e. when
    fraction <= 1 - 1/target.
    """
    _check_fraction(fraction)
    if target_speedup < 1.0:
        raise ValueError("target_speedup must be >= 1.0")
    if target_speedup == 1.0:
        return 1.0
    headroom = 1.0 / target_speedup - (1.0 - fraction)
    if headroom <= 0:
        return math.inf
    return fraction / headroom


def forward_bytes(ctx, kv_bytes_per_token=KV_BYTES_PER_TOKEN_FP16, weight_bytes=WEIGHT_BYTES_PER_VERIFY):
    """Modeled bytes moved in one forward: fixed weights plus KV history traffic."""
    if ctx < 0:
        raise ValueError("ctx must be >= 0")
    return weight_bytes + kv_bytes_per_token * ctx


def kv_traffic_fraction(ctx, kv_bytes_per_token=KV_BYTES_PER_TOKEN_FP16, weight_bytes=WEIGHT_BYTES_PER_VERIFY):
    """KV share of modeled forward bytes. This is a BANDWIDTH share, not a compute share."""
    total = forward_bytes(ctx, kv_bytes_per_token, weight_bytes)
    return (kv_bytes_per_token * ctx) / total


def bandwidth_bound_time_ms(bytes_moved, bandwidth_gb_s):
    """Lower-bound forward time if purely bandwidth bound. A floor, not a prediction."""
    if bandwidth_gb_s <= 0:
        raise ValueError("bandwidth_gb_s must be positive")
    return 1e3 * bytes_moved / (bandwidth_gb_s * 1e9)


def option_end_to_end(option, attention_fraction, mixer_fraction=0.0, kv_fraction=0.0):
    """End-to-end speedup for one ranked option given measured-on-device fractions.

    The caller supplies the fraction of forward time actually spent in each phase
    (measure on an M6; do not guess). Each option only moves the phase it affects.
    """
    if option not in OPTION_CEILINGS:
        raise ValueError(f"unknown option {option!r}; choose from {', '.join(OPTION_CEILINGS)}")
    spec = OPTION_CEILINGS[option]
    affects = spec["affects"]
    fraction_by_phase = {
        "attention_compute": attention_fraction,
        "mixer_compute": mixer_fraction,
        "kv_bandwidth": kv_fraction,
        "none": 0.0,
    }
    frac = fraction_by_phase[affects]
    speedup = amdahl_speedup(frac, spec["kernel_speedup"])
    return {
        "option": option,
        "affects": affects,
        "assumed_kernel_speedup_ceiling": spec["kernel_speedup"],
        "phase_fraction_used": frac,
        "modeled_end_to_end_speedup": speedup,
        "basis": spec["basis"],
        "label": "inferred_ceiling_times_on_device_fraction_not_measured",
    }


def _check_fraction(fraction):
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")


def cmd_amdahl(args):
    speedup = amdahl_speedup(args.fraction, _parse_speedup(args.kernel_speedup))
    print(json.dumps({
        "fraction": args.fraction,
        "kernel_speedup": args.kernel_speedup,
        "modeled_end_to_end_speedup": speedup,
        "label": "arithmetic_not_measured",
    }, indent=2))
    return 0


def cmd_required(args):
    target = args.target
    rows = []
    for s in (1.5, 2.0, 4.0, math.inf):
        rf = required_fraction(target, s)
        rows.append({
            "kernel_speedup": "inf" if math.isinf(s) else s,
            "required_time_fraction_in_kernel": rf,
        })
    rows_by_fraction = []
    for f in (0.1, 0.2, 0.3, 0.5, 0.7):
        rk = required_kernel_speedup(target, f)
        rows_by_fraction.append({
            "time_fraction_in_kernel": f,
            "required_kernel_speedup": "unreachable" if math.isinf(rk) else rk,
        })
    print(json.dumps({
        "target_end_to_end_speedup": target,
        "by_kernel_speedup": rows,
        "by_time_fraction": rows_by_fraction,
        "note": "to reach target T a kernel must occupy more than 1 - 1/T of the time",
        "label": "arithmetic_not_measured",
    }, indent=2))
    return 0


def cmd_traffic(args):
    rows = []
    for ctx in args.ctx:
        for name, kv in (("fp16", KV_BYTES_PER_TOKEN_FP16), ("v8", KV_BYTES_PER_TOKEN_V8)):
            rows.append({
                "ctx": ctx,
                "kv_format": name,
                "forward_bytes": forward_bytes(ctx, kv),
                "kv_bandwidth_fraction": kv_traffic_fraction(ctx, kv),
            })
    print(json.dumps({
        "weight_bytes_per_forward": WEIGHT_BYTES_PER_VERIFY,
        "rows": rows,
        "warning": "kv_bandwidth_fraction is a bytes share, not an attention-compute share",
        "label": "modeled_from_documented_bytes_not_measured_time",
    }, indent=2))
    return 0


def cmd_options(args):
    rows = [
        option_end_to_end(opt, args.attention_fraction, args.mixer_fraction, args.kv_fraction)
        for opt in OPTION_CEILINGS
    ]
    print(json.dumps({
        "attention_fraction": args.attention_fraction,
        "mixer_fraction": args.mixer_fraction,
        "kv_fraction": args.kv_fraction,
        "options": rows,
        "reminder": "fractions must be measured on an M6; defaults are placeholders",
        "label": "inferred_ceilings_not_measured",
    }, indent=2))
    return 0


def _parse_speedup(text):
    if str(text).lower() in ("inf", "infinity"):
        return math.inf
    value = float(text)
    if value <= 0:
        raise ValueError("kernel_speedup must be positive")
    return value


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("amdahl", help="end-to-end speedup from a kernel fraction and speedup")
    a.add_argument("--fraction", type=float, required=True)
    a.add_argument("--kernel-speedup", default="2.0")

    r = sub.add_parser("required", help="what a target end-to-end speedup demands")
    r.add_argument("--target", type=float, default=2.0)

    t = sub.add_parser("traffic", help="modeled forward bytes and KV share by context")
    t.add_argument("--ctx", type=int, nargs="+", default=[8192, 16384, 32768, 49152, 65536])

    o = sub.add_parser("options", help="per-option end-to-end given on-device fractions")
    o.add_argument("--attention-fraction", type=float, default=0.25)
    o.add_argument("--mixer-fraction", type=float, default=0.0)
    o.add_argument("--kv-fraction", type=float, default=0.0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "amdahl":
        return cmd_amdahl(args)
    if args.cmd == "required":
        return cmd_required(args)
    if args.cmd == "traffic":
        return cmd_traffic(args)
    if args.cmd == "options":
        return cmd_options(args)
    raise ValueError(f"unhandled command {args.cmd!r}")


if __name__ == "__main__":
    sys.exit(main())
