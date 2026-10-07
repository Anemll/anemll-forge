#!/usr/bin/env python3
"""Convert a local Jeff / Qwen3.5-0.8B decision checkpoint to a prefill-only Core AI package.

    python forge.py jeff-convert --model /Users/anemll/Models/jeff/jeff-base-v1.3 --output ~/Models/jeff-coreai
    python scripts/jeff_coreai_convert.py --model ... --output ... [--quant fp16|int8] [--dry-run]

Does not download weights. Does not run GPTQ/VQ or build DFlash2. Core AI export needs the
conversion SDK (macOS). --dry-run only prints the config-driven plan.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai import (JEFF_DEFAULT, JeffCheckpoint, convert_plan, is_jeff_decision_checkpoint, load_text_config,
                         prefill_widths)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=JEFF_DEFAULT,
                   help="local Jeff checkpoint (config.json, model.safetensors, readout.safetensors)")
    p.add_argument("--output", type=Path, required=True, help="new empty directory: model/ + coreai/")
    p.add_argument("--ctx", type=int, default=2048, help="KV history length of the prefill entry")
    p.add_argument("--prefill", type=int, default=256, help="largest prefill rows (multiple of 8, > 8)")
    p.add_argument("--prefill-extra", default="",
                   help="extra prefill widths compiled in the same packages, comma-separated (e.g. 32,64)")
    p.add_argument("--quant", choices=("fp16", "int8"), default="fp16",
                   help="fp16 dense (default) or per-channel INT8 projections; not GPTQ/VQ")
    p.add_argument("--chunk-layers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    model = a.model.expanduser().resolve()
    out = a.output.expanduser().resolve()
    if not (model / "config.json").is_file():
        raise SystemExit(f"Missing {model / 'config.json'}")
    cfg = load_text_config(model)
    if not is_jeff_decision_checkpoint(model, cfg):
        raise SystemExit("Need a Qwen3.5 hybrid checkpoint with readout.safetensors "
                         f"(layer_types + readout) at {model}")
    ck = JeffCheckpoint(model)
    extra = [int(part) for part in a.prefill_extra.split(",") if part.strip()]
    widths = prefill_widths(a.prefill, extra)
    plan = convert_plan(ck, a.ctx, a.prefill, a.quant, a.chunk_layers, widths)
    if a.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Use a new or empty --output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    from jeff_coreai_build import export_jeff
    result = export_jeff(ck, out, a.ctx, a.prefill, a.quant, a.chunk_layers, widths)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
