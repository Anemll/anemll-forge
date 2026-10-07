#!/usr/bin/env python3
"""Prefill + readout smoke for a local Jeff checkpoint.

    python forge.py jeff-smoke --model /Users/anemll/Models/jeff/jeff-base-v1.3
    python scripts/jeff_coreai_smoke.py --model ... [--build ~/Models/jeff-coreai/coreai]

Without --build, runs the host hybrid DecodeLayer + 255-way readout (no Core AI).
With --build on macOS, loads the prefill-only Core AI package when the SDK is present.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai import (JEFF_DEFAULT, JeffCheckpoint, encode_prompt, host_decision,
                         render_jeff_prompt)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=JEFF_DEFAULT)
    p.add_argument("--build", type=Path, help="coreai/ directory from jeff-convert")
    p.add_argument("--state", default="The disk on db-02 is 97 percent full and still growing.")
    p.add_argument("--options", default="page,wait,ignore")
    p.add_argument("--instructions", default="Choose the best next action.")
    p.add_argument("--ids", help="comma-separated token ids; skips tokenizer + prompt render")
    return p


def coreai_decision(build: Path, model: Path, token_ids: list[int], n_options: int) -> dict:
    from jeff_coreai_runtime import try_coreai_decision
    return try_coreai_decision(build, model, token_ids, n_options)


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    model = a.model.expanduser().resolve()
    ck = JeffCheckpoint(model)
    options = [x.strip() for x in a.options.split(",") if x.strip()]
    if a.ids:
        token_ids = [int(x) for x in a.ids.split(",") if x.strip()]
        text = None
    else:
        text = render_jeff_prompt(a.state, options, a.instructions)
        token_ids = encode_prompt(model, text)
    result = host_decision(ck, token_ids, len(options))
    result["prompt"] = text
    if a.build:
        build = a.build.expanduser().resolve()
        try:
            result["coreai"] = coreai_decision(build, model, token_ids, len(options))
        except Exception as e:  # noqa: BLE001  smoke: report host result plus why Core AI did not run
            result["coreai_error"] = str(e)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
