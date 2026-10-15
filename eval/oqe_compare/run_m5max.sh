#!/bin/bash
# Single-command KL comparison on an M5 Max (macOS, Apple Silicon).
# Needs ~60 GiB free unified memory for the BF16 Qwen3.8-27B reference.
# Do not run this on the Linux CI / Cloud Agent VM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="${OQE_COMPARE_OUT:-/Users/anemll/Models/anemll-forge/oqe_compare}"
exec python3 eval/oqe_compare/run_compare.py --preset m5max --out "$OUT" "$@"
