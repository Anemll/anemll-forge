#!/bin/zsh
O=/Volumes/SSD4TB/anemll-forge-research/a8probe
cd /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/w8a8
for l in 36 4 20 60; do
  echo "== layer $l"
  MODEL=/Volumes/SSD4TB/Qwen3.8-27B /Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python a8_ffn_real_probe.py --out $O --layer $l --n 1024 --act $O/hid2k_L$l.npz || exit 1
done
