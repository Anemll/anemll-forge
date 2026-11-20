#!/bin/zsh
# 4-bit / 3-bit fake-quantized JetSpec++ drafter (adapter + B16 head, RTN g128) on the first 8 gsm8k traces (20 Nov 2026)
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov
J=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/jetspec
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
CAP=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/capture_gsm8k32_20nov
free_gb() { vm_stat | awk '/Pages free/ {gsub("\\.","",$NF); printf "%d", $NF*16384/1e9}'; }
for q in ${=QS:-adapter:4,head:4 adapter:3,head:4}; do
  tag=q_${q//[:,]/_}
  for i in $(seq 1 60); do [ $(free_gb) -ge 18 ] && break; sleep 10; done
  echo "$(date '+%H:%M') replay gsm8k 0-7 with --quant $q ($tag)" >> $R/ops.log
  $PY $J/jsp_replay.py --traces $CAP/traces --limit 8 --max-new 2048 --rows outcome --schedule J3-R3 --dataset gsm8k \
    --quant $q --out $R/jetspec/replay_$tag.json > $R/jetspec/replay_$tag.log 2>&1 & BP=$!
  $WD $BP 6144 $R/jetspec/replay_$tag.watchdog.log >/dev/null 2>&1 &
  wait $BP; echo "$tag rc $?"
done
echo ALLDONE
