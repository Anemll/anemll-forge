#!/bin/zsh
# B8 (20 Nov 2026): JetSpec++ acceptance by schedule (bf16), gsm8k / humaneval / mbpp 0-7, one process per schedule
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov
J=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/jetspec
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
C=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp
MULTI=gsm8k=$C/capture_gsm8k32_20nov/traces,humaneval=$C/capture_humaneval8_20nov/traces,mbpp=$C/capture_mbpp8_20nov/traces
free_gb() { vm_stat | awk '/Pages free/ {gsub("\\.","",$NF); printf "%d", $NF*16384/1e9}'; }
mkdir -p $R/b8
for s in ${=SCHEDS:-J1-R1 J1-R0 J0-R0}; do
  tag=b8_sched_${s}
  for i in $(seq 1 60); do [ $(free_gb) -ge 12 ] && break; sleep 10; done
  echo "$(date '+%H:%M') B8 replay schedule $s (bf16), gsm8k / humaneval / mbpp 0-7" >> $R/ops.log
  echo "STATE=working ITEM=B8-schedules NOTE=replay schedule $s (bf16) to get tau vs drafter passes" > /tmp/m6_yolo_status.txt
  $PY $J/jsp_replay.py --traces $C/capture_gsm8k32_20nov/traces --multi $MULTI --limit 8 --max-new 2048 --rows outcome \
    --schedule $s --out $R/b8/$tag.json > $R/b8/$tag.log 2>&1 & BP=$!
  $WD $BP 6144 $R/b8/$tag.watchdog.log >/dev/null 2>&1 &
  wait $BP; echo "$tag rc $?"
done
echo ALLDONE
