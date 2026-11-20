#!/bin/zsh
# B7 (20 Nov 2026): JetSpec++ quant ladder by replay, one process per setting over gsm8k 0-7 / humaneval 0-7 / mbpp 0-7
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov
J=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/jetspec
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
C=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp
MULTI=gsm8k=$C/capture_gsm8k32_20nov/traces,humaneval=$C/capture_humaneval8_20nov/traces,mbpp=$C/capture_mbpp8_20nov/traces
free_gb() { vm_stat | awk '/Pages free/ {gsub("\\.","",$NF); printf "%d", $NF*16384/1e9}'; }
mkdir -p $R/b7
for q in ${=QS:-adapter:8c,head:8c adapter:6,head:6 adapter:lut4 adapter:lut3}; do
  tag=b7_${q//[:,]/_}
  for i in $(seq 1 60); do [ $(free_gb) -ge 12 ] && break; sleep 10; done
  echo "$(date '+%H:%M') B7 replay --quant $q ($tag), gsm8k / humaneval / mbpp 0-7, free $(free_gb) GB" >> $R/ops.log
  echo "STATE=working ITEM=B7-quant-ladder NOTE=replay $q (gsm8k/humaneval/mbpp 0-7)" > /tmp/m6_yolo_status.txt
  $PY $J/jsp_replay.py --traces $C/capture_gsm8k32_20nov/traces --multi $MULTI --limit 8 --max-new 2048 --rows outcome \
    --schedule J3-R3 --quant $q --out $R/b7/$tag.json > $R/b7/$tag.log 2>&1 & BP=$!
  $WD $BP 6144 $R/b7/$tag.watchdog.log >/dev/null 2>&1 &
  wait $BP; echo "$tag rc $?"
done
echo ALLDONE
