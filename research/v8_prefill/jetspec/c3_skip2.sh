#!/bin/zsh
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov
J=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/jetspec
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
C=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp
for spec in "s43:extra_stream_layers.4.3" "s32s43:extra_stream_layers.3.2,extra_stream_layers.4.3"; do
  name=${spec%%:*}; blocks=${spec#*:}; tag=c3_skip_${name}_J1-R1
  echo "$(date '+%H:%M') C3 replay skip $name (J1-R1, gsm8k 0-7)" >> $R/ops.log
  echo "STATE=working ITEM=C3-cheaper-adapter NOTE=replay skip $name J1-R1 gsm8k" > /tmp/m6_yolo_status.txt
  JSP_SKIP_BLOCKS=$blocks $PY $J/jsp_replay.py --traces $C/capture_gsm8k32_20nov/traces --multi gsm8k=$C/capture_gsm8k32_20nov/traces \
    --limit 8 --max-new 2048 --rows outcome --schedule J1-R1 --out $R/c3/$tag.json > $R/c3/$tag.log 2>&1 & BP=$!
  $WD $BP 6144 $R/c3/$tag.watchdog.log >/dev/null 2>&1 &
  wait $BP; echo "$tag rc $?"
done
echo ALLDONE
