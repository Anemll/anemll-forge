#!/bin/zsh
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov
J=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/jetspec
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
C=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp
MULTI=gsm8k=$C/capture_gsm8k32_20nov/traces,humaneval=$C/capture_humaneval8_20nov/traces,mbpp=$C/capture_mbpp8_20nov/traces
for s in J3-R3 J0-R0; do
  tag=b8_lut6t_${s}
  echo "$(date '+%H:%M') B8 replay --quant adapter:lut6t,head:lut6t schedule $s" >> $R/ops.log
  echo "STATE=working ITEM=B8-lut6-accept NOTE=replay ANE-form LUT6 adapter+head, schedule $s" > /tmp/m6_yolo_status.txt
  JSP_LUT_DIR=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/adapter_lut6 $PY $J/jsp_replay.py --traces $C/capture_gsm8k32_20nov/traces --multi $MULTI --limit 8 --max-new 2048 --rows outcome \
    --schedule $s --quant adapter:lut6f,head:lut6t --out $R/b8/$tag.json > $R/b8/$tag.log 2>&1 & BP=$!
  $WD $BP 6144 $R/b8/$tag.watchdog.log >/dev/null 2>&1 &
  wait $BP; echo "$tag rc $?"
done
echo ALLDONE
