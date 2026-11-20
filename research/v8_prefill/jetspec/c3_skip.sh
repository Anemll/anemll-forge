#!/bin/zsh
# C3: JetSpec++ with adapter blocks skipped (identity), bf16, gsm8k / humaneval / mbpp 0-7
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov
J=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/jetspec
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
C=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp
MULTI=gsm8k=$C/capture_gsm8k32_20nov/traces,humaneval=$C/capture_humaneval8_20nov/traces,mbpp=$C/capture_mbpp8_20nov/traces
E3=extra_stream_layers.3.0,extra_stream_layers.3.1,extra_stream_layers.3.2
E4=extra_stream_layers.4.0,extra_stream_layers.4.1,extra_stream_layers.4.2,extra_stream_layers.4.3
mkdir -p $R/c3
for spec in "all7:$E3,$E4:J1-R1" "all7:$E3,$E4:J0-R0" "s4x:$E4:J1-R1"; do
  name=${spec%%:*}; rest=${spec#*:}; blocks=${rest%:*}; sch=${rest##*:}
  tag=c3_skip_${name}_${sch}
  echo "$(date '+%H:%M') C3 replay skip $name ($sch)" >> $R/ops.log
  echo "STATE=working ITEM=C3-cheaper-adapter NOTE=replay skip $name schedule $sch" > /tmp/m6_yolo_status.txt
  JSP_SKIP_BLOCKS=$blocks $PY $J/jsp_replay.py --traces $C/capture_gsm8k32_20nov/traces --multi $MULTI --limit 8 --max-new 2048 \
    --rows outcome --schedule $sch --out $R/c3/$tag.json > $R/c3/$tag.log 2>&1 & BP=$!
  $WD $BP 6144 $R/c3/$tag.watchdog.log >/dev/null 2>&1 &
  wait $BP; echo "$tag rc $?"
done
echo ALLDONE
