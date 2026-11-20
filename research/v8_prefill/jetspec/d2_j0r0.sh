#!/bin/zsh
# C4: LUT4 on the JetSpec++ adapter (head bf16), J3-R3, gsm8k / humaneval / mbpp 0-7: per-tensor LUT4 + row scale (RTN),
# + Hadamard (RTN), + Hadamard + GPTQ (calibrated on gsm8k 8-15)
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov
J=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/jetspec
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
C=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp
MULTI=gsm8k=$C/capture_gsm8k32_20nov/traces,humaneval=$C/capture_humaneval8_20nov/traces,mbpp=$C/capture_mbpp8_20nov/traces
specs=(lut4t:adapter:lut4t:- lut4h:adapter:lut4rf:$C/adapter_lut4h lut4hg:adapter:lut4rf:$C/adapter_lut4hg)
[ -n "${ONLY:-}" ] && specs=(${(M)specs:#${ONLY}:*})
for spec in $specs; do
  name=${spec%%:*}; rest=${spec#*:}; q=${rest%:*}; dir=${rest##*:}; tag=d2_${name}_J0-R0
  echo "$(date "+%H:%M") D2 replay $name (--quant $q, LUT dir $dir)" >> $R/ops.log
  echo "STATE=working ITEM=D2-J0R0-lut4-replay NOTE=replay $name J0-R0" > /tmp/m6_yolo_status.txt
  JSP_LUT_DIR=$dir $PY $J/jsp_replay.py --traces $C/capture_gsm8k32_20nov/traces --multi $MULTI --limit 8 --max-new 2048 \
    --rows outcome --schedule J0-R0 --quant $q --out $R/d2/$tag.json > $R/d2/$tag.log 2>&1 & BP=$!
  $WD $BP 6144 $R/d2/$tag.watchdog.log >/dev/null 2>&1 &
  wait $BP; echo "$tag rc $?"
done
echo ALLDONE
