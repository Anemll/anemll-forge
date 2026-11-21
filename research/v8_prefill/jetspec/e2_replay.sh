#!/bin/zsh
# E2 (20 Nov 2026): acceptance of the W4A8 adapter MLPs by replay: LUT4 Hadamard + GPTQ weights (spec lut4rf,
# adapter_lut4hg) + INT8 activations into gate / up / down in each matrix's rotated basis (JSP_ACTQ: tok = per token,
# tensor = calibrated constant per matrix input, e2/scales/act_scales_tensor.json), head bf16, schedule $SCHED,
# gsm8k / humaneval / mbpp 0-7 (the C4 / D2 sets). Forge must be down (one model at a time).
#   SCHED=J0-R0 [ONLY=tok|tensor] zsh e2_replay.sh
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov
J=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/jetspec
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
C=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp
MULTI=gsm8k=$C/capture_gsm8k32_20nov/traces,humaneval=$C/capture_humaneval8_20nov/traces,mbpp=$C/capture_mbpp8_20nov/traces
SCHED=${SCHED:-J0-R0}
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "forge listens on 8765: stop it first"; exit 1; }
for v in ${=ONLY:-tok tensor}; do
  tag=e2_w4a8${v}_$SCHED
  echo "$(date "+%H:%M") E2 replay $tag (--quant adapter:lut4rf, JSP_ACTQ=$v)" >> $R/ops.log
  echo "STATE=working ITEM=E2-replay NOTE=replay $tag (W4A8 acceptance; forge down)" > /tmp/m6_yolo_status.txt
  JSP_LUT_DIR=$C/adapter_lut4hg JSP_ACTQ=$v JSP_ACTQ_SCALES=$R/e2/scales/act_scales_tensor.json \
    $PY $J/jsp_replay.py --traces $C/capture_gsm8k32_20nov/traces --multi $MULTI --limit 8 --max-new 2048 \
    --rows outcome --schedule $SCHED --quant adapter:lut4rf --out $R/e2/$tag.json > $R/e2/$tag.log 2>&1 & BP=$!
  $WD $BP 6144 $R/e2/$tag.watchdog.log >/dev/null 2>&1 &
  wait $BP; echo "$tag rc $?"
  grep SUMMARY $R/e2/$tag.log
done
echo ALLDONE
