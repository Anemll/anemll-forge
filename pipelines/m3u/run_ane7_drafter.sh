#!/bin/zsh
# M3U helper: DFlash2 drafter re-calibration on the ane7 target (mix25in_aw_cal_lr64mix, KL 0.195). M6 decision, 12:5x
# 2026-09-27: runs only if the ane6 quant eval shows lut4_gptq >= 1% better than lut4_rtn in tokens per call
# (mean_emitted). Starts after run_dflash_now.sh has finished. Touch $L/.hold_dflash to pause before the next step.
#   dequant -> sim q7_cal (32 calib prompts x 256) -> quant eval (lut4_rtn, lut4_gptq; head = mix25in lm_head)
#   -> lut4_gptq export if it beats RTN. Output dirs: drafter_lut4_gptq_q_cal (ane6, renamed) / drafter_lut4_gptq_q7_cal.
cd ~/SourceRelease/GITHUB/ML_playground/ane-vector-lut/scripts
export PYTHONWARNINGS=ignore MODEL=/Volumes/SN8100/Qwen3.8-27B WORK=/Volumes/SN8100/dflash2_work THREADS=16
PY=~/venvs/vq27b/bin/python
L=/Volumes/SN8100/vq27b
E=$L/runs/export
W=$WORK
T7=mix25in_aw_cal_lr64mix
HEAD7=$E/mix25in_aw_cal/lm_head.safetensors
log() { echo "$(date +%H:%M:%S) $*"; }
free_gb() { df -g /Volumes/SN8100 | awk 'NR==2{print $4}'; }
need() { local g=$(free_gb); (( g >= $1 )) || { log "STOP: ${g} GiB free on SN8100, need $1 for $2"; exit 1; }; }
step() { local name=$1 logf=$2; shift 2
  while [[ -f $L/.hold_dflash ]]; do sleep 60; done
  log "START $name (free $(free_gb) GiB)"
  "$@" > $logf 2>&1 || { log "FAILED $name"; tail -5 $logf; exit 1; }
  log "OK $name: $(tail -1 $logf | cut -c1-300)"; }

log "waiting for run_dflash_now.sh (ane6 quant eval + export) to finish"
while pgrep -f run_dflash_now.sh > /dev/null; do sleep 60; done
gain=$(/usr/bin/python3 -c "import json;r=json.load(open('$W/quant_eval_q_cal_headlut4.json'));print(r['lut4_gptq']['mean_emitted']/r['lut4_rtn']['mean_emitted']-1)" 2>/dev/null)
if [[ -z $gain ]] || (( gain < 0.01 )); then
  log "SKIP ane7 drafter: lut4_gptq vs lut4_rtn tokens/call gain = ${gain:-missing} (< 1%)"; exit 0
fi
log "ane7 drafter: lut4_gptq gain on ane6 traces = $gain (>= 1%)"
$L/dflash_guard_ane7.sh &

if [[ $(ls $W/deq_$T7 2>/dev/null | grep -c '^layer_') != 64 || ! -f $W/deq_$T7/lm_head.safetensors ]]; then
  need 70 "dequant $T7 (~48 GB + 20 GB reserve)"
  step dequant7 $W/dequant_$T7.log env EXPORT_DIR=$E/$T7 DEQ_OUT=$W/deq_$T7 $PY -u dflash2_target_ref.py dequant_export
fi
n=$(ls $W/deq_$T7 | grep -c '^layer_')
[[ $n == 64 && -f $W/deq_$T7/lm_head.safetensors ]] || { log "STOP: dequant incomplete ($n layers)"; exit 1; }

step sim_q7_cal $W/sim_q7_cal.log env DEQ_DIR=$W/deq_$T7 TAG=q7_cal PROMPT_SET=calib N_PROMPTS=32 MAX_NEW=256 \
  $PY -u dflash2_target_ref.py simulate
step quant_eval7 $W/quant_eval_q7_cal.log env TRACE_TAG=q7_cal VARIANTS=lut4_rtn,lut4_gptq HEAD=lut4 HEAD_EXPORT=$HEAD7 \
  $PY -u dflash2_quant.py eval

better=$(/usr/bin/python3 -c "import json;r=json.load(open('$W/quant_eval_q7_cal_headlut4.json'));print(int(r['lut4_gptq']['mean_accepted']>r['lut4_rtn']['mean_accepted']))" 2>/dev/null)
if [[ $better == 1 ]]; then
  # dflash2_quant.py export always writes $W/drafter_lut4_gptq: keep the ane6 one under its own name first
  [[ -d $W/drafter_lut4_gptq && ! -e $W/drafter_lut4_gptq_q_cal ]] && mv $W/drafter_lut4_gptq $W/drafter_lut4_gptq_q_cal && log "renamed ane6 export -> drafter_lut4_gptq_q_cal"
  step export7 $W/export_lut4_gptq_q7_cal.log env TRACE_TAG=q7_cal VARIANT=lut4_gptq HEAD_EXPORT=$HEAD7 \
    $PY -u dflash2_quant.py export
  mv $W/drafter_lut4_gptq $W/drafter_lut4_gptq_q7_cal && log "ane7 export -> $W/drafter_lut4_gptq_q7_cal"
else
  log "SKIP ane7 drafter export (lut4_gptq does not beat lut4_rtn on q7_cal traces)"
fi
log "done"
