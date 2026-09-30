#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
# M3U helper follow-up queue (v2, 09:4x 2026-09-27). The DFlash2 steps 1-5 moved to run_dflash_now.sh (running next to
# run_quality5). This script only does:
#   A. mix25in_aw_cal_lr64mix factor export, if KL(mix25in_aw_cal) < 0.310, once run_quality5 has finished
#   B. sim q_old_cal (old target, diagnostic), when M3U is otherwise idle (run_quality5 and run_dflash_now.sh finished)
# Touch $L/.hold to pause it before its next step.
cd "$FORGE_ROOT/scripts" || exit 1
export PYTHONWARNINGS=ignore MODEL=$MODEL WORK=$WORK
PY="$FORGE_PYTHON"
L=$FORGE_WORK_DIR
E=$OUT/export
W=$WORK
log() { echo "$(date +%H:%M:%S) $*"; }
free_gb() { df -g "$FORGE_WORK_DIR" | awk 'NR==2{print $4}'; }
need() { local g=$(free_gb); (( g >= $1 )) || { log "STOP: ${g} GiB free on SN8100, need $1 for $2"; exit 1; }; }
q5_busy() { pgrep -f "run_quality5.sh|qwen38_gptq_27b.py|qwen38_kl.py" > /dev/null || [[ -f $L/.hold ]]; }
step() { local name=$1 logf=$2; shift 2
  log "START $name"
  "$@" > $logf 2>&1 || { log "FAILED $name"; tail -5 $logf; exit 1; }
  log "OK $name: $(tail -1 $logf | cut -c1-300)"; }

log "v2: waiting for run_quality5 to finish"
while q5_busy; do sleep 60; done
log "run_quality5 finished: $(tail -1 $L/run_quality5.out)"

kl_in=$(/usr/bin/python3 -c "import json;print(json.load(open('$TRACE/kl_mix25in_aw_cal.json'))['mean_kl'])" 2>/dev/null)
if [[ -n $kl_in ]] && (( kl_in > 0 && kl_in < 0.310 )); then
  need 30 "mix25in lr64mix"
  step lr64_mix25in $L/lowrank_mix25in_aw_cal.log env EXPORT_DIR=$E/mix25in_aw_cal OUT_DIR=$E/mix25in_aw_cal_lr64mix \
    LR_RANK=64 PARTS=gdn,attn $PY -u qwen38_lowrank_export.py
else
  log "SKIP mix25in_aw_cal_lr64mix export (KL mix25in_aw_cal = ${kl_in:-missing})"
fi

log "waiting for run_dflash_now.sh to finish before the diagnostic sim q_old_cal"
while pgrep -f run_dflash_now.sh > /dev/null || q5_busy; do sleep 60; done
MAX_NEW=256 N_PROMPTS=32; [[ -f $W/after_q5.env ]] && source $W/after_q5.env
step sim_q_old_cal $W/sim_q_old_cal.log env DEQ_DIR=$W/deq_full_mix25_mixer4_head4 TAG=q_old_cal PROMPT_SET=calib \
  N_PROMPTS=$N_PROMPTS MAX_NEW=$MAX_NEW $PY -u dflash2_target_ref.py simulate
log "done"
