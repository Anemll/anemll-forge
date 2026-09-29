#!/bin/zsh
# M3U helper: DFlash2 drafter re-calibration, run NOW next to run_quality5 (M6 request 09:3x, 2026-09-27).
#   1. mix25_aw_cal_lr64mix factors  2. dequant -> $WORK/deq_mix25_aw_cal_lr64mix  3. simulate q_cal (new target)
#   4. drafter quant eval (lut4_rtn vs lut4_gptq)  5. lut4_gptq export if it wins
# CPU jobs with THREADS=12 (the GPU queue keeps some CPU). A memory guard SIGSTOPs the running step when memory
# pressure rises or swap grows by more than 6 GB, and resumes it when pressure is normal and no KL eval is running.
# Touch $L/.hold_dflash to pause before the next step. sim q_old_cal is left to run_after_q5.sh (idle time).
cd ~/SourceRelease/GITHUB/ML_playground/ane-vector-lut/scripts
export PYTHONWARNINGS=ignore MODEL=/Volumes/SN8100/Qwen3.8-27B WORK=/Volumes/SN8100/dflash2_work THREADS=12
PY=~/venvs/vq27b/bin/python
L=/Volumes/SN8100/vq27b
E=$L/runs/export
W=$WORK
log() { echo "$(date +%H:%M:%S) $*"; }
free_gb() { df -g /Volumes/SN8100 | awk 'NR==2{print $4}'; }
swap_mb() { sysctl -n vm.swapusage | awk '{print int($6)}'; }
need() { local g=$(free_gb); (( g >= $1 )) || { log "STOP: ${g} GiB free on SN8100, need $1 for $2"; exit 1; }; }
step() { local name=$1 logf=$2; shift 2
  while [[ -f $L/.hold_dflash ]]; do sleep 60; done
  log "START $name (free $(free_gb) GiB, swap $(swap_mb) MB)"
  "$@" > $logf 2>&1 || { log "FAILED $name"; tail -5 $logf; exit 1; }
  log "OK $name: $(tail -1 $logf | cut -c1-300)"; }

guard() {
  local base=$(swap_mb) stopped="" p s pid
  while true; do
    p=$(sysctl -n kern.memorystatus_vm_pressure_level); s=$(swap_mb)
    pid=$(pgrep -f "dflash2_target_ref.py|dflash2_quant.py|qwen38_lowrank_export.py" | head -1)
    if [[ -z $stopped && -n $pid ]] && (( p >= 2 || s > base + 6144 )); then
      kill -STOP $pid && stopped=$pid && log "GUARD pause pid $pid (pressure $p, swap ${s} MB, base ${base} MB)"
    elif [[ -n $stopped ]] && (( p == 1 )) && ! pgrep -f qwen38_kl.py > /dev/null; then
      kill -CONT $stopped; log "GUARD resume pid $stopped (pressure $p, swap ${s} MB)"; stopped=""; base=$(swap_mb)
    fi
    sleep 20
  done
}
guard &
GUARD_PID=$!
trap 'kill $GUARD_PID 2>/dev/null' EXIT

# 1. factors for the target now on the M6 ANE (mix25_aw_cal)
need 30 "mix25 lr64mix"
step lr64_mix25 $W/lowrank_mix25_aw_cal.log env EXPORT_DIR=$E/mix25_aw_cal OUT_DIR=$E/mix25_aw_cal_lr64mix \
  LR_RANK=64 PARTS=gdn,attn $PY -u qwen38_lowrank_export.py

# 2. dequantize once (~50 GB fp16), keep >= 20 GB free (+ ~26 GB for the mix25in GPTQ export and factors still to come)
if [[ $(ls $W/deq_mix25_aw_cal_lr64mix 2>/dev/null | grep -c '^layer_') != 64 || ! -f $W/deq_mix25_aw_cal_lr64mix/lm_head.safetensors ]]; then
  need 100 "dequant (~50 GB + 26 GB queued exports + 20 GB reserve)"
  step dequant $W/dequant_mix25_aw_cal_lr64mix.log env EXPORT_DIR=$E/mix25_aw_cal_lr64mix DEQ_OUT=$W/deq_mix25_aw_cal_lr64mix \
    $PY -u dflash2_target_ref.py dequant_export
fi
n=$(ls $W/deq_mix25_aw_cal_lr64mix | grep -c '^layer_')
[[ $n == 64 && -f $W/deq_mix25_aw_cal_lr64mix/lm_head.safetensors ]] || { log "STOP: dequant incomplete ($n layers)"; exit 1; }

# 3. greedy speculative decoding with the bf16 drafter on the calib prompts, new target
MAX_NEW=256 N_PROMPTS=32; [[ -f $W/after_q5.env ]] && source $W/after_q5.env
step sim_q_cal $W/sim_q_cal.log env DEQ_DIR=$W/deq_mix25_aw_cal_lr64mix TAG=q_cal PROMPT_SET=calib \
  N_PROMPTS=$N_PROMPTS MAX_NEW=$MAX_NEW $PY -u dflash2_target_ref.py simulate

# 4. drafter quantization on the new target's traces (2-fold replay acceptance)
step quant_eval $W/quant_eval_q_cal.log env TRACE_TAG=q_cal VARIANTS=lut4_rtn,lut4_gptq HEAD=lut4 \
  HEAD_EXPORT=$E/mix25_aw_cal/lm_head.safetensors $PY -u dflash2_quant.py eval

# 5. export the GPTQ drafter if it beats RTN
better=$(/usr/bin/python3 -c "import json;r=json.load(open('$W/quant_eval_q_cal_headlut4.json'));print(int(r['lut4_gptq']['mean_accepted']>r['lut4_rtn']['mean_accepted']))" 2>/dev/null)
if [[ $better == 1 ]]; then
  step export_lut4_gptq $W/export_lut4_gptq_q_cal.log env TRACE_TAG=q_cal VARIANT=lut4_gptq \
    HEAD_EXPORT=$E/mix25_aw_cal/lm_head.safetensors $PY -u dflash2_quant.py export
else
  log "SKIP drafter export (lut4_gptq does not beat lut4_rtn, or the eval json is missing)"
fi
log "done"
