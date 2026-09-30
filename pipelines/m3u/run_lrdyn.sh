#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
# M3U helper (M6 request 17:4x 2026-09-27): per-matrix rank allocation of the mixer low-rank factors, same parameter
# budget as uniform rank 64, full-covariance whitened spectra, blocks of 8, cap 256, rank 0 allowed.
#   scripts/qwen38_lowrank_dyn.py: phase A (spectra + whitening -> lrdyn_work, ~16 GB), B (allocations raw, rel), C (fit)
#   -> runs/export/mix25in_aw_cal_lrdyn_{raw,rel} -> KL each vs the stored plain baseline (0.1952).
#   A variant with KL <= 0.97 x baseline is moved to runs/export/mix25in_aw_cal_lrdyn (the best one); anything else
#   (for example better p99 / top-1 at equal KL) is reported to M6 to decide.
cd "$FORGE_ROOT/scripts" || exit 1
export PYTHONWARNINGS=ignore MODEL=$MODEL TRACE=$TRACE NCAL=48 \
  CAL_MIX="$FORGE_WORK_DIR/calib_chat_ids.npy:16,$FORGE_WORK_DIR/calib_pi_ids.npy:16"
PY="$FORGE_PYTHON"
L=$FORGE_WORK_DIR
E=$OUT/export
log() { echo "$(date +%H:%M:%S) $*"; }
kljson() { /usr/bin/python3 -c "import json;print(json.load(open('$TRACE/kl_$1.json'))['mean_kl'])" 2>/dev/null; }
step() { local name=$1 logf=$2; shift 2
  while pgrep -f "qwen38_kl.py|qwen38_gptq_27b.py|qwen38_lowrank" > /dev/null; do sleep 30; done
  log "START $name (free $(df -g "$FORGE_WORK_DIR" | awk 'NR==2{print $4}') GiB)"
  "$@" > $logf 2>&1 || { log "FAILED $name"; tail -5 $logf; exit 1; }
  log "OK $name: $(tail -1 $logf | cut -c1-330)"; }

step lrdyn $L/lowrank_dyn.log env EXPORT_DIR=$E/mix25in_aw_cal OUT_DIR=$E/mix25in_aw_cal_lrdyn WORK=$L/lrdyn_work \
  ALLOCS=raw,rel $PY -u qwen38_lowrank_dyn.py
grep -E "^(phase B|rank histogram|mean rank|top matrices)" $L/lowrank_dyn.log | cut -c1-400
for a in raw rel; do
  step kl_$a $L/kl_eval_mix25in_aw_cal_lrdyn_$a.log env EXPORT_DIR=$E/mix25in_aw_cal_lrdyn_$a \
    TAG=mix25in_aw_cal_lrdyn_$a $PY -u qwen38_kl.py eval
done
base=$(kljson mix25in_aw_cal_lr64mix_stored); base=${base:-0.1952}
best=""; bestkl=$base
for a in raw rel; do
  k=$(kljson mix25in_aw_cal_lrdyn_$a)
  log "KL $a = ${k:-missing} (baseline $base)"
  [[ -n $k ]] && (( k > 0 && k <= 0.97 * base && k < bestkl )) && { best=$a; bestkl=$k; }
done
if [[ -n $best && ! -e $E/mix25in_aw_cal_lrdyn ]]; then
  mv $E/mix25in_aw_cal_lrdyn_$best $E/mix25in_aw_cal_lrdyn && log "EXPORT mix25in_aw_cal_lrdyn = $best (KL $bestkl vs $base)"
else
  log "NO AUTO EXPORT: no allocation beats the baseline KL by >= 3% (M6 decides on p99 / top-1)"
fi
log "done"
