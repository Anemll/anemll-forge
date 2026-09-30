#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
# M3U helper (M6 request 16:1x 2026-09-27): activation-weighted rank-64 mixer factors for mix25in_aw_cal.
#   full (QERA-style whitening, damp 0.01) -> KL, then diag (sqrt E[x^2]) -> KL. Baseline = the stored plain factors
#   (mix25in_aw_cal_lr64mix, KL eval tag mix25in_aw_cal_lr64mix_stored). A variant with KL <= 0.97 x baseline is moved
#   to runs/export/mix25in_aw_cal_lr64aw for the M6 (the best one if both qualify). Script: scripts/qwen38_lowrank_aw.py.
cd "$FORGE_ROOT/scripts" || exit 1
export PYTHONWARNINGS=ignore MODEL=$MODEL TRACE=$TRACE NCAL=48 \
  CAL_MIX="$FORGE_WORK_DIR/calib_chat_ids.npy:16,$FORGE_WORK_DIR/calib_pi_ids.npy:16"
PY="$FORGE_PYTHON"
L=$FORGE_WORK_DIR
E=$OUT/export
SRC=$E/mix25in_aw_cal
log() { echo "$(date +%H:%M:%S) $*"; }
kljson() { /usr/bin/python3 -c "import json;print(json.load(open('$TRACE/kl_$1.json'))['mean_kl'])" 2>/dev/null; }
step() { local name=$1 logf=$2; shift 2
  while pgrep -f "qwen38_kl.py|qwen38_gptq_27b.py|qwen38_lowrank" > /dev/null; do sleep 30; done
  log "START $name (free $(df -g "$FORGE_WORK_DIR" | awk 'NR==2{print $4}') GiB)"
  "$@" > $logf 2>&1 || { log "FAILED $name"; tail -5 $logf; exit 1; }
  log "OK $name: $(tail -1 $logf | cut -c1-330)"; }

for mode in full diag; do
  step lr_$mode $L/lowrank_aw_$mode.log env EXPORT_DIR=$SRC OUT_DIR=$E/mix25in_aw_cal_lr64aw_$mode MODE=$mode \
    $PY -u qwen38_lowrank_aw.py
  step kl_$mode $L/kl_eval_mix25in_aw_cal_lr64aw_$mode.log env EXPORT_DIR=$E/mix25in_aw_cal_lr64aw_$mode \
    TAG=mix25in_aw_cal_lr64aw_$mode $PY -u qwen38_kl.py eval
done

base=$(kljson mix25in_aw_cal_lr64mix_stored); base=${base:-0.1946}
best=""; bestkl=$base
for mode in full diag; do
  k=$(kljson mix25in_aw_cal_lr64aw_$mode)
  log "KL $mode = ${k:-missing} (baseline plain stored factors $base)"
  [[ -n $k ]] && (( k > 0 && k <= 0.97 * base && k < bestkl )) && { best=$mode; bestkl=$k; }
done
if [[ -n $best && ! -e $E/mix25in_aw_cal_lr64aw ]]; then
  mv $E/mix25in_aw_cal_lr64aw_$best $E/mix25in_aw_cal_lr64aw && log "EXPORT mix25in_aw_cal_lr64aw = $best (KL $bestkl vs $base)"
else
  log "NO EXPORT: no variant beats the baseline by >= 3% (best ${best:-none})"
fi
log "done"
