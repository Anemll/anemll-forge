#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
# M3U helper: mixer -> MLP byte trade, final phase (after run_mixr.sh's band KLs).
#   plan (scripts/qwen38_plan_mixr.py -> plan_mixr.json) -> GPTQ mix25in_mixr (same env as mix25in_aw_cal, PLAN=plan_mixr)
#   -> rank-64 plain factors (qwen38_lowrank_export.py) -> runs/export/mix25in_mixr_lr64mix -> KL vs 0.1952 (stored baseline).
#   The export stays in place only if KL <= 0.97 x baseline; otherwise it is renamed *_rejected for M6 to inspect.
cd "$FORGE_ROOT/scripts" || exit 1
export PYTHONWARNINGS=ignore TRACE=$TRACE MODEL=$MODEL OUT=$OUT
PY="$FORGE_PYTHON"
L=$FORGE_WORK_DIR
E=$OUT/export
log() { echo "$(date +%H:%M:%S) $*"; }
busy() { pgrep -f "qwen38_kl.py|qwen38_gptq_27b.py|qwen38_lowrank" > /dev/null; }
while busy; do sleep 30; done
$PY qwen38_plan_mixr.py --kl-dir "$TRACE" --sweep "$OUT/sweep_mlp_vector_2x16_+_pcs_online.json" --plan "$L/plan_indomain.json" --out $L/plan_mixr.json > $L/plan_mixr.log 2>&1 || { log "FAILED plan"; cat $L/plan_mixr.log; exit 1; }
grep -E "TAKE|skip|->|net|estimated" $L/plan_mixr.log
log "GPTQ mix25in_mixr"
env MIXER="LUT4 per-tensor + pcs" KV_FMT="INT8 per-channel" HEAD="LUT4 per-tensor + pcs" BASELINE=0 AW=1 NCAL=48 \
  CAL_MIX="$L/calib_chat_ids.npy:16,$L/calib_pi_ids.npy:16" PLAN=$L/plan_mixr.json TAG=mix25in_mixr \
  $PY -X faulthandler -u qwen38_gptq_27b.py > $L/run_mix25in_mixr.log 2>&1 || { log "FAILED GPTQ"; tail -5 $L/run_mix25in_mixr.log; exit 1; }
log "$(tail -1 $L/run_mix25in_mixr.log)"
n=$(ls $E/mix25in_mixr | wc -l | tr -d ' ')
[[ $n == 129 ]] || { log "STOP: mix25in_mixr has $n files, expected 129"; exit 1; }
env EXPORT_DIR=$E/mix25in_mixr OUT_DIR=$E/mix25in_mixr_lr64mix LR_RANK=64 PARTS=gdn,attn $PY -u qwen38_lowrank_export.py \
  > $L/lowrank_mix25in_mixr.log 2>&1 || { log "FAILED lowrank"; tail -3 $L/lowrank_mix25in_mixr.log; exit 1; }
log "$(tail -1 $L/lowrank_mix25in_mixr.log)"
env EXPORT_DIR=$E/mix25in_mixr_lr64mix TAG=mix25in_mixr_lr64mix $PY -u qwen38_kl.py eval > $L/kl_eval_mix25in_mixr_lr64mix.log 2>&1
log "KL mix25in_mixr_lr64mix: $(grep '^{"tag"' $L/kl_eval_mix25in_mixr_lr64mix.log | cut -c1-330)"
k=$(/usr/bin/python3 -c "import json;print(json.load(open('$TRACE/kl_mix25in_mixr_lr64mix.json'))['mean_kl'])" 2>/dev/null)
base=$(/usr/bin/python3 -c "import json;print(json.load(open('$TRACE/kl_mix25in_aw_cal_lr64mix_stored.json'))['mean_kl'])")
if [[ -n $k ]] && (( k > 0 && k <= 0.97 * base )); then
  log "EXPORT runs/export/mix25in_mixr_lr64mix: KL $k <= 0.97 x $base"
else
  mv $E/mix25in_mixr_lr64mix $E/mix25in_mixr_lr64mix_rejected && log "NO EXPORT: KL ${k:-missing} vs $base (renamed *_rejected)"
fi
log "done"
