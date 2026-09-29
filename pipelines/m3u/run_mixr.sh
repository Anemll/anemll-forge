#!/bin/zsh
# M3U helper (M6 request 18:5x 2026-09-27): mixer -> MLP byte trade, measurement phase.
#   1. GPTQ mixer2_aw_cal: every mixer matrix vector 2x16 + pcs (k/v INT8), MLP bf16 (plan_mlpbf16.json), head bf16,
#      same CAL_MIX / AW=1 -> runs/export/mixer2_aw_cal (its MLP files are bf16 placeholders, ~34 GB, deleted after)
#   2. KL, all mixers LUT4 + lr64 (mix25in_aw_cal), rest bf16            -> tag mixers4lr_only
#   3. 8 band KLs, mixers of one band 2-bit + lr64 re-fitted, rest bf16  -> tags mband2lr_<lo>-<hi>
#   4. KL, all mixers 2-bit + lr64, rest bf16 (additivity check)        -> tag mixers2lr_only
# The plan (which mixer layers go 2-bit, which MLP matrices go LUT4, at equal bytes) comes after, from these numbers.
cd ~/SourceRelease/GITHUB/ML_playground/ane-vector-lut/scripts
export PYTHONWARNINGS=ignore TRACE=/Volumes/SN8100/vq27b/kl MODEL=/Volumes/SN8100/Qwen3.8-27B OUT=/Volumes/SN8100/vq27b/runs
PY=~/venvs/vq27b/bin/python
L=/Volumes/SN8100/vq27b
E=$L/runs/export
log() { echo "$(date +%H:%M:%S) $*"; }
busy() { pgrep -f "qwen38_kl.py|qwen38_gptq_27b.py|qwen38_lowrank" > /dev/null; }
kl() { local T=$1; shift; while busy; do sleep 30; done
  env "$@" TAG=$T $PY -u qwen38_kl.py eval > $L/kl_eval_$T.log 2>&1
  log "KL $T: $(grep '^{"tag"' $L/kl_eval_$T.log | cut -c1-300 || tail -2 $L/kl_eval_$T.log)"; }

while busy; do sleep 30; done
log "GPTQ mixer2_aw_cal"
env MIXER="vector 2x16 + pcs" KV_FMT="INT8 per-channel" HEAD="" BASELINE=0 AW=1 NCAL=48 \
  CAL_MIX="$L/calib_chat_ids.npy:16,$L/calib_pi_ids.npy:16" PLAN=$L/plan_mlpbf16.json TAG=mixer2_aw_cal \
  $PY -X faulthandler -u qwen38_gptq_27b.py > $L/run_mixer2_aw_cal.log 2>&1 || { log "FAILED GPTQ"; tail -5 $L/run_mixer2_aw_cal.log; exit 1; }
log "$(tail -1 $L/run_mixer2_aw_cal.log)"
n=$(ls $E/mixer2_aw_cal | grep -c '_mixer.safetensors')
[[ $n == 64 ]] || { log "STOP: mixer2_aw_cal has $n mixer files, expected 64"; exit 1; }

kl mixers4lr_only EXPORT_DIR=$E/mix25in_aw_cal PARTS=gdn,attn LR_RANK=64 LR_PARTS=gdn,attn
for b in 0-7 8-15 16-23 24-31 32-39 40-47 48-55 56-63; do
  kl mband2lr_$b EXPORT_DIR=$E/mixer2_aw_cal PARTS=gdn,attn QLAYERS=$b LR_RANK=64 LR_PARTS=gdn,attn
done
kl mixers2lr_only EXPORT_DIR=$E/mixer2_aw_cal PARTS=gdn,attn LR_RANK=64 LR_PARTS=gdn,attn
log "done"
