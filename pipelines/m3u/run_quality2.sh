#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
# size-neutral quality plan (user constraint: quantized size = MLP speed). After the ablation:
#   low-rank (rank 64) compensation evals of the deployed export (DeltaNet only / MLP only)
#   in-domain calibration data (16 WikiText + 16 self-generated chat + 16 pi agentic rows), AW=1 codebooks
#   A  mix25_aw_cal : deployed plan, new calibration          (same size: calibration / imatrix effect)
#   M2 mlp2_aw_cal  : every MLP at vector 2x16, new calibration (smallest; clean 2-bit weights for the band sweep)
#   band sweep      : KL with only one 8-layer MLP band at 2-bit (rest bf16) -> in-domain re-allocation of the 4-bit budget
while pgrep -f "[r]un_ablation.sh" >/dev/null; do sleep 30; done
cd "$FORGE_ROOT/scripts" || exit 1
export PYTHONWARNINGS=ignore TRACE=$TRACE MODEL=$MODEL
PY="$FORGE_PYTHON"
L=$FORGE_WORK_DIR
E0=$OUT/export/full_mix25_mixer4_head4
kl() {  # tag, env...
  local T=$1; shift
  env "$@" TAG=$T $PY -u qwen38_kl.py eval > $L/kl_eval_$T.log 2>&1
  echo "$(date +%H:%M:%S) KL $T: $(tail -1 $L/kl_eval_$T.log | cut -c1-230)"
}
for part in gdn mlp; do kl abl_${part}_lr64 EXPORT_DIR=$E0 PARTS=$part LR_RANK=64; done
echo "$(date +%H:%M:%S) calibration data"
[ -f $L/calib_chat_ids.npy ] || CAL_OUT=$L/calib_chat_ids.npy $PY -u qwen38_calib_gen.py > $L/calib_gen.log 2>&1 || { echo "calib FAILED"; tail -5 $L/calib_gen.log; exit 1; }
tail -1 $L/calib_gen.log
export MIXER="LUT4 per-tensor + pcs" KV_FMT="INT8 per-channel" HEAD="LUT4 per-tensor + pcs" BASELINE=0 AW=1 NCAL=48 \
  CAL_MIX="$L/calib_chat_ids.npy:16,$L/calib_pi_ids.npy:16"
gptq() {  # tag, env...
  local T=$1; shift
  echo "$(date +%H:%M:%S) GPTQ $T"
  env "$@" TAG=$T $PY -X faulthandler -u qwen38_gptq_27b.py > $L/run_$T.log 2>&1 || { echo "FAILED $T"; tail -5 $L/run_$T.log; return 1; }
  tail -1 $L/run_$T.log
  kl $T EXPORT_DIR=$OUT/export/$T
}
gptq mix25_aw_cal PLAN=$L/plan_optiq_top48.json
gptq mlp2_aw_cal FORMAT="vector 2x16 + pcs"
for b in 0-7 8-15 16-23 24-31 32-39 40-47 48-55 56-63; do
  kl band2_$b EXPORT_DIR=$OUT/export/mlp2_aw_cal PARTS=mlp QLAYERS=$b
done
echo "$(date +%H:%M:%S) done"
