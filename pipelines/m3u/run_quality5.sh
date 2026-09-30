#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
# in-domain MLP allocation (exports on SN8100 again (SD1T is a slow SD card))
#   M2 mlp2_aw_cal (all MLP 2-bit, new calibration) -> 8-band KL sweep -> in-domain plan at the same budget
#   -> mix25in_aw_cal (the new plan) + KL, + KL with rank-64 factors on DeltaNet + attention
cd "$FORGE_ROOT/scripts" || exit 1
export PYTHONWARNINGS=ignore TRACE=$TRACE MODEL=$MODEL OUT=$OUT
PY="$FORGE_PYTHON"
L=$FORGE_WORK_DIR
R=$OUT
mkdir -p $R
kl() { local T=$1; shift; env "$@" TAG=$T $PY -u qwen38_kl.py eval > $L/kl_eval_$T.log 2>&1; echo "$(date +%H:%M:%S) KL $T: $(tail -1 $L/kl_eval_$T.log | cut -c1-230)"; }
export MIXER="LUT4 per-tensor + pcs" KV_FMT="INT8 per-channel" HEAD="LUT4 per-tensor + pcs" BASELINE=0 AW=1 NCAL=48 \
  CAL_MIX="$L/calib_chat_ids.npy:16,$L/calib_pi_ids.npy:16"
gptq() { local T=$1; shift; echo "$(date +%H:%M:%S) GPTQ $T"
  env "$@" TAG=$T $PY -X faulthandler -u qwen38_gptq_27b.py > $L/run_$T.log 2>&1 || { echo "FAILED $T"; tail -5 $L/run_$T.log; return 1; }
  tail -1 $L/run_$T.log; }
gptq mlp2_aw_cal FORMAT="vector 2x16 + pcs" || exit 1
for b in 0-7 8-15 16-23 24-31 32-39 40-47 48-55 56-63; do kl band2_$b EXPORT_DIR=$R/export/mlp2_aw_cal PARTS=mlp QLAYERS=$b; done
$PY qwen38_plan_indomain.py --kl-dir $TRACE --sweep "$OUT/sweep_mlp_vector_2x16_+_pcs_online.json" --plan $L/plan_optiq_top48.json --out $L/plan_indomain.json 2>&1 | tail -5
gptq mix25in_aw_cal PLAN=$L/plan_indomain.json && kl mix25in_aw_cal EXPORT_DIR=$R/export/mix25in_aw_cal && \
  kl mix25in_aw_cal_lr64mix EXPORT_DIR=$R/export/mix25in_aw_cal LR_RANK=64 LR_PARTS=gdn,attn
echo "$(date +%H:%M:%S) done"
