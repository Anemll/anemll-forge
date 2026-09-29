#!/bin/zsh
# size-neutral quality plan, v3 (after abl_gdn_lr64 = KL 0.024: low-rank correction of the DeltaNet error works):
#   wait for the running abl_mlp_lr64 eval; in-domain calibration data;
#   A  mix25_aw_cal (deployed plan, new calibration, AW) + KL, + KL with rank-64 correction on DeltaNet + attention
#   BR full_mix25_br (block reconstruction of the deployed export, running separately) -> KL when it is done
#   M2 mlp2_aw_cal (all MLP 2-bit, new calibration) + KL, then the 8-band in-domain MLP sweep
while pgrep -f "[q]wen38_kl.py" >/dev/null; do sleep 20; done
cd ~/SourceRelease/GITHUB/ML_playground/ane-vector-lut/scripts
export PYTHONWARNINGS=ignore TRACE=/Volumes/SN8100/vq27b/kl MODEL=/Volumes/SN8100/Qwen3.8-27B
PY=~/venvs/vq27b/bin/python
L=/Volumes/SN8100/vq27b
echo "$(date +%H:%M:%S) KL abl_mlp_lr64: $(tail -1 $L/kl_eval_abl_mlp_lr64.log | cut -c1-230)"
kl() {  # tag, env...
  local T=$1; shift
  env "$@" TAG=$T $PY -u qwen38_kl.py eval > $L/kl_eval_$T.log 2>&1
  echo "$(date +%H:%M:%S) KL $T: $(tail -1 $L/kl_eval_$T.log | cut -c1-230)"
}
kl_ready() {  # KL of a block-reconstruction export once it is written (and not yet evaluated)
  [ -f $L/runs/export/$1/blockrecon.json ] && [ ! -f $L/kl/kl_$1.json ] && kl $1 EXPORT_DIR=$L/runs/export/$1
}
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
  kl $T EXPORT_DIR=$L/runs/export/$T
}
gptq mix25_aw_cal PLAN=$L/plan_optiq_top48.json
kl mix25_aw_cal_lr64mix EXPORT_DIR=$L/runs/export/mix25_aw_cal LR_RANK=64 LR_PARTS=gdn,attn
kl_ready full_mix25_br; kl_ready full_mix25_br_lr64
kl_ready full_mix25_br; kl_ready full_mix25_br_lr64
gptq mlp2_aw_cal FORMAT="vector 2x16 + pcs"
for b in 0-7 8-15 16-23 24-31 32-39 40-47 48-55 56-63; do
  kl band2_$b EXPORT_DIR=$L/runs/export/mlp2_aw_cal PARTS=mlp QLAYERS=$b
done
while pgrep -f "[q]wen38_blockrecon.py" >/dev/null; do sleep 60; done; kl_ready full_mix25_br; kl_ready full_mix25_br_lr64
echo "$(date +%H:%M:%S) done"
