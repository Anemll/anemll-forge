#!/bin/zsh
# after the ablation: in-domain calibration (16 WikiText + 16 self-generated chat + 16 pi agentic rows), AW=1
# (imatrix-weighted codebooks), GPTQ + online Hadamard; each export followed by the KL eval.
#   A mix25_aw_cal   deployed MLP plan (2x16, top layers LUT4)   - calibration / imatrix effect at the same speed
#   B mlp4_aw_cal    all MLP LUT4 per-tensor + pcs                - 4-bit quality ceiling
#   C mlp2x64_aw_cal all MLP vector 2x64 + pcs (3 bits/w)          - middle option
while pgrep -f "[r]un_ablation.sh" >/dev/null; do sleep 30; done
cd ~/SourceRelease/GITHUB/ML_playground/ane-vector-lut/scripts
export PYTHONWARNINGS=ignore TRACE=/Volumes/SN8100/vq27b/kl MODEL=/Volumes/SN8100/Qwen3.8-27B
PY=~/venvs/vq27b/bin/python
L=/Volumes/SN8100/vq27b
# closed-form low-rank (LoRA-style) compensation of the deployed export, DeltaNet only and MLP only (rank 64)
E0=$L/runs/export/full_mix25_mixer4_head4
for spec in gdn mlp; do
  EXPORT_DIR=$E0 TAG=abl_${spec}_lr64 PARTS=$spec LR_RANK=64 $PY -u qwen38_kl.py eval > $L/kl_eval_abl_${spec}_lr64.log 2>&1
  echo "$(date +%H:%M:%S) KL abl_${spec}_lr64: $(tail -1 $L/kl_eval_abl_${spec}_lr64.log | cut -c1-200)"
done
echo "$(date +%H:%M:%S) calibration data"
[ -f $L/calib_chat_ids.npy ] || CAL_OUT=$L/calib_chat_ids.npy $PY -u qwen38_calib_gen.py > $L/calib_gen.log 2>&1 || exit 1
tail -1 $L/calib_gen.log
export MIXER="LUT4 per-tensor + pcs" KV_FMT="INT8 per-channel" HEAD="LUT4 per-tensor + pcs" BASELINE=0 AW=1 NCAL=48 \
  CAL_MIX="$L/calib_chat_ids.npy:16,$L/calib_pi_ids.npy:16"
run() {  # tag, then env assignments
  local T=$1; shift
  echo "$(date +%H:%M:%S) GPTQ $T"
  env "$@" TAG=$T $PY -X faulthandler -u qwen38_gptq_27b.py > $L/run_$T.log 2>&1 || { echo "FAILED $T"; tail -3 $L/run_$T.log; return 1; }
  tail -1 $L/run_$T.log
  EXPORT_DIR=$L/runs/export/$T $PY -u qwen38_kl.py eval > $L/kl_eval_$T.log 2>&1
  echo "$(date +%H:%M:%S) KL $T: $(tail -1 $L/kl_eval_$T.log)"
}
run mix25_aw_cal PLAN=$L/plan_optiq_top48.json
run mlp4_aw_cal FORMAT="LUT4 per-tensor + pcs"
run mlp2x64_aw_cal FORMAT="vector 2x64 + pcs"
echo "$(date +%H:%M:%S) done"
