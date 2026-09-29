#!/bin/zsh
# after the ablation: in-domain calibration data, then MLP LUT4 (all layers) + LUT4 mixers + INT8 k/v + LUT4 head,
# GPTQ with half WikiText / half chat calibration, then the KL eval
while pgrep -f "[r]un_ablation.sh" >/dev/null; do sleep 30; done
cd ~/SourceRelease/GITHUB/ML_playground/ane-vector-lut/scripts
export PYTHONWARNINGS=ignore TRACE=/Volumes/SN8100/vq27b/kl MODEL=/Volumes/SN8100/Qwen3.8-27B
PY=~/venvs/vq27b/bin/python
L=/Volumes/SN8100/vq27b
echo "$(date +%H:%M:%S) calibration data"
[ -f $L/calib_chat_ids.npy ] || CAL_OUT=$L/calib_chat_ids.npy $PY -u qwen38_calib_gen.py > $L/calib_gen.log 2>&1 || exit 1
tail -1 $L/calib_gen.log
T=full_mlp4_mixer4_head4_chat
echo "$(date +%H:%M:%S) GPTQ $T"
FORMAT="LUT4 per-tensor + pcs" MIXER="LUT4 per-tensor + pcs" KV_FMT="INT8 per-channel" HEAD="LUT4 per-tensor + pcs" \
  BASELINE=0 CAL_MIX=$L/calib_chat_ids.npy TAG=$T $PY -X faulthandler -u qwen38_gptq_27b.py > $L/run_$T.log 2>&1 || exit 1
tail -1 $L/run_$T.log
echo "$(date +%H:%M:%S) KL eval"
EXPORT_DIR=$L/runs/export/$T $PY -u qwen38_kl.py eval > $L/kl_eval_$T.log 2>&1
tail -1 $L/kl_eval_$T.log
echo "$(date +%H:%M:%S) done"
