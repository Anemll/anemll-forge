#!/bin/zsh
# M3U helper: WikiText ppl with the lr64 factors applied (M6 deploy check, 23:1x 2026-09-27), plus the no-factor exports
# as a sanity check against the GPTQ script's "quantized ppl" (7.107 / 8.675).
cd ~/SourceRelease/GITHUB/ML_playground/ane-vector-lut/scripts
export PYTHONWARNINGS=ignore TRACE=/Volumes/SN8100/vq27b/kl MODEL=/Volumes/SN8100/Qwen3.8-27B
PY=~/venvs/vq27b/bin/python
L=/Volumes/SN8100/vq27b
E=$L/runs/export
for x in mix25in_aw_cal_lr64mix mix25in_mixr_lr64mix mix25in_aw_cal mix25in_mixr; do
  env EXPORT_DIR=$E/$x $PY -u qwen38_wikippl.py > $L/wikippl_$x.log 2>&1
  echo "$(date +%H:%M:%S) WIKI $x: $(grep '^{"tag"' $L/wikippl_$x.log | cut -c1-200 || tail -2 $L/wikippl_$x.log)"
done
echo "$(date +%H:%M:%S) done"
