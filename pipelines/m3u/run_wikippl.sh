#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
# M3U helper: WikiText ppl with the lr64 factors applied (M6 deploy check, 23:1x 2026-09-27), plus the no-factor exports
# as a sanity check against the GPTQ script's "quantized ppl" (7.107 / 8.675).
cd "$FORGE_ROOT/scripts" || exit 1
export PYTHONWARNINGS=ignore TRACE=$TRACE MODEL=$MODEL
PY="$FORGE_PYTHON"
L=$FORGE_WORK_DIR
E=$OUT/export
for x in mix25in_aw_cal_lr64mix mix25in_mixr_lr64mix mix25in_aw_cal mix25in_mixr; do
  env EXPORT_DIR=$E/$x $PY -u qwen38_wikippl.py > $L/wikippl_$x.log 2>&1
  echo "$(date +%H:%M:%S) WIKI $x: $(grep '^{"tag"' $L/wikippl_$x.log | cut -c1-200 || tail -2 $L/wikippl_$x.log)"
done
echo "$(date +%H:%M:%S) done"
