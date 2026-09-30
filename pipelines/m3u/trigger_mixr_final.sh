#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
# M3U helper: when the last band KL (kl_mband2lr_56-63.json) exists, stop run_mixr.sh before its optional additivity
# eval (mixers2lr_only) and start run_mixr_final.sh.
L=$FORGE_WORK_DIR
until [[ -f $TRACE/kl_mband2lr_56-63.json ]]; do sleep 10; done
pkill -f run_mixr.sh; sleep 2; pkill -f "TAG=mixers2lr_only" ; pkill -f "qwen38_kl.py eval"; sleep 5
echo "$(date +%H:%M:%S) trigger: bands done; run_mixr.sh stopped before mixers2lr_only; starting run_mixr_final.sh" >> $L/run_mixr.out
nohup zsh "$FORGE_PIPELINE_DIR/run_mixr_final.sh" > "$L/run_mixr_final.out" 2>&1 < /dev/null &
