#!/bin/zsh
# M3U helper: when the last band KL (kl_mband2lr_56-63.json) exists, stop run_mixr.sh before its optional additivity
# eval (mixers2lr_only) and start run_mixr_final.sh.
L=/Volumes/SN8100/vq27b
until [[ -f $L/kl/kl_mband2lr_56-63.json ]]; do sleep 10; done
pkill -f run_mixr.sh; sleep 2; pkill -f "TAG=mixers2lr_only" ; pkill -f "qwen38_kl.py eval"; sleep 5
echo "$(date +%H:%M:%S) trigger: bands done; run_mixr.sh stopped before mixers2lr_only; starting run_mixr_final.sh" >> $L/run_mixr.out
cd $L && nohup ./run_mixr_final.sh > run_mixr_final.out 2>&1 < /dev/null &
