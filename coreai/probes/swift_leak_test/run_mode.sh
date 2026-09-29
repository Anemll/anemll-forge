#!/bin/zsh
# run_mode.sh <mode> <calls> [entry] : run one mode in its own process, log to log_<mode>_<calls>.txt,
# then sample system wired memory right after the process exits (the drop = what the test process held).
cd "${0:A:h}"
MODEL=${MODEL:-/Volumes/SSD4TB/vq27b-ane/builds/coreai_ane6/mix25_aw_cal_lr64mix/chunk_L00-03.aimodel}
ENTRY=${3:-v8_2k}
LOG=log_${1}_${2}_${ENTRY}_${PRIO:-userInitiated}${INPUT_LAYOUT:+_$INPUT_LAYOUT}.txt
wired() { vm_stat | awk '/page size of/{ps=$8} /wired down/{gsub("\\.","",$4); printf "%.2f", $4*ps/2^30}'; }
./coreai_leak_test $1 $2 $ENTRY $MODEL > $LOG 2>&1
echo "exit code $?" >> $LOG
echo "wired right after exit: $(wired) GB" >> $LOG
sleep 3
echo "wired 3 s after exit: $(wired) GB" >> $LOG
cat $LOG
