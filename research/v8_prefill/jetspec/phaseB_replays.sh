#!/bin/zsh
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov
J=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/jetspec
free_gb() { vm_stat | awk '/Pages free/ {gsub("\\.","",$NF); printf "%d", $NF*16384/1e9}'; }
for ds in humaneval mbpp mt-bench; do
  for i in $(seq 1 60); do [ $(free_gb) -ge 18 ] && break; sleep 10; done
  echo "$(date '+%H:%M') replay $ds (free $(free_gb) GB)" >> $R/ops.log
  rm -f $R/jetspec/replay_$ds.json
  WDMB=6144 CAP=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/capture_${ds}8_20nov RES=$R/jetspec DATASET=$ds TAG=$ds zsh $J/phaseB.sh replay
done
echo ALLDONE
