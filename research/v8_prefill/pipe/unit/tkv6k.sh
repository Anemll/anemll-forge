#!/bin/zsh
# 6K Pi prompt: GPU attention key tile TKV 2048 vs 4096 (separate processes, the dylib reads it at init), 4 streams,
# attproj never / from 4K alternating.
S=${0:a:h}; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs
export BUILD=/Volumes/SSD4TB/anemll-forge-research/v8_shared2 MAN=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared2r/manifest.json
while pgrep -f pi6k.sh >/dev/null; do sleep 10; done
for T in 2048 4096; do
  sleep 20
  TKV=$T $S/run_wd.sh 6144 ref_6k.npz tkv${T}_6k --reps 4 --shared --streams 4 --attproj --attproj-from-alt 1000000,4096 --dylib libgpu_v8_attcore.dylib --cool-s 30
done
