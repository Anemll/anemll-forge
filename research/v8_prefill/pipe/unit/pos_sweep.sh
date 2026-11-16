#!/bin/zsh
# Position sweep (timing only, driver --start): a 4K-token slice at each position P, as if [0, P) were in the KV cache.
# Per P: 4 streams with attproj off / on; 8 streams with {no ANE mixers, M16} x {attproj off, on}. Shared build v8_shared2.
# usage: pos_sweep.sh [positions...]   then: pos_summ.py
S=${0:a:h}; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs
export BUILD=/Volumes/SSD4TB/anemll-forge-research/v8_shared2 MAN=${MAN:-/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared2r/manifest.json}
M16="1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61"
PS=(${=${*:-28672 61440 0 94208 8192 45056 16384 77824}})
for P in $PS; do
  $S/run_wd.sh 4096 ref_2k.npz pos4_$P --start $P --reps 4 --shared --attproj --attproj-alt --dylib libgpu_v8_attcore.dylib --cool-s 30
  sleep 20
  $S/run_wd.sh 4096 ref_2k.npz pos8_$P --start $P --reps 4 --shared --streams 8 --attproj --attproj-alt --ane-mix-alt ";;$M16;$M16" --dylib libgpu_v8_attcore.dylib --cool-s 30
  sleep 20
done
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python $S/pos_summ.py
