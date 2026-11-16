#!/bin/zsh
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; W=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work
for n in "$@"; do
  r=ref_$((n / 1024))k.npz
  echo "== $n $(date +%T) swap $(sysctl -n vm.swapusage | awk '{print $6}')"
  env MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1 I8KV=1 LRM=1 QK8=1 TKV=2048 \
    /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python $W/pipe/v8_a8.py --build /Volumes/SSD4TB/anemll-forge-research/v8_shared2 \
    --gpu-manifest $W/pipe/w_a8/manifest_a8.json --a8-dir $W/pipe/w_a8 --a8-pkg /Volumes/SSD4TB/anemll-forge-research/a8probe/real_n1024.aimodel \
    --tokens $n --ref $W/v8/$r --out $L/a8_len_$n.json --reps 2 --cool-s 20 > $L/a8_len_$n.log 2>&1
  grep -E '"rep"' $L/a8_len_$n.log | cut -c1-420
  sleep 20
done
