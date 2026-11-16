#!/bin/zsh
# V8 baseline with KV8 (GPU attention on the ANE's INT8 segments, fixed softmax8, INT8 scores attn_qk8is) + attproj,
# real end to end, 64K and 97K; engine libgpu_v8_a8.dylib (superset of the fuse engine)
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; U=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit
export BUILD=/Volumes/SSD4TB/anemll-forge-research/v8_shared2 MAN=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared2r/manifest_best.json
for spec in 65536:ref_64k.npz 97280:run-a; do
  N=${spec%%:*}; R=${spec#*:}
  env FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1 I8KV=1 LRM=1 QK8=1 TKV=2048 $U/run_wd.sh $N $R v8kv8_$N --reps 1 --shared --streams 8 \
    --dylib libgpu_v8_a8.dylib --handover --ho-overlap --kv8 --attproj --cool-s 30
  grep -E '^\{"ane_only_same|^\{"e2e|Error|Traceback' $L/v8kv8_$N.log | cut -c1-330
  sleep 30
done
