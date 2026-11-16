#!/bin/zsh
# V8 with K8V16 (keys from the ANE's INT8 cache, the GPU's own fp16 V copy for PV) + attproj, real end to end, 64K
# (ref_64k.npz): the same run as v8kv8_long.sh (KV8: kl_mean 0.0592, 401.7 tok/s); engine libgpu_v8_srv2.dylib
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; U=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit
export BUILD=/Volumes/SSD4TB/anemll-forge-research/v8_shared2 MAN=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared2r/manifest_best.json
for spec in ${@:-65536:ref_64k.npz}; do
  N=${spec%%:*}; R=${spec#*:}
  env FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1 I8KV=1 LRM=1 QK8=1 TKV=2048 K8V16=1 $U/run_wd.sh $N $R k8v16_$N --reps 1 --shared --streams 8 \
    --dylib libgpu_v8_srv2.dylib --handover --ho-overlap --kv8 --attproj --cool-s 30
  grep -E '^\{"ane_only_same|^\{"e2e|"kl_mean|Error|Traceback' $L/k8v16_$N.log | cut -c1-330
  sleep 30
done
