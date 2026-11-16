#!/bin/zsh
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; W=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work
for spec in 65536:$W/v8/ref_64k.npz 97280:run-a; do
  N=${spec%%:*}; R=${spec#*:}
  echo "== A8 KV8 $N $(date +%T) swap $(sysctl -n vm.swapusage | awk '{print $6}')"
  env MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1 I8KV=1 LRM=1 QK8=1 TKV=2048 A8EXP=t MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
    /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python $W/pipe/v8_a8.py --build /Volumes/SSD4TB/anemll-forge-research/v8_shared2 \
    --gpu-manifest $W/pipe/w_a8/manifest_a8.json --a8-dir $W/pipe/w_a8 --a8-pkg /Volumes/SSD4TB/anemll-forge-research/a8probe/real_n1024.aimodel \
    --tokens $N --ref $R --out $L/a8kv8_$N.json --reps 1 --cool-s 30 --att all --out-layers gdn --handover --kv8 > $L/a8kv8_$N.log 2>&1 &
  G=$!; /Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $G 1024 $L/watchdog_a8kv8_$N.log >/dev/null 2>&1 &
  wait $G; echo "rc=$? $(tail -1 $L/watchdog_a8kv8_$N.log)"
  grep -E '^\{"ane_only|^\{"e2e|Error|Traceback|stall' $L/a8kv8_$N.log | cut -c1-330
  sleep 30
done
