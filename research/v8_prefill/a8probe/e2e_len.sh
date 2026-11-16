#!/bin/zsh
# real end to end (prefill + overlapped handover + teacher-forced ANE decode check), baseline V8 fuse engine vs A8
# (--att all --out-layers gdn), one length per call:  e2e_len.sh TOKENS REF(file in v8/ | run-a) REPS
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; W=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work; U=$W/pipe/unit
N=$1; R=$2; RP=$3
export BUILD=/Volumes/SSD4TB/anemll-forge-research/v8_shared2 MAN=$W/pipe/w_shared2r/manifest_best.json
ENVS="FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1 I8KV=1 LRM=1"
[ $N -ge 4096 ] && ENVS="$ENVS QK8=1 TKV=2048"
APJ=""; [ $N -ge 32768 ] && APJ="--attproj"
REFA8=$W/v8/$R; [ "$R" = run-a ] && REFA8=run-a
echo "== $N baseline $(date +%T)"
env ${=ENVS} $U/run_wd.sh $N $R e2ex${N}_base --reps $RP --shared --streams 8 --dylib libgpu_v8_fuse.dylib --handover --ho-overlap --cool-s 30 ${=APJ}
grep -E '^\{"ane_only_same|^\{"e2e' $L/e2ex${N}_base.log | cut -c1-330
sleep 30
echo "== $N A8 $(date +%T)"
env MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 MODEL=/Volumes/SSD4TB/Qwen3.8-27B ${=ENVS} A8EXP=t /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python $W/pipe/v8_a8.py \
  --build $BUILD --gpu-manifest $W/pipe/w_a8/manifest_a8.json --a8-dir $W/pipe/w_a8 --a8-pkg /Volumes/SSD4TB/anemll-forge-research/a8probe/real_n1024.aimodel \
  --tokens $N --ref $REFA8 --out $L/e2ex${N}_a8.json --reps $RP --cool-s 30 --att all --out-layers gdn --handover > $L/e2ex${N}_a8.log 2>&1 &
G=$!; /Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $G 1024 $L/watchdog_e2ex${N}_a8.log >/dev/null 2>&1 &
wait $G; echo "A8 rc=$?"
grep -E '^\{"ane_only_same|^\{"e2e|Error|Traceback|stall' $L/e2ex${N}_a8.log | cut -c1-330
sleep 30
