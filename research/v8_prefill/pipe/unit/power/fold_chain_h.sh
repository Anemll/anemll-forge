#!/bin/zsh
# F8=x + folded softmax, stage 2 (18 Nov 2026, evening): V8 traces at 65,536 tokens, x / x+fold (F8F=w128, every call) /
# plain / x / x+fold, one fresh process each, all through libgpu_v8_f8f.dylib (F8F unset = x exactly), banner from
# the loaded dylib (EXPECT_F8 = off | x | x+w128), abort on a mismatch. Forge must be down; restarted after (FP8 serving).
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; R=/Volumes/SSD4TB/anemll-forge-research/contention/fold/f8f/e2e64
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "forge is up on 8765: stop it first"; exit 1; }
mkdir -p $R
for T in x_a xf_a plain_a x_b xf_b; do
  mkdir -p $R/tr_$T; echo "$(date +%T) trace $T (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"
  case $T in
    x_*)     env -u F8F V8_DYLIB=libgpu_v8_f8f.dylib F8=x EXPECT_F8=x $D/v8_trace.sh $R/tr_$T 65536 ;;
    xf_*)    env V8_DYLIB=libgpu_v8_f8f.dylib F8=x F8F=w128 EXPECT_F8=x+w128 $D/v8_trace.sh $R/tr_$T 65536 ;;
    plain_*) env -u F8 -u F8F V8_DYLIB=libgpu_v8_f8f.dylib EXPECT_F8=off $D/v8_trace.sh $R/tr_$T 65536 ;;
  esac
  grep -E "^BANNER|ABORT" $R/tr_$T.log | cut -c1-300
  grep -q "^ABORT" $R/tr_$T.log && { echo "variant mismatch: stopping"; exit 1; }
  $PY $D/trace_analyze.py $R/tr_$T > /dev/null 2>$R/tr_$T/analyze.err
  grep "^run" $R/tr_$T.log
  sleep 30
done
$PY $D/logits_cmp.py $R/tr_plain_a $R/tr_x_a $R/tr_x_b $R/tr_xf_a $R/tr_xf_b
echo "$(date +%T) restarting forge on 8765 (FP8 serving defaults)"; cd $P; PORT=8765 nohup $P/v8_server_dec8.sh >/dev/null 2>&1 &
for i in $(seq 1 60); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
echo "$(date +%T) forge ready"; grep -E "v8: variant|dec8: decode GPU" /Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log | cut -c1-200
echo "$(date +%T) all done"
