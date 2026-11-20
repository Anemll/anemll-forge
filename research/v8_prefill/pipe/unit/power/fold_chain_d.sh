#!/bin/zsh
# FP8 P.V at 95K (18 Nov 2026): interleaved V8 traces plain / F8=x / plain / F8=x at 95,232 tokens, one fresh process per
# run (memory), the same binary (libgpu_v8_f8.dylib), last-token logits saved; forge stopped first and restarted after.
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; R=/Volumes/SSD4TB/anemll-forge-research/contention/fold/f8
echo "$(date +%T) stopping forge on 8765"; PORT=8765 $P/srv_stop.sh
sleep 30
for T in plain95_a f8x95_a plain95_b f8x95_b; do
  mkdir -p $R/tr_$T; echo "$(date +%T) trace $T (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"
  if [[ $T == f8x* ]]; then V8_DYLIB=libgpu_v8_f8.dylib F8=x $D/v8_trace.sh $R/tr_$T 95232
  else V8_DYLIB=libgpu_v8_f8.dylib $D/v8_trace.sh $R/tr_$T 95232; fi
  $PY $D/trace_analyze.py $R/tr_$T > /dev/null 2>$R/tr_$T/analyze.err
  grep "^run" $R/tr_$T.log | tail -1
  sleep 20
done
echo "$(date +%T) restarting forge on 8765"; cd $P; PORT=8765 nohup $P/v8_server_dec8.sh >/dev/null 2>&1 &
for i in $(seq 1 60); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
echo "$(date +%T) forge ready"
$PY $D/logits_cmp.py $R/tr_plain95_a $R/tr_plain95_b $R/tr_f8x95_a $R/tr_f8x95_b
echo "$(date +%T) all done"
