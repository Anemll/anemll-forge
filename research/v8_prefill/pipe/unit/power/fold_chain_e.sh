#!/bin/zsh
# FP8 P.V at 95K, repeat (18 Nov 2026, evening): interleaved V8 traces plain / F8=x / plain / F8=x at 95,232 tokens, one
# fresh process per run, the same binary (libgpu_v8_f8.dylib with v8_variant) and setup as fold_chain_d.sh; each run's
# banner from the loaded dylib (EXPECT_F8), abort on a mismatch. Forge stopped first; NOT restarted (decided after).
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; R=/Volumes/SSD4TB/anemll-forge-research/contention/fold/f8/rep95
mkdir -p $R
echo "$(date +%T) stopping forge on 8765"; PORT=8765 $P/srv_stop.sh
sleep 30
for T in plain95_a f8x95_a plain95_b f8x95_b; do
  mkdir -p $R/tr_$T; echo "$(date +%T) trace $T (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"
  if [[ $T == f8x* ]]; then env V8_DYLIB=libgpu_v8_f8.dylib F8=x EXPECT_F8=x $D/v8_trace.sh $R/tr_$T 95232
  else env -u F8 V8_DYLIB=libgpu_v8_f8.dylib EXPECT_F8=off $D/v8_trace.sh $R/tr_$T 95232; fi
  grep -E "^BANNER|ABORT" $R/tr_$T.log | cut -c1-400
  grep -q "^ABORT" $R/tr_$T.log && { echo "variant mismatch: stopping"; exit 1; }
  $PY $D/trace_analyze.py $R/tr_$T > /dev/null 2>$R/tr_$T/analyze.err
  grep "^run" $R/tr_$T.log | tail -1
  sleep 20
done
$PY $D/logits_cmp.py $R/tr_plain95_a $R/tr_plain95_b $R/tr_f8x95_a $R/tr_f8x95_b
echo "$(date +%T) all done (forge left down)"
