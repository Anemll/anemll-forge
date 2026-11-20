#!/bin/zsh
# Queue 8 (19 Nov 2026): the staged Q.K epilogue (QKE=tg, attn_qk8isk_tg) end to end against the served F8=x path, the
# same binary (libgpu_v8_f8q.dylib; QKE unset = attn_qk8isk), interleaved fresh processes x a / tg a / x b / tg b, each
# 95K, 64K, 32K in that order (v8_trace's 45 s idle before each run, 30 s between processes). Banner from the loaded
# dylib (F8=x and the Q.K kernel; abort on a mismatch). Then the last-token logits compared (bit-exact / KL) and forge
# restarted on 8765 with the standard launcher (FP8 x serving, keep-alive).
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; R=/Volumes/SSD4TB/anemll-forge-research/contention/qkf8/e2e; mkdir -p $R
echo "$(date +%T) stopping forge on 8765"; PORT=8765 $P/srv_stop.sh; sleep 30
restart() { echo "$(date +%T) restarting forge on 8765 (FP8 serving defaults)"; cd $P; PORT=8765 nohup $P/v8_server_dec8.sh >/dev/null 2>&1 &
  for i in $(seq 1 90); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
  echo "$(date +%T) forge ready"; grep -E "v8: variant" /Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log | cut -c1-160; }
for T in x_a tg_a x_b tg_b; do
  mkdir -p $R/tr_$T; echo "$(date +%T) trace $T (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"
  if [[ $T == tg* ]]; then env V8_DYLIB=libgpu_v8_f8q.dylib F8=x QKE=tg EXPECT_F8=x EXPECT_QK=attn_qk8isk_tg $D/v8_trace.sh $R/tr_$T 95232,65536,32768
  else env V8_DYLIB=libgpu_v8_f8q.dylib F8=x QKE= EXPECT_F8=x EXPECT_QK=attn_qk8isk $D/v8_trace.sh $R/tr_$T 95232,65536,32768; fi
  grep -E "^BANNER|ABORT" $R/tr_$T.log | cut -c1-300
  grep -q "^ABORT" $R/tr_$T.log && { echo "variant mismatch: stopping"; restart; exit 1; }
  $PY $D/trace_analyze.py $R/tr_$T > /dev/null 2>$R/tr_$T/analyze.err
  grep "^run" $R/tr_$T.log
  sleep 30
done
$PY $D/logits_cmp.py $R/tr_x_a $R/tr_x_b $R/tr_tg_a $R/tr_tg_b
restart
echo "$(date +%T) all done"
