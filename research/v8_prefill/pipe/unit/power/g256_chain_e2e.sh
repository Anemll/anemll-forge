#!/bin/zsh
# Queue 9b (19 Nov 2026): 256-row DeltaNet calls with the hidden-decode LUT GEMM (v8_engine_trace_g256.py +
# libgpu_v8_g256.dylib, V8G_I8TV=_k256ws in that instance only; attention unchanged)
# against the served config (v8_engine_trace.py, libgpu_v8_f8q.dylib, F8=x, QKE=tg), interleaved fresh processes:
# a 4K smoke pair first (abort unless the logits agree), then s a / g a / s b / g b at 95K and 32K. Banners from the
# loaded dylibs (and the DeltaNet instance); last-token logits compared; forge restarted with the serving launcher.
# The 256-row arm runs with V8_ANE_LOCK=1: two simultaneous ANE calls of the same layer's ffn2 function (both units of
# a quad are released together) corrupt each other's results (debug 19 Nov, g256_debug3/4); the lock serializes them.
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; R=/Volumes/SSD4TB/anemll-forge-research/contention/gdn/g256_e2e; mkdir -p $R
restart() { echo "$(date +%T) restarting forge on 8765 (v8_server_dec8_qk.sh)"; cd $P; PORT=8765 nohup $P/v8_server_dec8_qk.sh >/dev/null 2>&1 &
  for i in $(seq 1 90); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
  echo "$(date +%T) forge ready"; grep -E "v8: variant" /Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log | cut -c1-200; }
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "$(date +%T) stopping forge on 8765"; PORT=8765 $P/srv_stop.sh; sleep 30; }
trace_run() {   # trace_run NAME LENGTHS
  local T=$1 N=$2; mkdir -p $R/tr_$T; echo "$(date +%T) trace $T $N (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"
  if [[ $T == g* ]]; then env V8_DYLIB=libgpu_v8_f8q.dylib F8=x QKE=tg V8_ENGINE_MOD=v8_engine_trace_g256 V8G_I8TV=_k256ws V8_ANE_LOCK=1 EXPECT_G256=1 EXPECT_G256_LUT=lut_gemm32_i8t_k256ws EXPECT_F8=x EXPECT_QK=attn_qk8isk_tg $D/v8_trace.sh $R/tr_$T $N
  else env V8_DYLIB=libgpu_v8_f8q.dylib F8=x QKE=tg EXPECT_F8=x EXPECT_QK=attn_qk8isk_tg $D/v8_trace.sh $R/tr_$T $N; fi
  grep -E "^BANNER|ABORT|Traceback|Error" $R/tr_$T.log | cut -c1-300
  grep -q -E "^ABORT|Traceback" $R/tr_$T.log && { echo "$T failed: stopping"; restart; exit 1; }
  $PY $D/trace_analyze.py $R/tr_$T > /dev/null 2>$R/tr_$T/analyze.err
  grep "^run" $R/tr_$T.log
  sleep 30
}
trace_run s_smoke 4096; trace_run g_smoke 4096
$PY $D/logits_cmp.py $R/tr_s_smoke $R/tr_g_smoke | tee $R/smoke_cmp.jsonl
$PY -c "
import json, sys
d = [json.loads(l) for l in open('$R/smoke_cmp.jsonl')]
ok = d and all(x['bitexact'] for x in d)
print('smoke', 'OK' if ok else 'FAILED', d); sys.exit(0 if ok else 1)" || { echo "smoke failed: stopping"; restart; exit 1; }
for T in s_a g_a s_b g_b; do trace_run $T 95232,32768; done
$PY $D/logits_cmp.py $R/tr_s_a $R/tr_s_b $R/tr_g_a $R/tr_g_b
restart
echo "$(date +%T) all done"
