#!/bin/zsh
# Queue 9a (19 Nov 2026): the drop-in hidden-decode LUT GEMM (I8TV=_k256ws, lut_gemm32_i8t_k256ws) end to end at 95K
# against the served config (QKE=tg, I8TV=_k256), the same binary (libgpu_v8_gdnw.dylib), interleaved fresh processes
# k a / ws a / k b / ws b; banner checks F8, Q.K and LUT kernels; last-token logits compared; forge restarted on 8765
# with the serving launcher (v8_server_dec8_qk.sh: FP8 x + QKE=tg + keep-alive).
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; R=/Volumes/SSD4TB/anemll-forge-research/contention/gdn/ws_e2e; mkdir -p $R
restart() { echo "$(date +%T) restarting forge on 8765 (v8_server_dec8_qk.sh)"; cd $P; PORT=8765 nohup $P/v8_server_dec8_qk.sh >/dev/null 2>&1 &
  for i in $(seq 1 90); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
  echo "$(date +%T) forge ready"; grep -E "v8: variant" /Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log | cut -c1-200; }
echo "$(date +%T) stopping forge on 8765"; PORT=8765 $P/srv_stop.sh; sleep 30
for T in k_a ws_a k_b ws_b; do
  mkdir -p $R/tr_$T; echo "$(date +%T) trace $T (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"
  V=_k256; [[ $T == ws* ]] && V=_k256ws
  env V8_DYLIB=libgpu_v8_gdnw.dylib F8=x QKE=tg I8TV=$V EXPECT_F8=x EXPECT_QK=attn_qk8isk_tg EXPECT_LUT=lut_gemm32_i8t$V $D/v8_trace.sh $R/tr_$T 95232
  grep -E "^BANNER|ABORT" $R/tr_$T.log | cut -c1-300
  grep -q "^ABORT" $R/tr_$T.log && { echo "variant mismatch: stopping"; restart; exit 1; }
  $PY $D/trace_analyze.py $R/tr_$T > /dev/null 2>$R/tr_$T/analyze.err
  grep "^run" $R/tr_$T.log
  sleep 30
done
$PY $D/logits_cmp.py $R/tr_k_a $R/tr_k_b $R/tr_ws_a $R/tr_ws_b
restart
echo "$(date +%T) all done"
