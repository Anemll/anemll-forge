#!/bin/zsh
# Queue 16b (19 Nov 2026): REAL end-to-end hybrid decode A/B: ANE (8-row verify FFN + DeltaNet, DFlash2 drafter) + the
# 16 attention layers on the GPU (split verify forced on at every context, V8_DEC8=on, both arms):
#   A  served decode attention: libdec_gpu_clk.dylib + dec_kernels.metal, dec_attn_r4 (FP16 P x INT8 V)
#   B  research: libdec_gpu_f8d.dylib + dec_kernels_f8d.metal, dec_attn_r4f8 (e4m3 P x e4m3 V copy), DEC_F8V=1
# Fresh server per arm on port 8771 (v8_server_dec8_qk.sh: FP8 x prefill + QKE=tg + keep-alive), A B A B;
# scripts/m6_compare_bench.py at 32,768 / 65,536 / 98,304 (cold prompt = entry - 512 tokens, 256 tokens greedy, then 3
# identical requests: decode median). Banner from the server log (dylib / kernels path + sha256 + variant; the server
# refuses to start on an EXPECT_* mismatch). DEC_LOGITS_DIR keeps the cold request's verify logits for KL. Forge on
# 8765 stays down.
F=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge; PY=$F/.venv/bin/python; P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
O=/Volumes/SSD4TB/anemll-forge-research/contention/dec_f8e2e; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs
TOK=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model; CTX=${CTXS:-32768,65536,98304}; PORT=8771; mkdir -p $O
for ARM in ${=ARMS:-A1 B1 A2 B2}; do
  lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && { PORT=$PORT $P/srv_stop.sh; sleep 20; }
  rm -rf $O/lg_$ARM; mkdir -p $O/lg_$ARM
  if [[ $ARM == A* ]]; then
    VARS=(V8_DEC8=on DEC_LOGITS_DIR=$O/lg_$ARM EXPECT_DEC_ATTN=dec_attn_r4 EXPECT_DEC_METAL=00f351b5edc893b7 EXPECT_DEC_DYLIB=2b64e979a91a414f)
  else
    VARS=(V8_DEC8=on DEC_LOGITS_DIR=$O/lg_$ARM DEC_DYLIB=libdec_gpu_f8d.dylib DEC_METAL=dec_kernels_f8d.metal DEC_ATTN=dec_attn_r4f8 DEC_F8V=1
          EXPECT_DEC_ATTN=dec_attn_r4f8 EXPECT_DEC_METAL=e97f2c93f9cb1a18 EXPECT_DEC_DYLIB=ad3783b24795d7c0)
  fi
  echo "$(date +%T) arm $ARM: starting server on $PORT (swap $(sysctl -n vm.swapusage | awk '{print $6}')) env $VARS"
  (cd $P && env $VARS PORT=$PORT nohup $P/v8_server_dec8_qk.sh >/dev/null 2>&1 &)
  for i in $(seq 1 120); do curl -s -m 2 localhost:$PORT/v1/models | grep -q -i qwen && break; grep -q -E "ABORT|Traceback" $L/v8_server.log 2>/dev/null && break; sleep 5; done
  grep -E "ABORT|Traceback" $L/v8_server.log && { echo "arm $ARM: server failed (banner mismatch or error): stopping"; tail -20 $L/v8_server.log; exit 1; }
  curl -s -m 2 localhost:$PORT/v1/models | grep -q -i qwen || { echo "arm $ARM: server did not come up"; tail -20 $L/v8_server.log; exit 1; }
  BAN=$(grep -E "dec8: BANNER" $L/v8_server.log | tail -1 | cut -c12-); [[ -z $BAN ]] && { echo "arm $ARM: no decode banner: stopping"; exit 1; }
  echo "BANNER arm $ARM | $BAN"; echo "BANNER arm $ARM prefill | $(grep -E 'v8: variant' $L/v8_server.log | tail -1 | cut -c12-300)"
  FID=$(curl -s -m 2 localhost:$PORT/v1/models | $PY -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])")
  (cd $F/scripts && $PY m6_compare_bench.py --url http://127.0.0.1:$PORT/v1 --model "$FID" --tokenizer $TOK --ctx $CTX --label f8e2e_$ARM --out $O/bench_$ARM.json > $O/bench_$ARM.log 2>&1)
  echo "$(date +%T) arm $ARM bench rc=$?"; grep -E "prompt [0-9]+ \(cached" $L/v8_server.log | cut -c1-330
  grep -E "dec8: [0-9]+ split|dec8: variant" $L/v8_server.log | tail -8 | cut -c1-200
  PORT=$PORT $P/srv_stop.sh; sleep 15; cp $L/v8_server.log $O/server_$ARM.log
done
echo "$(date +%T) all done (forge on 8765 left down)"
