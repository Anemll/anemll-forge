#!/bin/zsh
# V8-prefill server (v8_server.py over the anemll-forge-v02 release server): v8_shared2 build, 96K context, DFlash2
# all3 drafter, the V8 engine's best GPU switches. Foreground; log in $L/v8_server.log; swap watchdog on the process.
#   v8_server.sh [extra v8_server.py options, e.g. --v8-kv8 0 --v8-attproj-from 32768]
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
B=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K; PORT=${PORT:-8767}
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && { echo "port $PORT in use"; exit 1; }
[ -s $L/v8_server.log ] && mv $L/v8_server.log $L/v8_server.prev.log   # the previous run's log is kept once
env MODEL=$B/model SILU=tanh MLP_SILU=tanh GDN_SQ=16 GDN_SV=64 MLP_DS=1 MLP_DS_DYN=0 V3_KV_IN=1 V3_PREFILL=0 \
    EMBED_NPY=$B/model/embed_tokens_fp16.npy RUNTIME=coreai DRAFTER=$B/drafter/all3 COREAI_DRAFTER_COMPUTE=ane \
    MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 \
    FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1 I8KV=1 LRM=1 QK8=1 TKV=2048 \
  /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python $P/v8_server.py "$@" \
    --hf $B/model --model-dir /Volumes/SSD4TB/anemll-forge-research/v8_shared2 --ctx ${CTX:-98304} --host 127.0.0.1 --port $PORT \
    --runtime coreai --kv-cache-dtype auto --draft $B/drafter/all3/dflash2_lut4_gptq_nohead.aimodel --drafter $B/drafter/all3 \
    > $L/v8_server.log 2>&1 &
G=$!; echo "server pid $G (port $PORT, log $L/v8_server.log)"
/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $G 1536 $L/watchdog_v8_server.log >/dev/null 2>&1 &
wait $G
