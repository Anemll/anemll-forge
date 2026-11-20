#!/bin/zsh
# V8 server on the 8-layer decode-split build (v8_dec8: v8/build_dec8_all.sh, then v8/dec8_srv_manifest.py) with the
# split verify on (--v8-dec8 1; V8_DEC8=auto|on|off, V8_DEC8_MID) and the GPU tables of that build (w_dec8). Otherwise
# exactly v8_server.sh. Foreground; log in $L/v8_server.log; swap watchdog on the process.
# 18 Nov 2026: FP8 P.V prefill attention served (libgpu_v8_f8.dylib, F8=x; results/18nov/f8): V8_F8 (default x; empty =
# the plain fp16 P.V in the same binary), V8_SRV_DYLIB (default libgpu_v8_f8.dylib; libgpu_v8_srv3.dylib = before).
# The decode keep-alive is dec_split.py's default (libdec_gpu_clk.dylib, DEC_KA_HOLD_MS=60).
#   v8_server_dec8.sh [extra v8_server.py options]
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
B=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K; PORT=${PORT:-8767}
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && { echo "port $PORT in use"; exit 1; }
[ -s $L/v8_server.log ] && mv $L/v8_server.log $L/v8_server.$(date -r $L/v8_server.log +%Y%m%d-%H%M%S).log   # every previous log kept, timestamped
env MODEL=$B/model SILU=tanh MLP_SILU=tanh GDN_SQ=16 GDN_SV=64 MLP_DS=1 MLP_DS_DYN=0 V3_KV_IN=1 V3_PREFILL=0 \
    EMBED_NPY=$B/model/embed_tokens_fp16.npy RUNTIME=coreai DRAFTER=$B/drafter/all3 COREAI_DRAFTER_COMPUTE=ane \
    MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 F8=${V8_F8-x} \
    FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1 I8KV=1 LRM=1 QK8=1 TKV=2048 \
  /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python $P/v8_server.py --v8-dec8 1 \
    --v8-gpu-manifest $P/w_dec8/manifest_best.json --v8-dylib ${V8_SRV_DYLIB:-libgpu_v8_f8.dylib} "$@" \
    --hf $B/model --model-dir /Volumes/SSD4TB/anemll-forge-research/v8_dec8 --ctx ${CTX:-98304} --host 127.0.0.1 --port $PORT \
    --runtime coreai --kv-cache-dtype auto --draft $B/drafter/all3/dflash2_lut4_gptq_nohead.aimodel --drafter $B/drafter/all3 \
    > $L/v8_server.log 2>&1 &
G=$!; echo "server pid $G (port $PORT, log $L/v8_server.log)"
/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $G 1536 $L/watchdog_v8_server.log >/dev/null 2>&1 &
wait $G
