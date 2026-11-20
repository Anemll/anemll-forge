#!/bin/zsh
# 20 Nov 2026 tree study: the served configuration (v8_server_dec8_qk.sh: FP8 x prefill, QKE=tg, keep-alive) on the
# dec8t build (v8/build_dec8t_all.sh: the served entries + tree verify entries, same weights) with tree verify on
# (TREE=1, tree_verify.py; TREE_POLICY etc. pass through), the decode GPU dylib / kernels with ancestor masks
# (libdec_gpu_tree.dylib, dec_kernels_tree.metal) and the build's own GPU tables (w_dec8t). TREE=0: the same build and
# binaries with the chain only. Foreground; log in $L/v8_server.log; swap watchdog.
#   PORT=8765 v8_server_tree.sh [extra v8_server.py options]
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
B=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K; PORT=${PORT:-8767}; BUILD=${BUILD:-/Volumes/SSD4TB/anemll-forge-research/v8_dec8t}
GM=${GM:-$P/w_dec8t/manifest_best.json}
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && { echo "port $PORT in use"; exit 1; }
[ -s $L/v8_server.log ] && mv $L/v8_server.log $L/v8_server.$(date -r $L/v8_server.log +%Y%m%d-%H%M%S).log
env MODEL=$B/model SILU=tanh MLP_SILU=tanh GDN_SQ=16 GDN_SV=64 MLP_DS=1 MLP_DS_DYN=0 V3_KV_IN=1 V3_PREFILL=0 \
    EMBED_NPY=$B/model/embed_tokens_fp16.npy RUNTIME=coreai DRAFTER=$B/drafter/all3 COREAI_DRAFTER_COMPUTE=ane \
    MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 F8=${V8_F8-x} QKE=${QKE-tg} TREE=${TREE-1} \
    DEC_DYLIB=${DEC_DYLIB:-libdec_gpu_tree.dylib} DEC_METAL=${DEC_METAL:-dec_kernels_tree.metal} \
    FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1 I8KV=1 LRM=1 QK8=1 TKV=2048 \
  /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python $P/v8_server.py --v8-dec8 1 \
    --v8-gpu-manifest $GM --v8-dylib ${V8_SRV_DYLIB:-libgpu_v8_f8q.dylib} "$@" \
    --hf $B/model --model-dir $BUILD --ctx ${CTX:-98304} --host 127.0.0.1 --port $PORT \
    --runtime coreai --kv-cache-dtype auto --draft $B/drafter/all3/dflash2_lut4_gptq_nohead.aimodel --drafter $B/drafter/all3 \
    > $L/v8_server.log 2>&1 &
G=$!; echo "server pid $G (port $PORT, log $L/v8_server.log)"
/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $G 1536 $L/watchdog_v8_server.log >/dev/null 2>&1 &
wait $G
