#!/bin/zsh
# Portable launcher for the Hugging Face V8-96K repo (22 Nov 2026): the served configuration of the development Mac
# (pipe/v8_server_dec8_qk.sh -> v8_server_tree.sh, which carry that Mac's paths) with every path a variable:
#   build coreai/m6-96k-dec8ts (8 packages, 81 entries each: target, V8 prefill, decode split, tree8 verify and the
#   sliced decode entries), tree8 branched verify for greedy and sampled requests, sliced decode entries 24/40/48/56K,
#   FP8 x P.V prefill, staged Q.K epilogue (QKE=tg), INT8 KV, GPU clock keep-alive, drafter ring snapshot, prompt
#   lookup off, ctx 98304, the KV-wrap leak fix (the *_uw dylibs), forge's own memory watchdog.
#   REPO=<HF repo download> [TABLES=<GPU table dir>] [PORT=8765] [LOGDIR=~/Library/Logs/anemll-forge] [PY=python3]
#   zsh code/anemll-forge/research/v8_prefill/hf/serve_v8_96k.sh [extra v8_server.py options]
# Before the first run (DEPLOY.md): forge.py compile of the build and the drafter, the GPU tables
# (pipe/gpu_tables_kit.py install, TABLES defaults to $REPO/gpu/w_dec8ts_local), the engine dylibs
# (pipe/engine/build_engines.sh uw). Foreground until the server exits; log $LOGDIR/v8_server.log.
set -u
H=${0:a:h}; V=${H:h}; P=$V/pipe; CODE=${V:h:h}
: ${REPO:?set REPO to the downloaded Hugging Face repo}
REPO=${REPO:a}; TABLES=${TABLES:-$REPO/gpu/w_dec8ts_local}; PORT=${PORT:-8765}; PY=${PY:-python3}
LOGDIR=${LOGDIR:-$HOME/Library/Logs/anemll-forge}; mkdir -p $LOGDIR; L=$LOGDIR/v8_server.log
BUILD=${BUILD:-$REPO/coreai/m6-96k-dec8ts}; GM=$TABLES/manifest_best.json
for f in $BUILD/manifest.json $GM $P/engine/libgpu_v8_f8q_uw.dylib $P/engine/libdec_gpu_tree_uw.dylib $REPO/drafter/all3 $REPO/model/tokenizer.json; do
  [ -e $f ] || { echo "missing $f (see DEPLOY.md)"; exit 1; }
done
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && { echo "port $PORT in use"; exit 1; }
[ -s $L ] && mv $L ${L:r}.$(date -r $L +%Y%m%d-%H%M%S).log
# deploy_journal: the portable copy in tools/ (journal under DEPLOY_JOURNAL_ROOT, default ~/.anemll-forge)
env PYTHONPATH=$V/tools${PYTHONPATH:+:$PYTHONPATH} V8_V02=$CODE \
    MODEL=$REPO/model SILU=tanh MLP_SILU=tanh GDN_SQ=16 GDN_SV=64 MLP_DS=1 MLP_DS_DYN=0 V3_KV_IN=1 V3_PREFILL=0 \
    EMBED_NPY=$REPO/model/embed_tokens_fp16.npy RUNTIME=coreai DRAFTER=$REPO/drafter/all3 COREAI_DRAFTER_COMPUTE=ane \
    MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 F8=x QKE=tg TREE=1 TREE_SAMPLED=1 NGRAM_DRAFT=0 DRAFT_RING_SNAP=fix \
    RUNG_SLICE=${RUNG_SLICE-24,40,48,56} DEC_DYLIB=libdec_gpu_tree_uw.dylib DEC_METAL=dec_kernels_tree.metal \
    DEC_KA_CTL=${DEC_KA_CTL:-$LOGDIR/ka} \
    FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1 I8KV=1 LRM=1 QK8=1 TKV=2048 \
  $PY $P/v8_server.py --v8-dec8 1 --v8-gpu-manifest $GM --v8-dylib libgpu_v8_f8q_uw.dylib "$@" \
    --hf $REPO/model --model-dir $BUILD --ctx ${CTX:-98304} --host 127.0.0.1 --port $PORT \
    --runtime coreai --kv-cache-dtype auto --draft $REPO/drafter/all3/dflash2_lut4_gptq_nohead.aimodel --drafter $REPO/drafter/all3 \
    > $L 2>&1 &
G=$!; echo "server pid $G (port $PORT, log $L)"
bash $V/tools/forge_watchdog.sh $G ${FORGE_FP_MAX_MB:-12288} $LOGDIR/watchdog_v8_server.log >/dev/null 2>&1 &
wait $G
