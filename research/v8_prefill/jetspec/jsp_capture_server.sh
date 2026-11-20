#!/bin/zsh
# JetSpec++ tap capture server (queue 7, 19 Nov 2026). The served configuration of v8_server_dec8_qk.sh, as the forge
# process on 8765 runs it (same environment, arguments, build v8_dec8 and DFlash2 drafter; F8=x, QKE=tg, keep-alive
# defaults), with pipe/v8_server_tapdump.py in place of v8_server.py, DRAFT_INGEST=all, port 8771 and its own log.
# ONLY with the served forge stopped (one model at a time on the M6); refuses to start while 8765 listens.
#   OUT=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/capture_<name> PROMPTS=<prompts json> zsh jsp_capture_server.sh
set -u
OUT=${OUT:?OUT=capture dir} PROMPTS=${PROMPTS:?PROMPTS=pinned prompts json} PORT=${PORT:-8771}
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
B=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "served forge listens on 8765: stop it first (one model at a time)"; exit 1; }
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && { echo "port $PORT in use"; exit 1; }
mkdir -p $OUT/traces
[ -n "$(ls -A $OUT/traces)" ] && { echo "$OUT/traces is not empty: use a fresh OUT"; exit 1; }
{
  echo "jsp capture server $(date '+%Y-%m-%d %H:%M:%S %Z'), port $PORT, prompts $PROMPTS"
  for f in $P/v8_server_tapdump.py $P/v8_server.py $P/v8_server_dec8.sh $P/v8_server_dec8_qk.sh $P/dec_split.py \
           $P/engine/libgpu_v8_f8q.dylib $P/engine/libdec_gpu_clk.dylib $P/engine/dec_kernels.metal $PROMPTS; do
    [ -f $f ] && echo "sha256 $(shasum -a 256 $f | cut -c1-16) $f"
  done
  echo "build v8_dec8 manifest sha256 $(shasum -a 256 /Volumes/SSD4TB/anemll-forge-research/v8_dec8/manifest.json | cut -c1-16)"
  echo "ka switch: $(cat /Volumes/SSD4TB/anemll-forge-research/contention/ctl/ka 2>/dev/null)"
} | tee $OUT/banner.txt
env MODEL=$B/model SILU=tanh MLP_SILU=tanh GDN_SQ=16 GDN_SV=64 MLP_DS=1 MLP_DS_DYN=0 V3_KV_IN=1 V3_PREFILL=0 \
    EMBED_NPY=$B/model/embed_tokens_fp16.npy RUNTIME=coreai DRAFTER=$B/drafter/all3 COREAI_DRAFTER_COMPUTE=ane \
    MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 F8=x QKE=tg V8_SRV_DYLIB=libgpu_v8_f8q.dylib \
    FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1 I8KV=1 LRM=1 QK8=1 TKV=2048 \
    DRAFT_INGEST=all TAP_DUMP_DIR=$OUT/traces TAP_PROMPTS=$PROMPTS \
  $PY $P/v8_server_tapdump.py --v8-dec8 1 --v8-gpu-manifest $P/w_dec8/manifest_best.json --v8-dylib libgpu_v8_f8q.dylib \
    --hf $B/model --model-dir /Volumes/SSD4TB/anemll-forge-research/v8_dec8 --ctx 98304 --host 127.0.0.1 --port $PORT \
    --runtime coreai --kv-cache-dtype auto --draft $B/drafter/all3/dflash2_lut4_gptq_nohead.aimodel --drafter $B/drafter/all3 \
    > $OUT/server.log 2>&1 &
G=$!; echo "capture server pid $G (port $PORT, log $OUT/server.log)"
/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $G 1536 $OUT/watchdog.log >/dev/null 2>&1 &
wait $G
