#!/bin/zsh
# Serving launcher. 21 Nov 2026 (Anemll, 02:30): tree8 is the served default. The V8.2 served config (FP8 x P.V
# prefill, staged Q.K epilogue QKE=tg in libgpu_v8_f8q.dylib, INT8 KV, decode keep-alive 60 ms, ctx 98304, model id
# qwen38-27b-ane) on the v8_dec8tx build (the served entries + the exact tree8 verify entries, corrx0, same weights)
# with tree verify on (TREE=1, policy branched; v8_server_tree.sh: libdec_gpu_tree.dylib + dec_kernels_tree.metal,
# GPU tables w_dec8tx). Prompt lookup off (NGRAM_DRAFT=0). Port 8765 unless PORT is set.
#   v8_server_dec8_qk.sh [extra v8_server.py options]
# Rollback to the 19-20 Nov served chain (v8_dec8 build, libdec_gpu_clk.dylib): SERVED_TREE=0 v8_server_dec8_qk.sh
# (= v8_server_dec8_qk.pre_tree8.sh on port 8765). TREE=0 keeps this build and binaries with the chain only.
# 21 Nov 2026 (Anemll, 07:27): DRAFT_RING_SNAP=fix, the 'system' snapshot also keeps the drafter's ring rows and
# writes them back on restore (used-server replies = a fresh start's; results/21nov). DRAFT_RING_SNAP=off: before.
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
export PORT=${PORT:-8765} V8_SRV_DYLIB=${V8_SRV_DYLIB:-libgpu_v8_f8q.dylib} QKE=${QKE-tg} NGRAM_DRAFT=${NGRAM_DRAFT:-0} DRAFT_RING_SNAP=${DRAFT_RING_SNAP:-fix}
[ "${SERVED_TREE:-1}" = 0 ] && exec $P/v8_server_dec8.sh "$@"
export BUILD=${BUILD:-/Volumes/SSD4TB/anemll-forge-research/v8_dec8tx} GM=${GM:-$P/w_dec8tx/manifest_best.json} TREE=${TREE-1}
exec $P/v8_server_tree.sh "$@"
