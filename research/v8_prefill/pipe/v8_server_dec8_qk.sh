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
# 21 Nov 2026: TREE_SAMPLED=1, the tree for sampled requests too (Pi: temperature 1.0; tree_verify.walk_sampled, the
# chain's acceptance rule per child, the plain-sampling distribution). TREE_SAMPLED=0: sampled cycles on the chain.
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
export PORT=${PORT:-8765} V8_SRV_DYLIB=${V8_SRV_DYLIB:-libgpu_v8_f8q.dylib} QKE=${QKE-tg} NGRAM_DRAFT=${NGRAM_DRAFT:-0} DRAFT_RING_SNAP=${DRAFT_RING_SNAP:-fix}
[ "${SERVED_TREE:-1}" = 0 ] && exec $P/v8_server_dec8.sh "$@"
# 22 Nov 2026 (Anemll, deploy): sliced decode entries served. Build v8_dec8ts (= v8_dec8tx + sliced chain/tree
# entries, same weights, 81 functions per chunk) with RUNG_SLICE=24,40,48,56 (KV bound at the 32K/64K entries, attention
# over the prefix only): long session +2.5% (+3.4% incl. one slow served set), replies identical, KL 0 (results/22nov,
# step B). GPU tables w_dec8ts. Rollback: SERVED_SLICE=0 v8_server_dec8_qk.sh (the 21 Nov v8_dec8tx config; =
# v8_server_dec8_qk.pre_slice4.sh). RUNG_SLICE= (empty) keeps this build without slicing.
if [ "${SERVED_SLICE:-1}" = 0 ]; then
  export BUILD=${BUILD:-/Volumes/SSD4TB/anemll-forge-research/v8_dec8tx} GM=${GM:-$P/w_dec8tx/manifest_best.json}
else
  export BUILD=${BUILD:-/Volumes/SSD4TB/anemll-forge-research/v8_dec8ts} GM=${GM:-$P/w_dec8ts/manifest_best.json} RUNG_SLICE=${RUNG_SLICE-24,40,48,56}
fi
export TREE=${TREE-1} TREE_SAMPLED=${TREE_SAMPLED:-1}
# 22 Nov 2026: KV-wrap leak fix (forge grew 1.8 GB per context shrink and grow; results/22nov/memleak): engine/
# build_engines.sh uw builds libgpu_v8_f8q_uw.dylib + libdec_gpu_tree_uw.dylib; serve them with
# V8_SRV_DYLIB=libgpu_v8_f8q_uw.dylib DEC_DYLIB=libdec_gpu_tree_uw.dylib v8_server_dec8_qk.sh (served this way since 18:20,
# replies identical). The defaults at the top still name the 19 Nov dylibs until Anemll approves the switch.
exec $P/v8_server_tree.sh "$@"
