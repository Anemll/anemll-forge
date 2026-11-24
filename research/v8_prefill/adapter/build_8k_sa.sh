#!/bin/zsh
# 24 Nov 2026: 8K Core AI test build of the release export + the harness-hub super-adapter (adapter_export.py ->
# exports/release_vq3pA_sa), the same builder (anemll-forge-v02) and numerics as the base 8K build
# v3_tierA_s600_dual/release_vq3pA_mixh_s600_k1_mat_kvv8 (KL gate gate_kl512_tierA_s600), but only the 8K entries
# (v8_8k, p64_8k) and no M5 functions. Build on the SSD; forge.py compile --follow (one compile, journaled) alongside;
# swap watchdog on both. Cleanup after the test: coreai_cache_delete.py on the packages this run compiled (journal).
set -u
R=/Volumes/SSD4TB/anemll-forge-research; V02=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02
BPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python   # executable name "python"
H=${0:a:h}; T=$H/../tools; L=$R/coreai_seg_logs; OUT=$R/sa_test; DEST=$OUT/release_vq3pA_sa_kvv8
mkdir -p $OUT $L
[ -e $DEST/manifest.json ] && { echo "build exists: $DEST"; exit 1; }
cd $V02
env MODEL=/Volumes/SSD4TB/Qwen3.8-27B EXPORT_DIR=$R/exports/release_vq3pA_sa OUT=$OUT \
    SILU=tanh MLP_SILU=tanh GDN_SQ=16 GDN_SV=64 MLP_DS=1 GDN_FAST=1 ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096 \
    ATT_INT8MM=s8,s8b,sm8,pvf8 ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_PF8_UNIT=0.015625 QCONV_INT8=0 KV_KEYS_T=1 \
  nice -n 10 $BPY -u coreai/qwen38_coreai_build.py all --ctx 8192 --pctx 8192 --kv-cache-dtype v8 > $L/sa8k_build.log 2>&1 &
B=$!; echo "build pid $B -> $DEST (log $L/sa8k_build.log)"
bash $T/swap_watchdog.sh $B 1024 $L/watchdog_sa8k_build.log >/dev/null 2>&1 &
# journal (C/deployment_journal/m6.jsonl): attempt before the compile, result after
J() { DEPLOY_JOURNAL_ROOT=/Users/anemll/Documents/Codex/ios-prefill-20261030 PYTHONPATH=$T $PY - "$@" <<'EOF'
import sys, time, deploy_journal as dj
ev, aid, dest = sys.argv[1], sys.argv[2], sys.argv[3]
pkg = {"path": dest, "label": "super-adapter 8K test build (adapter/build_8k_sa.sh): chunks + head", "main_hash": "see coreai_cache_delete"}
run = {"mode": "forge_compile_follow", "out": dest, "driver_sha256": "build_8k_sa.sh"}
j = dj.Journal(run, aid.split(":")[0])
if ev == "attempt":
    j.n = 0; print(j.attempt(dj.m6_device(), pkg, request={"compile_mode": 2, "follow": True}))
else:
    j.result(dj.m6_device(), aid, pkg, sys.argv[4])
EOF
}
while [ ! -e $DEST/manifest.json ] && kill -0 $B 2>/dev/null; do sleep 15; done   # follow needs the manifest
[ -e $DEST/manifest.json ] || { wait $B; echo "build exited ($?) before writing a manifest"; tail -5 $L/sa8k_build.log; exit 1; }
AID=$(J attempt sa8k-$$-$(date +%s):1 $DEST)
cd $V02 && $PY forge.py compile --follow --build $DEST > $L/sa8k_compile.log 2>&1 &
C=$!; echo "compile pid $C (log $L/sa8k_compile.log, journal $AID)"
bash $T/swap_watchdog.sh $C 1024 $L/watchdog_sa8k_compile.log >/dev/null 2>&1 &
wait $B; BS=$?; wait $C; CS=$?
J result $AID $DEST $([ $CS = 0 ] && echo ok || echo failed) >/dev/null
echo "build exit $BS, compile exit $CS"; tail -3 $L/sa8k_compile.log
