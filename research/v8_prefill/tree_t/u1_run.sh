#!/bin/zsh
# U1 (20 Nov 2026): end-to-end, measured: chain / chain + lookup / tree8 (corrx0) / tree8 + lookup, each a forge launch
# (run_arm.sh), the realistic 9-prompt rungs 2K / 8K / 32K / 64K, then the code-heavy and agent-style sets
# (u1_code_agent.json) on the same server; banner expectations per arm (abort on mismatch). Restores the served
# config at the end (final_served6 smoke).
set -u
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree; P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
TX=/Volumes/SSD4TB/anemll-forge-research/v8_dec8tx; GX=$P/w_dec8tx/manifest_best.json
arm() {   # label launcher expect env...
  local lab=$1 la=$2 ex=$3; shift 3
  echo "$(date +%T) U1 arm $lab: $la $*" | tee -a $R/u/u1.log
  ( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE "$@" LAUNCHER=$la ARM=u1_$lab RUNGS=2k,8k,32k,64k EXPECT="$ex" timeout 3600 zsh run_arm.sh ) > $R/u/u1_$lab.driver.log 2>&1
  grep SUMMARY $R/bench/u1_$lab.log | tee -a $R/u/u1.log
  local EXA=(); for e in ${=ex}; do EXA+=(--expect $e); done
  $PY $T/tree_bench.py --prompts $T/u1_code_agent.json --label u1_${lab}_ca --out $R/bench/u1_${lab}_ca.json --rungs code,agent $EXA 2>&1 | tee $R/bench/u1_${lab}_ca.log | grep SUMMARY | tee -a $R/u/u1.log
  sleep 60
}
mkdir -p $R/u
arm chain   v8_server_dec8_qk.sh "v8=qk=attn_qk8isk_tg dec8=libdec_gpu_clk.dylib lookup=off"
arm chainlk v8_server_dec8_qk.sh "v8=qk=attn_qk8isk_tg dec8=libdec_gpu_clk.dylib lookup=on" NGRAM_DRAFT=6
arm tree8   v8_server_tree.sh    "tree=policy=branched tree=v8_dec8tx dec8=libdec_gpu_tree.dylib lookup=off" BUILD=$TX GM=$GX TREE=1
arm tree8lk v8_server_tree.sh    "tree=policy=branched tree=v8_dec8tx dec8=libdec_gpu_tree.dylib lookup=on" BUILD=$TX GM=$GX TREE=1 NGRAM_DRAFT=6
echo "$(date +%T) U1 arms done; restoring the served config" | tee -a $R/u/u1.log
( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE LAUNCHER=v8_server_dec8_qk.sh ARM=final_served6 RUNGS=2k TASKS=1,8 EXPECT="v8=qk=attn_qk8isk_tg dec8=libdec_gpu_clk.dylib" timeout 600 zsh run_arm.sh ) > $R/u/final_served6.driver.log 2>&1
grep -o "sha [0-9a-f]\{16\}" $R/bench/final_served6.log | tr '\n' ' ' | tee -a $R/u/u1.log; echo | tee -a $R/u/u1.log
echo "$(date +%T) U1 done, forge served pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $R/u/u1.log
