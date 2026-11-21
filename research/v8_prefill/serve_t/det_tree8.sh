#!/bin/zsh
# 21 Nov 2026: tree8 determinism. The 2K smoke (realistic_v1 positions 1, 8) as repeated runs:
#   (a) tree8 3rd run on the current server (after smoke1, smoke2, 2k rung, code/agent)
#   (b) chain (SERVED_TREE=0 = the 19-20 Nov served chain) fresh launch, run 1 and run 2 on the same server
#   (c) tree8 fresh launch (the served launcher's defaults), run 1 and run 2 on the same server (left serving)
set -u
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/21nov
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
EXT="v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8 tree=policy=branched tree=v8_dec8tx lookup=off"
EXC="v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_clk.dylib tree=mode=chain lookup=off"
same() {   # label expect
  local EXA=(); for e in ${=2}; do EXA+=(--expect $e); done
  $PY $T/tree_bench.py --prompts $T/realistic_v1.json --label $1 --out $R/bench/$1.json --rungs 2k --tasks 1,8 $EXA > $R/bench/$1.log 2>&1
  grep -E "sha [0-9a-f]{16}$|SUMMARY|ABORT" $R/bench/$1.log | sed "s/^/$1: /" | tee -a $R/det_tree8.log
}
fresh() {   # label expect env...
  local lab=$1 ex=$2; shift 2
  ( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB "$@" RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=$lab RUNGS=2k TASKS=1,8 EXPECT="$ex" timeout 900 zsh $T/run_arm.sh ) > $R/$lab.driver.log 2>&1
  grep -E "sha [0-9a-f]{16}$|SUMMARY|ABORT" $R/bench/$lab.log | sed "s/^/$lab: /" | tee -a $R/det_tree8.log
}
echo "$(date +%T) det: start" | tee -a $R/det_tree8.log
same served_tree8_smoke2b "$EXT"
fresh det_chain_fresh1 "$EXC" SERVED_TREE=0
same det_chain_same2 "$EXC"
fresh served_tree8_smoke3 "$EXT"
same served_tree8_smoke3b "$EXT"
echo "$(date +%T) det: done, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $R/det_tree8.log
