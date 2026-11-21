#!/bin/zsh
# 21 Nov 2026: tree8 for sampled (Pi) requests, A/B on two fresh launches of the standard serving launcher:
#   chain_s  TREE_SAMPLED=0 (sampled cycles on the served chain, as before)
#   tree_s   the launcher's default (TREE_SAMPLED=1: the tree for sampled cycles; tree_verify.walk_sampled)
# Each: the greedy smoke first (realistic 2K positions 1, 8; expect 29aef2ec9db18a3f, the greedy path is unchanged),
# then Pi-like sampled requests (thinking on, temperature 1.0, the server's thinking defaults top_p 0.95 / top_k 20,
# one seed per request, the same in both arms; 384 tokens) on the realistic 8k + 32k rungs and the code / agent sets.
# tree_s runs last and stays up (the served default).
set -u
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/21nov
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
G=$R/tree_sampled/ab.log
mkdir -p $R/tree_sampled $R/bench
EX="v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8 tree=policy=branched tree=v8_dec8tx lookup=off"
arm() {   # label TREE_SAMPLED-env-or-'' expect-sampled
  local lab=$1 ts=$2 es=$3
  local EXA=(); for e in ${=EX} tree=sampled=$es; do EXA+=(--expect $e); done
  echo "$(date +%T) arm $lab: restart (TREE_SAMPLED=${ts:-launcher default})" | tee -a $G
  ( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED \
      ${ts:+TREE_SAMPLED=$ts} RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=${lab}_smoke RUNGS=2k TASKS=1,8 EXPECT="$EX tree=sampled=$es" \
      timeout 900 zsh $T/run_arm.sh ) > $R/tree_sampled/$lab.driver.log 2>&1
  grep -E "drafter ring snapshot:|tree: BANNER|dec8: decode GPU engine" $SL | cut -c1-230 | sed "s/^/$lab banner: /" | tee -a $G
  grep -E "SUMMARY|ABORT" $R/bench/${lab}_smoke.log | sed "s/^/$lab: /" | tee -a $G
  grep -q "SUMMARY" $R/bench/${lab}_smoke.log || { echo "$lab: smoke failed" | tee -a $G; return 1; }
  $PY $T/tree_bench.py --prompts $T/realistic_v1.json --label ${lab}_long --out $R/bench/${lab}_long.json --rungs 8k,32k \
    --temp 1.0 --think 1 --seed 2100 --max-tokens 384 --idle-s 5 $EXA > $R/bench/${lab}_long.log 2>&1
  grep -E "SUMMARY|ABORT" $R/bench/${lab}_long.log | sed "s/^/$lab: /" | tee -a $G
  $PY $T/tree_bench.py --prompts $T/u1_code_agent.json --label ${lab}_ca --out $R/bench/${lab}_ca.json --rungs code,agent \
    --temp 1.0 --think 1 --seed 2200 --max-tokens 384 --idle-s 5 $EXA > $R/bench/${lab}_ca.log 2>&1
  grep -E "SUMMARY|ABORT" $R/bench/${lab}_ca.log | sed "s/^/$lab: /" | tee -a $G
  cp -p $SL $R/tree_sampled/$lab.server.log
}
echo "$(date +%T) tree sampled A/B: start" | tee -a $G
arm chain_s 0 off
arm tree_s "" on
echo "$(date +%T) tree sampled A/B: done, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $G
