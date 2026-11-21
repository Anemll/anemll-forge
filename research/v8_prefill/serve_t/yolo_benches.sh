#!/bin/zsh
# 21 Nov 2026 (Anemll: YOLO, after the outside review). Fresh launches of the standard serving launcher, banner checked:
#   A  sampled tree8 A/B in the reverse order of 10:22 (tree first), a second seed block (3100 / 3200):
#      tree_s2 (served: TREE_SAMPLED=1; also deploys the review fixes) then chain_s2 (TREE_SAMPLED=0); each the greedy
#      smoke (expect 29aef2ec9db18a3f) + realistic 8k / 32k + code / agent, thinking on, temperature 1.0, 384 tokens
#   B  prompt lookup ABBA (review window 3): off1, on1, on2, off2 (NGRAM_DRAFT=6 when on); each the code / agent set
#      greedy (320 tokens) and sampled (seed 4100, thinking on, temperature 1.0, 384 tokens)
# Every arm counts Pi requests in its server log (max_tokens 32768); forge stays up on the served config (off2).
set -u
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/21nov
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
O=$R/yolo; G=$O/yolo.log
mkdir -p $O $R/bench
BASE="v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8 tree=policy=branched tree=v8_dec8tx"
expa() { local A=(); for e in ${=1}; do A+=(--expect $e); done; echo ${A[@]}; }
out() { grep -E "SUMMARY|ABORT" $R/bench/$1.log | sed "s/^/$1: /" | tee -a $G; }
launch() {   # label prompts rungs tasks expect env...
  local lab=$1 pr=$2 ru=$3 ta=$4 ex=$5; shift 5
  echo "$(date +%T) launch $lab ($*)" | tee -a $G
  ( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED -u COLD_SCRUB \
      "$@" RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=$lab PROMPTS=$pr RUNGS=$ru ${ta:+TASKS=$ta} EXPECT="$ex" timeout 1200 zsh $T/run_arm.sh ) > $O/$lab.driver.log 2>&1
  grep -E "drafter ring snapshot:|tree: BANNER|prompt-lookup drafts:" $SL | cut -c1-230 | sed "s/^/$lab banner: /" | tee -a $G
  out $lab
}
same() {   # label prompts rungs expect seed
  $PY $T/tree_bench.py --prompts $T/$2 --label $1 --out $R/bench/$1.json --rungs $3 --temp 1.0 --think 1 --seed $5 --max-tokens 384 \
    --idle-s 5 $(expa "$4") > $R/bench/$1.log 2>&1
  out $1
}
pi() { echo "$1: Pi requests in this server's log: $(grep -c 'max_tokens 32768' $SL), errors: $(grep -cE 'Traceback|StopIteration' $SL)" | tee -a $G
       grep -vE "^   (state|prompt)( )?:" $SL > $O/$1.server.log; }
echo "$(date +%T) yolo benches: start" | tee -a $G
# ---- A: sampled tree8, reverse order, second seed block ----
EXT="$BASE tree=sampled=on lookup=off"; EXC="$BASE tree=sampled=off lookup=off"
launch tree_s2_smoke realistic_v1.json 2k 1,8 "$EXT"
same tree_s2_long realistic_v1.json 8k,32k "$EXT" 3100
same tree_s2_ca u1_code_agent.json code,agent "$EXT" 3200
pi tree_s2
launch chain_s2_smoke realistic_v1.json 2k 1,8 "$EXC" TREE_SAMPLED=0
same chain_s2_long realistic_v1.json 8k,32k "$EXC" 3100
same chain_s2_ca u1_code_agent.json code,agent "$EXC" 3200
pi chain_s2
$PY $R/../../serve_t/tree_sampled_compare.py $R/bench $O/sampled_rev_compare.json chain_s2 tree_s2 | tee -a $G
# ---- B: lookup ABBA ----
EXOFF="$BASE tree=sampled=on lookup=off"; EXON="$BASE tree=sampled=on tree=NGRAM_DRAFT="
for a in off1 on1 on2 off2; do
  if [[ $a == on* ]]; then ex=$EXON; envs=(NGRAM_DRAFT=6); else ex=$EXOFF; envs=(); fi
  launch lk_${a}_g u1_code_agent.json code,agent "" "$ex" $envs
  same lk_${a}_s u1_code_agent.json code,agent "$ex" 4100
  pi lk_$a
done
$PY $R/../../serve_t/lookup_compare.py $R/bench $O/lookup_compare.json | tee -a $G
echo "$(date +%T) yolo benches: done, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $G
