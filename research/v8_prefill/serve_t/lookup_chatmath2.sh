#!/bin/zsh
# 21 Nov 2026 (M6 ANE bot batch, item 2, rerun of part A: the first run's --tasks argument was not split by zsh). Runs only while Pi is idle (no client connection for 60 s and the last
# request >= 2 min old; waits up to 45 min, else exits without touching forge). Fresh launches of the standard
# serving launcher, banner checked first (one greedy request), then:
#   A  prompt lookup off1 / on1 / on2 / off2 (NGRAM_DRAFT=6 when on) with tree8 sampled: realistic_v1 math + chat tasks
#      (3-8) at 2k / 8k / 32k, thinking on, temperature 1.0 (Pi settings: the server's thinking defaults), seed 5100, 384 tok
#   B  one sampled logging launch (served config + TREE_LOG + TREE_LOG_DIST=1: per cycle the drafter inputs and every
#      verified row's sampling distribution) on all 9 tasks at 2k / 8k / 32k, seed 6100, for fork_rank_sim.py sampled
#   C  the served config restored (launcher defaults: tree8 sampled, lookup off, ring fix) + the greedy smoke
set -u
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/21nov
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
TL=/Volumes/SSD4TB/anemll-forge-research/treelog_s21
O=$R/lookup_cm; G=$O/run.log
mkdir -p $O $R/bench
BASE="v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8 tree=policy=branched tree=v8_dec8tx tree=sampled=on"
EXOFF="$BASE lookup=off"; EXON="$BASE tree=NGRAM_DRAFT="
expa() { local A=(); for e in ${=1}; do A+=(--expect $e); done; echo ${A[@]}; }
out() { grep -E "SUMMARY|ABORT" $R/bench/$1.log | sed "s/^/$1: /" | tee -a $G; }
pi_idle() {
  local i quiet=0 last now age
  for i in $(seq 1 540); do
    if [ "$(lsof -nP -iTCP:8765 -sTCP:ESTABLISHED 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')" = 0 ]; then quiet=$((quiet + 5)); else quiet=0; fi
    last=$(grep -oE "^\[[0-9:]+\] request: prompt" $SL 2>/dev/null | tail -1 | cut -c2-9)
    now=$(date +%s); age=999
    [ -n "$last" ] && age=$(( now - $(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) $last" +%s) ))
    [ $quiet -ge 60 ] && [ $age -ge 120 ] && return 0
    sleep 5
  done
  return 1
}
launch() {   # label expect env...
  local lab=$1 ex=$2; shift 2
  echo "$(date +%T) launch $lab ($*)" | tee -a $G
  ( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED -u COLD_SCRUB \
      -u TREE_LOG -u TREE_LOG_DIST "$@" RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=${lab}_b RUNGS=2k TASKS=1 EXPECT="$ex" timeout 900 zsh $T/run_arm.sh ) > $O/$lab.driver.log 2>&1
  grep -E "drafter ring snapshot:|tree: BANNER|prompt-lookup drafts:" $SL | cut -c1-230 | sed "s/^/$lab banner: /" | tee -a $G
  grep -q "SUMMARY" $R/bench/${lab}_b.log || { echo "$lab: banner / launch failed" | tee -a $G; tail -5 $O/$lab.driver.log | tee -a $G; return 1; }
}
sampled() {   # label expect tasks seed
  local TA=(); [ -n "$3" ] && TA=(--tasks $3)
  $PY $T/tree_bench.py --prompts $T/realistic_v1.json --label $1 --out $R/bench/$1.json --rungs 2k,8k,32k $TA \
    --temp 1.0 --think 1 --seed $4 --max-tokens 384 --idle-s 5 $(expa "$2") > $R/bench/$1.log 2>&1
  out $1
}
pi() { echo "$1: Pi requests in this server's log: $(grep -c 'max_tokens 32768' $SL), errors: $(grep -cE 'Traceback|StopIteration' $SL)" | tee -a $G
       grep -vE "^   (state|prompt)( )?:" $SL > $O/$1.server.log; }
echo "$(date +%T) lookup chat/math + sampled tree log: waiting for Pi idle" | tee -a $G
pi_idle || { echo "$(date +%T) Pi not idle for 45 min: nothing run, forge untouched" | tee -a $G; exit 2; }
echo "$(date +%T) Pi idle: start (forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN))" | tee -a $G
for a in off1 on1 on2 off2; do
  if [[ $a == on* ]]; then ex=$EXON; envs=(NGRAM_DRAFT=6); else ex=$EXOFF; envs=(); fi
  launch lkc2_$a "$ex" $envs || continue
  sampled lkc2_$a "$ex" 3,4,5,6,7,8 5100
  pi lkc2_$a
done
$PY $R/../../serve_t/lookup_compare_rm.py $R/bench $O/lookup_compare_rm.json lkc2 | tee -a $G
echo "$(date +%T) restore the served config" | tee -a $G
( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED -u COLD_SCRUB \
    -u TREE_LOG -u TREE_LOG_DIST RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=served_after_lkc2 RUNGS=2k TASKS=1,8 EXPECT="$EXOFF" timeout 900 zsh $T/run_arm.sh ) > $O/restore.driver.log 2>&1
grep -E "drafter ring snapshot:|dec8: decode GPU engine|tree: BANNER|serving OpenAI" $SL | cut -c1-230 | sed "s/^/served banner: /" | tee -a $G
out served_after_lkc2
echo "$(date +%T) done, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $G
