#!/bin/zsh
# 21 Nov 2026 (M6 ANE bot: context-aware lookup gate). Runs only while Pi is idle. Fresh launches of the standard
# serving launcher, banner first, in both orders: off_a, ctx_a, ctx_b, off_b (ctx = NGRAM_DRAFT=6
# LOOKUP_GATE=ctx:32768:48:1.25: below 32K of context only a match >= 48 that also passes yield:1.25, from 32K
# yield:1.25). Each arm: u1 code + agent greedy (320 tok) and sampled (seed 4100), realistic math + chat (tasks 3-8)
# at 2k / 8k / 32k greedy and sampled (seed 5100), the long Pi-like agent session (19 turns, 8.6K -> 55.5K, sampled
# seed 7100, 512 tok). Sampled = Pi settings. Ends with the served config restored + smoke. zsh arrays throughout.
set -u
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/21nov
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
O=$R/lookup_gate; G=$O/ctx_gate.log
mkdir -p $O $R/bench
BASE=(v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8 tree=policy=branched tree=v8_dec8tx tree=sampled=on)
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
bench() {   # label prompts rungs tasks(or -) temp(0 = greedy) seed maxtok expect...
  local lab=$1 pr=$2 ru=$3 ta=$4 te=$5 sd=$6 mt=$7; shift 7
  local args=(--prompts $T/$pr --label $lab --out $R/bench/$lab.json --rungs $ru --idle-s 5)
  [ "$ta" != "-" ] && args+=(--tasks $ta)
  [ "$te" != "0" ] && args+=(--temp $te --think 1 --seed $sd --max-tokens $mt)
  for e in "$@"; do args+=(--expect "$e"); done
  $PY $T/tree_bench.py "${args[@]}" > $R/bench/$lab.log 2>&1
  out $lab
}
launch() {   # label env-words... ; the expect list in the global EX
  local lab=$1; shift
  echo "$(date +%T) launch $lab ($*)" | tee -a $G
  ( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED -u COLD_SCRUB \
      -u TREE_LOG -u TREE_LOG_DIST -u LOOKUP_OBSERVE -u LOOKUP_GATE "$@" RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=${lab}_b RUNGS=2k TASKS=1 \
      EXPECT="${EX[*]}" timeout 900 zsh $T/run_arm.sh ) > $O/$lab.driver.log 2>&1
  grep -E "tree: BANNER|prompt-lookup drafts:" $SL | cut -c1-230 | sed "s/^/$lab banner: /" | tee -a $G
  grep -q "SUMMARY" $R/bench/${lab}_b.log || { echo "$lab: launch failed" | tee -a $G; tail -5 $O/$lab.driver.log | tee -a $G; return 1; }
}
echo "$(date +%T) lookup gate A/B: waiting for Pi idle" | tee -a $G
pi_idle || { echo "$(date +%T) Pi not idle for 45 min: nothing run" | tee -a $G; exit 2; }
armcfg() {
  case $1 in
    off*)  EX=($BASE lookup=off); envs=() ;;
    len16) EX=($BASE tree=NGRAM_DRAFT= tree=lookup_gate=len:16); envs=(NGRAM_DRAFT=6 LOOKUP_GATE=len:16) ;;
    yield) EX=($BASE tree=NGRAM_DRAFT= tree=lookup_gate=yield:1.25); envs=(NGRAM_DRAFT=6 LOOKUP_GATE=yield:1.25) ;;
    ctx*)  EX=($BASE tree=NGRAM_DRAFT= tree=lookup_gate=ctx:32768:48:1.25); envs=(NGRAM_DRAFT=6 LOOKUP_GATE=ctx:32768:48:1.25) ;;
  esac
}
note() { echo "$1: Pi requests $(grep -c 'max_tokens 32768' $SL), errors $(grep -cE 'Traceback|StopIteration' $SL), lookup gated $(grep -oE 'lookup gated [0-9]+' $SL | awk '{s+=$3} END {print s+0}')" | tee -a $G
         grep -vE "^   (state|prompt)( )?:" $SL > $O/$1.server.log; }
for arm in off_a ctx_a ctx_b off_b; do
  armcfg $arm
  launch lcg_$arm "${envs[@]}" || continue
  bench lcg_${arm}_gca u1_code_agent.json code,agent - 0 0 0 "${EX[@]}"
  bench lcg_${arm}_sca u1_code_agent.json code,agent - 1.0 4100 384 "${EX[@]}"
  bench lcg_${arm}_grm realistic_v1.json 2k,8k,32k 3,4,5,6,7,8 0 0 0 "${EX[@]}"
  bench lcg_${arm}_srm realistic_v1.json 2k,8k,32k 3,4,5,6,7,8 1.0 5100 384 "${EX[@]}"
  bench lcg_${arm}_long long_agent_v1.json long - 1.0 7100 512 "${EX[@]}"
  note lcg_$arm
done
$PY $R/../../serve_t/ctx_gate_compare.py $R/bench $O/ctx_gate_compare.json | tee -a $G
echo "$(date +%T) restore the served config" | tee -a $G
EX=($BASE lookup=off)
( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED -u COLD_SCRUB \
    -u TREE_LOG -u TREE_LOG_DIST -u LOOKUP_OBSERVE -u LOOKUP_GATE RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=served_after_lcg RUNGS=2k TASKS=1,8 \
    EXPECT="${EX[*]}" timeout 900 zsh $T/run_arm.sh ) > $O/restore_ctx.driver.log 2>&1
grep -E "drafter ring snapshot:|tree: BANNER|serving OpenAI" $SL | cut -c1-230 | sed "s/^/served banner: /" | tee -a $G
out served_after_lcg
echo "$(date +%T) done, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $G
