#!/bin/zsh
# 21-22 Nov 2026 (NEXT.md item 2): the cost-aware chain fallback TREE_POLICY=dink vs the served branched policy, and
# the sampled chain (TREE_SAMPLED=0, same build) as the reference. Waits for Pi idle (no 8765 connection for 60 s, last
# request >= 600 s ago) and no reviewer window (only looked at). Fresh launches of the standard launcher, banner first,
# order ch_a, br_a, dk_a, dk_b, br_b, ch_b. Lookup off throughout. Each arm, all sampled (Pi settings: temperature 1.0,
# thinking on): u1 agent set seeds 4100 / 4200 (384 tok), u1 code set seed 4100 (the guardrail), the long Pi-like agent
# session seeds 7100 / 7200 / 7300 (512 tok). Pi traffic or a reviewer window stops the run before the next launch
# (and kills a running bench), then the served config is restored. Ends with the served config restored + smoke.
set -u
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/22nov
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
LOCK=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/22nov_review/windows/runner.lock
TL=/private/tmp/claude-501/-Users-anemll-SourceRelease-GITHUB-ML-playground-anemll-forge/e4e29b3c-8360-4ddd-8f48-0762d3ef8250/scratchpad/timing_main.lock
O=$R/dink; G=$O/ab.log
mkdir -p $O $R/bench
V8K=(v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8 tree=v8_dec8tx lookup=off)
reviewer_busy() { pgrep -f "window[.]zsh" > /dev/null && return 0; [ -e $LOCK ] && [ -n "$(lsof -t $LOCK 2>/dev/null)" ] && return 0; return 1; }
conns() { lsof -nP -iTCP:8765 -sTCP:ESTABLISHED 2>/dev/null | tail -n +2 | wc -l | tr -d ' '; }
last_req_age() {
  local last; last=$(grep -oE "^\[[0-9:]+\] request: prompt" $SL 2>/dev/null | tail -1 | cut -c2-9)
  [ -z "$last" ] && { echo 99999; return; }
  local a=$(( $(date +%s) - $(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) $last" +%s) ))
  [ $a -lt 0 ] && a=99999; echo $a
}
pi_seen() { [ "$(grep -c 'max_tokens 32768' $SL)" != 0 ]; }
out() { grep -E "SUMMARY|ABORT" $R/bench/$1.log | sed "s/^/$1: /" | tee -a $G; }
armcfg() {
  case $1 in
    ch*) EX=($V8K tree=sampled=off tree=policy=branched); envs=(TREE_SAMPLED=0) ;;
    br*) EX=($V8K tree=sampled=on tree=policy=branched); envs=() ;;
    dk*) EX=($V8K tree=sampled=on tree=policy=dink); envs=(TREE_POLICY=dink) ;;
  esac
}
UNSET=(-u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED -u COLD_SCRUB
       -u TREE_LOG -u TREE_LOG_DIST -u LOOKUP_OBSERVE -u LOOKUP_GATE -u DRAFT_PROF -u TREE_POLICY)
launch() {
  local lab=$1; shift
  echo "$(date +%T) launch $lab ($*)" | tee -a $G
  ( cd $T && env $UNSET "$@" RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=${lab}_b RUNGS=2k TASKS=1 EXPECT="${EX[*]}" timeout 900 zsh $T/run_arm.sh ) > $O/$lab.driver.log 2>&1
  grep -E "tree: BANNER|prompt-lookup drafts:" $SL | cut -c1-230 | sed "s/^/$lab banner: /" | tee -a $G
  grep -q "SUMMARY" $R/bench/${lab}_b.log || { echo "$lab: launch failed" | tee -a $G; tail -5 $O/$lab.driver.log | tee -a $G; return 1; }
}
stop=0
bench() {   # label prompts rungs seed maxtok
  local lab=$1 pr=$2 ru=$3 sd=$4 mt=$5
  local args=(--prompts $T/$pr --label $lab --out $R/bench/$lab.json --rungs $ru --idle-s 5 --temp 1.0 --think 1 --seed $sd --max-tokens $mt)
  for e in $EX; do args+=(--expect $e); done
  $PY $T/tree_bench.py "${args[@]}" > $R/bench/$lab.log 2>&1 & local BP=$!
  while kill -0 $BP 2>/dev/null; do
    if pi_seen || reviewer_busy; then kill $BP 2>/dev/null; stop=1; echo "$(date +%T) Pi traffic or reviewer window during $lab: stop" | tee -a $G; break; fi
    sleep 2
  done
  wait $BP 2>/dev/null
  out $lab
}
note() { echo "$1: Pi requests $(grep -c 'max_tokens 32768' $SL), errors $(grep -cE 'Traceback|StopIteration' $SL)" | tee -a $G
         grep -vE "^   (state|prompt)( )?:" $SL > $O/$1.server.log; }
restore() {
  echo "$(date +%T) restore the served config ($1)" | tee -a $G
  EX=($V8K tree=sampled=on tree=policy=branched)
  ( cd $T && env $UNSET RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=served_after_dink RUNGS=2k TASKS=1,8 EXPECT="${EX[*]}" timeout 900 zsh $T/run_arm.sh ) > $O/restore.driver.log 2>&1
  grep -E "drafter ring snapshot:|tree: BANNER|serving OpenAI" $SL | cut -c1-230 | sed "s/^/served banner: /" | tee -a $G
  out served_after_dink
  echo "$(date +%T) done, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $G
}

echo "$(date +%T) dink A/B: waiting for Pi idle 10+ min, no reviewer window, no other timing" | tee $G
quiet=0
for i in $(seq 1 2880); do   # up to 4 h
  if reviewer_busy || [ "$(conns)" != 0 ] || [ -e $TL ]; then quiet=0; else quiet=$((quiet + 5)); fi
  [ $quiet -ge 60 ] && [ $(last_req_age) -ge 600 ] && break
  sleep 5
done
[ $quiet -ge 60 ] && [ $(last_req_age) -ge 600 ] || { echo "$(date +%T) not idle within 4 h: nothing run" | tee -a $G; exit 2; }
echo "dink A/B $(date +%T)" > $TL
for arm in ch_a br_a dk_a dk_b br_b ch_b; do
  if reviewer_busy; then echo "$(date +%T) reviewer window: stop before $arm" | tee -a $G; stop=1; break; fi
  armcfg $arm
  launch dk_$arm "${envs[@]}" || { stop=1; break; }
  if pi_seen; then echo "$(date +%T) Pi traffic at launch of $arm: stop" | tee -a $G; stop=1; break; fi
  for sd in 4100 4200; do [ $stop = 0 ] && bench dk_${arm}_sag$sd u1_code_agent.json agent $sd 384; done
  [ $stop = 0 ] && bench dk_${arm}_sco4100 u1_code_agent.json code 4100 384
  for sd in 7100 7200 7300; do [ $stop = 0 ] && bench dk_${arm}_long$sd long_agent_v1.json long $sd 512; done
  note dk_$arm
  [ $stop = 1 ] && break
done
[ $stop = 0 ] && $PY $R/../../serve_t/dink_compare.py $R/bench $O/compare.json | tee -a $G
if reviewer_busy; then echo "$(date +%T) reviewer window: no restore, forge left to it" | tee -a $G; rm -f $TL; exit 3; fi
restore "$([ $stop = 1 ] && echo stopped || echo end of run)"
rm -f $TL
