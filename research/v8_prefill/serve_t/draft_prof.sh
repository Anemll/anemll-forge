#!/bin/zsh
# 21 Nov 2026 (M6 ANE bot, item 2): the timing-only DRAFT_PROF=1 launch. Only if Pi has been idle 10+ min (last
# request >= 600 s ago, no 8765 connection for 60 s) and no reviewer window. One fresh launch of the standard serving
# launcher with DRAFT_PROF=1 and otherwise the served config (banner first), then the long Pi-like agent session greedy
# and sampled seed 7100 (thinking on, 512 tokens): per request the server logs "drafter prof: body / head / ctx /
# propose host" next to the cycle split. Pi traffic during the run (a max_tokens 32768 request) aborts it at once,
# then the served config is restored. Always ends with the served config restored + smoke. zsh arrays.
set -u
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/21nov
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
LOCK=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/22nov_review/windows/runner.lock
O=$R/draft_prof; G=$O/run.log
mkdir -p $O $R/bench
BASE=(v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8 tree=policy=branched tree=v8_dec8tx tree=sampled=on lookup=off)
reviewer_busy() { pgrep -f "window[.]zsh" > /dev/null && return 0; [ -e $LOCK ] && [ -n "$(lsof -t $LOCK 2>/dev/null)" ] && return 0; return 1; }
last_req_age() {
  local last; last=$(grep -oE "^\[[0-9:]+\] request: prompt" $SL 2>/dev/null | tail -1 | cut -c2-9)
  [ -z "$last" ] && { echo 99999; return; }
  local a=$(( $(date +%s) - $(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) $last" +%s) ))
  [ $a -lt 0 ] && a=99999; echo $a
}
conns() { lsof -nP -iTCP:8765 -sTCP:ESTABLISHED 2>/dev/null | tail -n +2 | wc -l | tr -d ' '; }
restore() {
  echo "$(date +%T) restore the served config ($1)" | tee -a $G
  local EX=($BASE)
  ( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED -u COLD_SCRUB \
      -u TREE_LOG -u TREE_LOG_DIST -u LOOKUP_OBSERVE -u LOOKUP_GATE -u DRAFT_PROF RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=served_after_dprof RUNGS=2k TASKS=1,8 \
      EXPECT="${EX[*]}" timeout 900 zsh $T/run_arm.sh ) > $O/restore.driver.log 2>&1
  grep -E "drafter ring snapshot:|tree: BANNER|DRAFT_PROF|serving OpenAI" $SL | cut -c1-230 | sed "s/^/served banner: /" | tee -a $G
  grep -E "SUMMARY|ABORT" $R/bench/served_after_dprof.log | sed "s/^/served_after_dprof: /" | tee -a $G
  echo "$(date +%T) done, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $G
}

echo "$(date +%T) DRAFT_PROF launch: waiting for Pi idle 10+ min and no reviewer window" | tee $G
quiet=0
for i in $(seq 1 1440); do   # up to 2 h
  if reviewer_busy || [ "$(conns)" != 0 ] || [ -e /private/tmp/claude-501/-Users-anemll-SourceRelease-GITHUB-ML-playground-anemll-forge/e4e29b3c-8360-4ddd-8f48-0762d3ef8250/scratchpad/timing_gdn.lock ]; then quiet=0; else quiet=$((quiet + 5)); fi   # also: no GDN kernel timing running
  [ $quiet -ge 60 ] && [ $(last_req_age) -ge 600 ] && break
  sleep 5
done
[ $quiet -ge 60 ] && [ $(last_req_age) -ge 600 ] || { echo "$(date +%T) Pi not idle 10+ min within 2 h: nothing run" | tee -a $G; exit 2; }
echo "$(date +%T) Pi idle (last request $(last_req_age) s ago): launch" | tee -a $G

EX=($BASE)
( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED -u COLD_SCRUB \
    -u TREE_LOG -u TREE_LOG_DIST -u LOOKUP_OBSERVE -u LOOKUP_GATE DRAFT_PROF=1 RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=dprof_b RUNGS=2k TASKS=1 \
    EXPECT="${EX[*]}" timeout 900 zsh $T/run_arm.sh ) > $O/launch.driver.log 2>&1
grep -E "tree: BANNER|DRAFT_PROF|prompt-lookup drafts:" $SL | cut -c1-230 | sed "s/^/dprof banner: /" | tee -a $G
grep -q "DRAFT_PROF timing armed" $SL || { echo "$(date +%T) DRAFT_PROF not armed: abort" | tee -a $G; restore "not armed"; exit 1; }
grep -q "SUMMARY" $R/bench/dprof_b.log || { echo "$(date +%T) launch smoke failed" | tee -a $G; restore "launch failed"; exit 1; }

aborted=0
for lab in dprof_g:0:- dprof_s7100:1.0:7100; do
  l=${lab%%:*}; rest=${lab#*:}; te=${rest%%:*}; sd=${rest#*:}
  args=(--prompts $T/long_agent_v1.json --label $l --out $R/bench/$l.json --rungs long --idle-s 5 --temp $te --think 1 --max-tokens 512)
  [ "$sd" != "-" ] && args+=(--seed $sd)
  for e in $EX; do args+=(--expect $e); done
  $PY $T/tree_bench.py "${args[@]}" > $R/bench/$l.log 2>&1 & BP=$!
  while kill -0 $BP 2>/dev/null; do
    if [ "$(grep -c 'max_tokens 32768' $SL)" != 0 ] || reviewer_busy; then
      kill $BP 2>/dev/null; aborted=1
      echo "$(date +%T) Pi traffic or reviewer window during $l: aborted" | tee -a $G; break
    fi
    sleep 2
  done
  wait $BP 2>/dev/null
  grep -E "SUMMARY|ABORT" $R/bench/$l.log | sed "s/^/$l: /" | tee -a $G
  [ $aborted = 1 ] && break
done
echo "dprof: Pi requests $(grep -c 'max_tokens 32768' $SL), errors $(grep -cE 'Traceback|StopIteration' $SL), drafter prof lines $(grep -c 'drafter prof:' $SL)" | tee -a $G
grep -vE "^   (state|prompt)( )?:" $SL > $O/dprof.server.log
restore "$([ $aborted = 1 ] && echo aborted || echo end of run)"
