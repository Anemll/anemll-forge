#!/bin/zsh
# 21 Nov 2026 (M6 ANE bot: reasoning-span lookup gate, live). Waits until the outside reviewer's bench window is over
# (no lookup_adaptation/window.zsh process, nobody holding results/22nov_review/windows/runner.lock; both only looked
# at, never touched) and Pi is idle. Fresh launches of the standard serving launcher, banner first, ABBA over three
# configs: off_a, ctx_a, thk_a, thk_b, ctx_b, off_b.
#   off  served (lookup off)
#   ctx  NGRAM_DRAFT=6 LOOKUP_GATE=ctx:32768:48:1.25
#   thk  NGRAM_DRAFT=6 LOOKUP_GATE=think:24:yield:1.25 (inside the reasoning span only matches >= 24, and yield:1.25)
# Each arm: the long Pi-like agent session (19 turns, 8.6K -> 55.5K, 512 tok, thinking on): greedy once, sampled
# (Pi settings, temperature 1.0) with seeds 7100 / 7200 / 7300. If the reviewer's window starts meanwhile, the run
# stops before the next launch and leaves forge to that window. Ends with the served config restored + smoke.
# zsh arrays throughout.
set -u
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/21nov
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
LOCK=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/22nov_review/windows/runner.lock
O=$R/think_gate; G=$O/ab.log
mkdir -p $O $R/bench
BASE=(v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8 tree=policy=branched tree=v8_dec8tx tree=sampled=on)
out() { grep -E "SUMMARY|ABORT" $R/bench/$1.log | sed "s/^/$1: /" | tee -a $G; }
reviewer_busy() {   # 0 = the reviewer's window runs or holds its lock
  pgrep -f "window[.]zsh" > /dev/null && return 0
  [ -e $LOCK ] && [ -n "$(lsof -t $LOCK 2>/dev/null)" ] && return 0
  return 1
}
served_default() { grep -E "tree: BANNER" $SL | tail -1 | grep -q "| lookup off"; }
pi_idle() {
  local i quiet=0 last now age
  for i in $(seq 1 540); do
    if [ "$(lsof -nP -iTCP:8765 -sTCP:ESTABLISHED 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')" = 0 ]; then quiet=$((quiet + 5)); else quiet=0; fi
    last=$(grep -oE "^\[[0-9:]+\] request: prompt" $SL 2>/dev/null | tail -1 | cut -c2-9)
    now=$(date +%s); age=999
    [ -n "$last" ] && age=$(( now - $(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) $last" +%s) ))
    reviewer_busy && quiet=0
    [ $quiet -ge 60 ] && [ $age -ge 120 ] && return 0
    sleep 5
  done
  return 1
}
bench() {   # label temp seed expect...   (the long session, thinking on, 512 tok)
  local lab=$1 te=$2 sd=$3; shift 3
  local args=(--prompts $T/long_agent_v1.json --label $lab --out $R/bench/$lab.json --rungs long --idle-s 5
              --temp $te --think 1 --max-tokens 512)
  [ "$sd" != "-" ] && args+=(--seed $sd)
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
armcfg() {
  case $1 in
    off*) EX=($BASE lookup=off); envs=() ;;
    ctx*) EX=($BASE tree=NGRAM_DRAFT= tree=lookup_gate=ctx:32768:48:1.25); envs=(NGRAM_DRAFT=6 LOOKUP_GATE=ctx:32768:48:1.25) ;;
    thk*) EX=($BASE tree=NGRAM_DRAFT= tree=lookup_gate=think:24:yield:1.25); envs=(NGRAM_DRAFT=6 LOOKUP_GATE=think:24:yield:1.25) ;;
  esac
}
note() { echo "$1: Pi requests $(grep -c 'max_tokens 32768' $SL), errors $(grep -cE 'Traceback|StopIteration' $SL), lookup gated $(grep -oE 'lookup gated [0-9]+' $SL | awk '{s+=$3} END {print s+0}'), in reasoning lookup $(grep -oE 'in reasoning: lookup [0-9]+' $SL | awk '{s+=$4} END {print s+0}') gated $(grep -oE 'in reasoning: lookup [0-9]+, gated [0-9]+' $SL | awk '{s+=$6} END {print s+0}')" | tee -a $G
         grep -vE "^   (state|prompt)( )?:" $SL > $O/$1.server.log; }
echo "$(date +%T) think gate A/B: waiting for the reviewer's window to end and Pi idle" | tee -a $G
n=0
while reviewer_busy || ! served_default; do
  n=$((n + 1)); [ $n -gt 480 ] && { echo "$(date +%T) reviewer window / served default not clear after 4 h: nothing run" | tee -a $G; exit 2; }
  sleep 30
done
echo "$(date +%T) reviewer window over (served default banner up)" | tee -a $G
pi_idle || { echo "$(date +%T) Pi not idle for 45 min: nothing run" | tee -a $G; exit 2; }

for arm in off_a ctx_a thk_a thk_b ctx_b off_b; do
  if reviewer_busy; then echo "$(date +%T) the reviewer's window started: stop here, forge left to it" | tee -a $G; exit 3; fi
  armcfg $arm
  launch tgl_$arm "${envs[@]}" || continue

  bench tgl_${arm}_g 0 - "${EX[@]}"
  for sd in 7100 7200 7300; do bench tgl_${arm}_s$sd 1.0 $sd "${EX[@]}"; done
  note tgl_$arm
done
$PY $R/../../serve_t/think_gate_compare.py $R/bench $O/compare.json | tee -a $G
if reviewer_busy; then echo "$(date +%T) the reviewer's window started: no restore, forge left to it" | tee -a $G; exit 3; fi
echo "$(date +%T) restore the served config" | tee -a $G
EX=($BASE lookup=off)
( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED -u COLD_SCRUB \
    -u TREE_LOG -u TREE_LOG_DIST -u LOOKUP_OBSERVE -u LOOKUP_GATE RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=served_after_tgl RUNGS=2k TASKS=1,8 \
    EXPECT="${EX[*]}" timeout 900 zsh $T/run_arm.sh ) > $O/restore.driver.log 2>&1
grep -E "drafter ring snapshot:|tree: BANNER|serving OpenAI" $SL | cut -c1-230 | sed "s/^/served banner: /" | tee -a $G
out served_after_tgl
echo "$(date +%T) done, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $G
