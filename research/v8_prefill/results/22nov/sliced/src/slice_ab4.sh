#!/bin/zsh
# 22 Nov 2026 (NEXT.md item 1 step B): sliced decode entries vs the served path on the long Pi-like agent session.
#   sv   the served config (v8_server_dec8_qk.sh as is: v8_dec8tx, GPU tables w_dec8tx)
#   s2   the temporary build v8_dec8ts + its GPU tables, RUNG_SLICE=24,48 (pipe/rung_slice.py; research flag)
#   s4   the same build, RUNG_SLICE=24,40,48,56
#   kc   the same build, RUNG_SLICE=24,40,48,56 RUNG_SLICE_CHECK=1: parity only (every sliced call also runs the bigger
#        entry and keeps its outputs), not timed
# First kc on the greedy long session: the parity gate (no error, every argmax equal, max KL < 1e-6) or no timed arm.
# Block 1 (the 24K+48K set, first): sv_a, s2_a, s2_b, sv_b. Block 2 (4-entry set) only if it can end by DEADLINE
# (default 07:00): s4_a, sv_c, s4_b. Each timed arm: the long session greedy (thinking on) + sampled seeds
# 7100 / 7200 / 7300 (temperature 1.0, thinking on), 512 tok. Waits for Pi idle (no 8765 connection for 60 s, last
# request >= 600 s ago), no reviewer window (only looked at), no other timing lock. Fresh launches of the standard
# launcher, banner checked (build, tree, slice). Pi traffic or a reviewer window stops the run (and kills a running
# bench), then the served config is restored. Ends with the served config restored + smoke (checksum 29aef2ec9db18a3f).
set -u
# 22 Nov 09:20 copy for block 2 only (Anemll): the 4-entry set ABBA vs served, gate kc2 first, restore served_after_slice4.
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/22nov
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
LOCK=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/22nov_review/windows/runner.lock
TL=/private/tmp/claude-501/-Users-anemll-SourceRelease-GITHUB-ML-playground-anemll-forge/e4e29b3c-8360-4ddd-8f48-0762d3ef8250/scratchpad/timing_main.lock
TB=/Volumes/SSD4TB/anemll-forge-research/v8_dec8ts; TG=/Volumes/SSD4TB/anemll-forge-research/v8_dec8ts_w/manifest_best.json
SMOKE=29aef2ec9db18a3f; DEADLINE=${DEADLINE:-07:00}
O=$R/sliced/block2; G=$O/ab.log
mkdir -p $O $R/bench
V8K=(v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8
     tree=policy=branched tree=sampled=on lookup=off)
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
    sv*) EX=($V8K tree=v8_dec8tx "slice=off (no slice banner)"); envs=() ;;
    s2*) EX=($V8K tree=v8_dec8ts "slice=sliced decode entries [24, 48]K"); envs=(BUILD=$TB GM=$TG RUNG_SLICE=24,48) ;;
    s4*) EX=($V8K tree=v8_dec8ts "slice=sliced decode entries [24, 40, 48, 56]K"); envs=(BUILD=$TB GM=$TG RUNG_SLICE=24,40,48,56) ;;
    kc*) EX=($V8K tree=v8_dec8ts "slice=check on"); envs=(BUILD=$TB GM=$TG RUNG_SLICE=24,40,48,56 RUNG_SLICE_CHECK=1) ;;
  esac
}
UNSET=(-u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED -u COLD_SCRUB
       -u TREE_LOG -u TREE_LOG_DIST -u LOOKUP_OBSERVE -u LOOKUP_GATE -u DRAFT_PROF -u TREE_POLICY -u RUNG_SLICE -u RUNG_SLICE_CHECK
       -u TREE_SLICED)
launch() {
  local lab=$1; shift
  echo "$(date +%T) launch $lab ($*)" | tee -a $G
  local ex1=(${EX:#slice=*})   # run_arm's own EXPECT is word-split: the slice banner is checked below and by every bench
  ( cd $T && env $UNSET "$@" RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=${lab}_b RUNGS=2k TASKS=1 EXPECT="${ex1[*]}" timeout 900 zsh $T/run_arm.sh ) > $O/$lab.driver.log 2>&1
  grep -E "tree: BANNER|slice: BANNER|prompt-lookup drafts:|serving OpenAI" $SL | cut -c1-250 | sed "s/^/$lab banner: /" | tee -a $G
  grep -q "SUMMARY" $R/bench/${lab}_b.log || { echo "$lab: launch failed" | tee -a $G; tail -5 $O/$lab.driver.log | tee -a $G; return 1; }
}
stop=0
bench() {   # label set rungs temp seed
  local lab=$1 tp=$2 sd=$3
  local args=(--prompts $T/long_agent_v1.json --label $lab --out $R/bench/$lab.json --rungs long --idle-s 5 --temp $tp --think 1 --seed $sd --max-tokens 512)
  for e in $EX; do args+=(--expect $e); done
  $PY $T/tree_bench.py "${args[@]}" > $R/bench/$lab.log 2>&1 & local BP=$!
  while kill -0 $BP 2>/dev/null; do
    if pi_seen || reviewer_busy; then kill $BP 2>/dev/null; stop=1; echo "$(date +%T) Pi traffic or reviewer window during $lab: stop" | tee -a $G; break; fi
    sleep 2
  done
  wait $BP 2>/dev/null
  out $lab
  grep -q "SUMMARY" $R/bench/$lab.log || { echo "$(date +%T) $lab: no SUMMARY (banner abort or failure): stop" | tee -a $G; stop=1; }
}
note() { echo "$1: Pi requests $(grep -c 'max_tokens 32768' $SL), errors $(grep -cE 'Traceback|StopIteration|AssertionError' $SL)" | tee -a $G
         grep -vE "^   (state|prompt)( )?:" $SL > $O/$1.server.log; }
arm() {   # one arm: launch + greedy + 3 seeds
  local a=$1
  if reviewer_busy; then echo "$(date +%T) reviewer window: stop before $a" | tee -a $G; stop=1; return; fi
  armcfg $a
  launch sl_$a "${envs[@]}" || { stop=1; return; }
  if pi_seen; then echo "$(date +%T) Pi traffic at launch of $a: stop" | tee -a $G; stop=1; return; fi
  [ $stop = 0 ] && bench sl_${a}_g 0 -1
  for sd in 7100 7200 7300; do [ $stop = 0 ] && bench sl_${a}_s$sd 1.0 $sd; done
  note sl_$a
}
restore() {
  echo "$(date +%T) restore the served config ($1)" | tee -a $G
  EX=($V8K tree=v8_dec8tx)
  ( cd $T && env $UNSET RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=served_after_slice4 RUNGS=2k TASKS=1,8 EXPECT="${EX[*]}" timeout 900 zsh $T/run_arm.sh ) > $O/restore.driver.log 2>&1
  grep -E "drafter ring snapshot:|tree: BANNER|slice: BANNER|serving OpenAI" $SL | cut -c1-230 | sed "s/^/served banner: /" | tee -a $G
  out served_after_slice4
  if grep -q "output checksum $SMOKE" $R/bench/served_after_slice4.log; then echo "smoke checksum $SMOKE: MATCH" | tee -a $G
  else echo "smoke checksum: MISMATCH (expected $SMOKE)" | tee -a $G; fi
  echo "$(date +%T) done, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $G
}
compare_now() { CHECK_ARM=kc2 $PY $R/../../serve_t/slice_compare.py $R/bench $O $O/compare.json | tee -a $G; }

[ -f $TG ] && [ -f $TB/manifest.json ] || { echo "temporary build or its GPU tables missing ($TB, $TG)"; exit 1; }
echo "$(date +%T) slice A/B block 2 (24/40/48/56K, ABBA sv_c s4_a s4_b sv_d): waiting for Pi idle 10+ min, no reviewer window, no other timing" | tee $G
quiet=0
for i in $(seq 1 2880); do   # up to 4 h
  if reviewer_busy || [ "$(conns)" != 0 ] || [ -e $TL ]; then quiet=0; else quiet=$((quiet + 5)); fi
  [ $quiet -ge 60 ] && [ $(last_req_age) -ge 600 ] && break
  sleep 5
done
[ $quiet -ge 60 ] && [ $(last_req_age) -ge 600 ] || { echo "$(date +%T) not idle within 4 h: nothing run" | tee -a $G; exit 2; }
echo "slice A/B $(date +%T)" > $TL
# parity first (not timed): RUNG_SLICE_CHECK on all four widths over the greedy long session; any error, any argmax
# change or KL >= 1e-6 stops the run before the timed arms
armcfg kc
if launch sl_kc2 "${envs[@]}" && ! pi_seen; then
  bench sl_kc2_g 0 -1; note sl_kc2; compare_now
  $PY - $O/compare.json $O/sl_kc2.server.log > $O/gate.txt 2>&1 <<'Y'
import json, re, sys
c = json.load(open(sys.argv[1])).get("check", {})
err = len(re.findall(r"Traceback|StopIteration|AssertionError", open(sys.argv[2], errors="replace").read()))
ok = bool(c) and c["sliced_calls"] > 0 and err == 0 and c["argmax_equal_rows"] == c["rows"] and c["max_kl"] < 1e-6
print(f"parity gate: {'PASS' if ok else 'FAIL'} ({c or 'no check lines'}; errors {err})")
sys.exit(0 if ok else 1)
Y
  grc=$?; tee -a $G < $O/gate.txt; [ $grc = 0 ] || stop=1
else
  stop=1
fi
[ $stop = 1 ] && echo "$(date +%T) parity check failed or interrupted: no timed arms" | tee -a $G
for a in sv_c s4_a s4_b sv_d; do [ $stop = 1 ] && break; arm $a; done
[ $stop = 0 ] && compare_now
if reviewer_busy; then echo "$(date +%T) reviewer window: no restore, forge left to it" | tee -a $G; rm -f $TL; exit 3; fi
restore "$([ $stop = 1 ] && echo stopped || echo end of run)"
rm -f $TL
