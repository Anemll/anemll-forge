#!/bin/zsh
# 22 Nov 2026 (Anemll: "1- and measure, also update pacing"): forge memory across the 32K -> 64K entry switch on the new
# (v8_dec8ts, sliced) and old (v8_dec8tx, SERVED_SLICE=0) builds, fresh launches, long session greedy (8.6K-55.5K),
# footprint + swap sampled every 2 s; then stream pacing: chunk timing off (new build) vs on (STREAM_PACE=1). Ends with
# forge on the served config + STREAM_PACE=1 (env only; launcher default unchanged). Pi traffic: stop, served restored.
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
RS=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/serve_t
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/22nov; O=$R/mem; mkdir -p $O $R/bench
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log; SMOKE=29aef2ec9db18a3f
LOCK=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/22nov_review/windows/runner.lock
G=$O/mem.log
V8K=(v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8 tree=policy=branched tree=sampled=on lookup=off)
UNSET=(-u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u SERVED_SLICE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED
       -u COLD_SCRUB -u TREE_LOG -u TREE_LOG_DIST -u LOOKUP_OBSERVE -u LOOKUP_GATE -u DRAFT_PROF -u TREE_POLICY -u RUNG_SLICE
       -u RUNG_SLICE_CHECK -u TREE_SLICED -u STREAM_PACE -u FORGE_WD -u FORGE_FP_MAX_MB)
reviewer_busy() { pgrep -f "window[.]zsh" > /dev/null && return 0; [ -e $LOCK ] && [ -n "$(lsof -t $LOCK 2>/dev/null)" ] && return 0; return 1; }
conns() { lsof -nP -iTCP:8765 -sTCP:ESTABLISHED 2>/dev/null | tail -n +2 | wc -l | tr -d ' '; }
last_req_age() {
  local last; last=$(grep -oE "^\[[0-9:]+\] request: prompt" $SL 2>/dev/null | tail -1 | cut -c2-9)
  [ -z "$last" ] && { echo 99999; return; }
  local a=$(( $(date +%s) - $(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) $last" +%s) ))
  [ $a -lt 0 ] && a=99999; echo $a
}
pi_seen() { [ "$(grep -c 'max_tokens 32768' $SL)" != 0 ]; }
stop=0
launch() {   # label tree-build [env...]
  local lab=$1 tb=$2; shift 2
  local EX=($V8K tree=$tb)
  echo "$(date +%T) launch $lab ($*)" | tee -a $G
  ( cd $T && env $UNSET "$@" RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=$lab RUNGS=2k TASKS=1,8 EXPECT="${EX[*]}" timeout 900 zsh $T/run_arm.sh ) > $O/$lab.driver.log 2>&1
  local rc=$?
  grep -E "tree: BANNER|slice: BANNER|stream pace: BANNER|serving OpenAI" $SL | cut -c1-220 | sed "s/^/$lab banner: /" | tee -a $G
  grep -E "SUMMARY|ABORT" $R/bench/$lab.log | sed "s/^/$lab: /" | tee -a $G
  [ $rc = 0 ] && grep -q "output checksum $SMOKE" $R/bench/$lab.log || { echo "$(date +%T) $lab: launch or smoke failed (rc $rc)" | tee -a $G; return 1; }
  echo "$(date +%T) $lab: smoke MATCH, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $G
}
memrun() {   # label tree-build
  local lab=$1 tb=$2
  local args=(--prompts $T/long_agent_v1.json --label $lab --out $R/bench/$lab.json --rungs long --idle-s 5 --temp 0 --think 1 --seed -1 --max-tokens 512)
  for e in $V8K tree=$tb; do args+=(--expect $e); done
  $RS/mem_sampler.sh $O/$lab.csv & local MP=$!
  $PY $T/tree_bench.py "${args[@]}" > $R/bench/$lab.log 2>&1 & local BP=$!
  while kill -0 $BP 2>/dev/null; do
    if pi_seen || reviewer_busy; then kill $BP 2>/dev/null; stop=1; echo "$(date +%T) Pi traffic or reviewer window during $lab: stop" | tee -a $G; break; fi
    sleep 2
  done
  wait $BP 2>/dev/null; sleep 4; kill $MP 2>/dev/null
  grep -E "SUMMARY|ABORT" $R/bench/$lab.log | sed "s/^/$lab: /" | tee -a $G
  grep -E "KV segments registered|\] prompt [0-9]+ \(cached" $SL | cut -c1-90 > $O/$lab.switch.txt
  grep -vE "^   (state|prompt)( )?:" $SL > $O/$lab.server.log
  $PY - $O/$lab.csv <<'Y' | tee -a $G
import csv, sys
r = list(csv.DictReader(open(sys.argv[1])))
f = lambda k: [int(x[k]) for x in r]
print(f"{sys.argv[1].rsplit('/',1)[1]}: {len(r)} samples; footprint {f('fp_mb')[0]} -> peak {max(f('fp_mb'))} MB "
      f"(IOSurface peak {max(f('iosurface_mb'))}, GPU peak {max(f('ioaccel_mb'))}); swap {f('swap_mb')[0]} -> peak {max(f('swap_mb'))} MB; "
      f"free min {min(f('free_mb'))} MB")
Y
}
echo "$(date +%T) mem + pace: waiting for Pi idle 10+ min, no reviewer window" | tee $G
quiet=0; t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 14400 ]; do
  if reviewer_busy || [ "$(conns)" != 0 ]; then quiet=0; else quiet=$((quiet + 5)); fi
  [ $quiet -ge 60 ] && [ $(last_req_age) -ge 600 ] && break
  sleep 5
done
[ $quiet -ge 60 ] && [ $(last_req_age) -ge 600 ] || { echo "$(date +%T) not idle within 4 h: nothing run" | tee -a $G; exit 2; }
# 1. new build (served default)
if launch mem_new_b v8_dec8ts && ! pi_seen; then
  memrun mem_new_g v8_dec8ts
  [ $stop = 0 ] && $PY $RS/chunk_timing.py pace_off $O/pace_off.json | tee -a $G
else stop=1; fi
# 2. old build
if [ $stop = 0 ]; then
  if launch mem_old_b v8_dec8tx SERVED_SLICE=0 && ! pi_seen; then memrun mem_old_g v8_dec8tx; else stop=1; fi
fi
# 3. served + STREAM_PACE=1 (or plain served after a stop)
if [ $stop = 0 ] && launch served_pace_on v8_dec8ts STREAM_PACE=1 && grep "stream pace: BANNER" $SL | tail -1 | grep -q "BANNER on"; then
  $PY $RS/chunk_timing.py pace_on $O/pace_on.json | tee -a $G
  echo "$(date +%T) done: forge on the served config + STREAM_PACE=1 (env only), forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $G
else
  echo "$(date +%T) stopped or pace launch failed: restore the plain served config" | tee -a $G
  launch served_after_mem v8_dec8ts || echo "$(date +%T) RESTORE FAILED" | tee -a $G
fi
echo "$(date +%T) mem_pace.sh exit" | tee -a $G
