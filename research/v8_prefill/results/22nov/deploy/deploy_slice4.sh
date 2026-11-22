#!/bin/zsh
# 22 Nov deploy (Anemll "ok, deploy"): 4-entry sliced decode entries served. Waits for Pi idle (same rule as slice_ab4),
# switches the launcher defaults, fresh launch + banner check + smoke TASKS=1,8. On any failure: launcher reverted, old
# served config relaunched + smoke.
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; L=$P/v8_server_dec8_qk.sh; BK=$P/v8_server_dec8_qk.pre_slice4.sh
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/22nov; O=$R/deploy; mkdir -p $O
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log; SMOKE=29aef2ec9db18a3f
LOCK=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/22nov_review/windows/runner.lock
G=$O/deploy.log
V8K=(v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8 tree=policy=branched tree=sampled=on lookup=off)
UNSET=(-u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u SERVED_SLICE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED -u COLD_SCRUB -u RUNG_SLICE -u RUNG_SLICE_CHECK)
reviewer_busy() { pgrep -f "window[.]zsh" > /dev/null && return 0; [ -e $LOCK ] && [ -n "$(lsof -t $LOCK 2>/dev/null)" ] && return 0; return 1; }
conns() { lsof -nP -iTCP:8765 -sTCP:ESTABLISHED 2>/dev/null | tail -n +2 | wc -l | tr -d ' '; }
last_req_age() {
  local last; last=$(grep -oE "^\[[0-9:]+\] request: prompt" $SL 2>/dev/null | tail -1 | cut -c2-9)
  [ -z "$last" ] && { echo 99999; return; }
  local a=$(( $(date +%s) - $(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) $last" +%s) ))
  [ $a -lt 0 ] && a=99999; echo $a
}
smoke() {  # $1 arm label, $2 tree build, $3 slice expect or empty
  local EX=($V8K tree=$2)
  ( cd $T && env $UNSET RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=$1 RUNGS=2k TASKS=1,8 EXPECT="${EX[*]}" timeout 900 zsh $T/run_arm.sh ) > $O/$1.driver.log 2>&1
  local rc=$?
  grep -E "drafter ring snapshot:|tree: BANNER|slice: BANNER|serving OpenAI" $SL | tail -4 | cut -c1-220 | sed "s/^/$1 banner: /" | tee -a $G
  grep -E "SUMMARY|ABORT" $R/bench/$1.log | sed "s/^/$1: /" | tee -a $G
  [ $rc = 0 ] || { echo "$(date +%T) $1: run_arm rc $rc" | tee -a $G; return 1; }
  grep -q "output checksum $SMOKE" $R/bench/$1.log || { echo "$(date +%T) $1: smoke MISMATCH" | tee -a $G; return 1; }
  if [ -n "$3" ]; then grep "slice: BANNER" $SL | tail -1 | grep -qF "$3" || { echo "$(date +%T) $1: slice banner missing" | tee -a $G; return 1; }; fi
  echo "$(date +%T) $1: smoke $SMOKE MATCH, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $G
}
echo "$(date +%T) deploy 4-entry: waiting for Pi idle 10+ min, no reviewer window" | tee $G
quiet=0; t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 14400 ]; do
  if reviewer_busy || [ "$(conns)" != 0 ]; then quiet=0; else quiet=$((quiet + 5)); fi
  [ $quiet -ge 60 ] && [ $(last_req_age) -ge 600 ] && break
  sleep 5
done
[ $quiet -ge 60 ] && [ $(last_req_age) -ge 600 ] || { echo "$(date +%T) not idle within 4 h: nothing changed" | tee -a $G; exit 2; }
echo "$(date +%T) idle: switching launcher" | tee -a $G
cp -p $L $BK
python3 - $L <<'Y' || { echo "launcher edit failed" | tee -a $G; exit 1; }
import sys
p=sys.argv[1]; s=open(p).read()
old='export BUILD=${BUILD:-/Volumes/SSD4TB/anemll-forge-research/v8_dec8tx} GM=${GM:-$P/w_dec8tx/manifest_best.json} TREE=${TREE-1} TREE_SAMPLED=${TREE_SAMPLED:-1}\n'
new=('# 22 Nov 2026 (Anemll, deploy): sliced decode entries served. Build v8_dec8ts (= v8_dec8tx + sliced chain/tree\n'
'# entries, same weights, 81 functions per chunk) with RUNG_SLICE=24,40,48,56 (KV bound at the 32K/64K entries, attention\n'
'# over the prefix only): long session +2.5% (+3.4% incl. one slow served set), replies identical, KL 0 (results/22nov,\n'
'# step B). GPU tables w_dec8ts. Rollback: SERVED_SLICE=0 v8_server_dec8_qk.sh (the 21 Nov v8_dec8tx config; =\n'
'# v8_server_dec8_qk.pre_slice4.sh). RUNG_SLICE= (empty) keeps this build without slicing.\n'
'if [ "${SERVED_SLICE:-1}" = 0 ]; then\n'
'  export BUILD=${BUILD:-/Volumes/SSD4TB/anemll-forge-research/v8_dec8tx} GM=${GM:-$P/w_dec8tx/manifest_best.json}\n'
'else\n'
'  export BUILD=${BUILD:-/Volumes/SSD4TB/anemll-forge-research/v8_dec8ts} GM=${GM:-$P/w_dec8ts/manifest_best.json} RUNG_SLICE=${RUNG_SLICE-24,40,48,56}\n'
'fi\n'
'export TREE=${TREE-1} TREE_SAMPLED=${TREE_SAMPLED:-1}\n')
assert s.count(old)==1; open(p,"w").write(s.replace(old,new))
Y
zsh -n $L || { cp -p $BK $L; echo "syntax error, reverted" | tee -a $G; exit 1; }
if smoke served_slice4 v8_dec8ts "sliced decode entries [24, 40, 48, 56]K"; then
  echo "$(date +%T) DEPLOYED" | tee -a $G; exit 0
fi
echo "$(date +%T) deploy failed: reverting launcher, relaunching v8_dec8tx" | tee -a $G
cp -p $BK $L
smoke served_slice4_rollback v8_dec8tx "" && exit 1 || { echo "$(date +%T) ROLLBACK SMOKE FAILED" | tee -a $G; exit 4; }
