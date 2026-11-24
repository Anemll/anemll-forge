#!/bin/zsh
# 22 Nov 2026 (research/v8_prefill/results/22nov, NEXT.md item 1 step B): a TEMPORARY build, v8_dec8ts = the served
# v8_dec8tx (TREE_FORM=corrx0, every entry unchanged) + sliced decode entries g<i>_v8s_<w>k / g<i>_t8s_<w>k at
# TREE_SLICED widths (default 24K, 40K, 48K, 56K), build_dec8t_chunk.py via build_dec8t_one.sh. Build_dec8t_all.sh's flow
# (export chunk f, then forge.py compile of the packages exported so far; one compile at a time) with Anemll's guardrails:
#   - everything on /Volumes/SSD4TB: the build and the GPU tables (GW); internal: the compile caches and TMPDIR (transient)
#   - internal free space checked before every compile and every 10 min: a compile starts only with >= GUARD + 3 GiB
#     free (one chunk is about 2.7 GiB: Core AI entry + aned HWX), and a running compile is stopped below GUARD (15 GiB);
#     then exports continue to the SSD and the script exits 3 (no cache is deleted)
#   - forge on 8765 keeps serving (not touched); no compile or export starts while a reviewer window runs
#   - nice 10, swap watchdog on every process
#   OUT=... GW=... TREE_SLICED=... GUARD=15 build_dec8ts_all.sh
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; W=${0:a:h}
O=${OUT:-/Volumes/SSD4TB/anemll-forge-research/v8_dec8ts}; GWD=${GW:-/Volumes/SSD4TB/anemll-forge-research/v8_dec8ts_w}
LT=${LT:-dec8ts}; GUARD=${GUARD:-15}; NEED=$((GUARD + 3))
R=${RDIR:-/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/22nov/sliced}
# 24 Nov 2026: EXPORT_DIR (exported: the package builder and the GPU tables), LT (log tag), RDIR (results) from the
# environment; the defaults are the 22 Nov build's
export EXPORT_DIR=${EXPORT_DIR:-/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat}
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
BPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
# TMPDIR stays the default (internal, transient): on the exFAT SSD the first export (22 Nov 02:12) failed in the
# palettizer (NUM_PALETTES 256 vs 4-bit indices: weights no longer matching their registered LUTs)
export TREE_FORM=corrx0 TREE_SLICED=${TREE_SLICED:-24576,40960,49152,57344}
mkdir -p $O $R; GL=$R/disk_guard.log
free_gib() { df -k / | tail -1 | awk '{printf "%.1f", $4 / 1048576}'; }
glog() { echo "$(date +%T) internal free $(free_gib) GiB | $*" | tee -a $GL; }
LOCK=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/22nov_review/windows/runner.lock
reviewer_busy() { pgrep -f "window[.]zsh" > /dev/null && return 0; [ -e $LOCK ] && [ -n "$(lsof -t $LOCK 2>/dev/null)" ] && return 0; return 1; }
reviewer_wait() {   # only looked at (pgrep / lsof), never touched
  while reviewer_busy; do echo "$(date +%T) reviewer window running: waiting" >> $GL; sleep 60; done
}
killtree() { local c; for c in $(pgrep -P $1); do killtree $c; done; kill -TERM $1 2>/dev/null; }
CP=; PAUSED=
compile_wait() {   # wait for the running compile; the guard stops it below GUARD
  [ -n "$CP" ] || return 0
  local n=0
  while kill -0 $CP 2>/dev/null; do
    if (( $(free_gib) < GUARD )); then
      glog "below the ${GUARD} GiB guard during a compile: stopping it (pid $CP)"; killtree $CP; wait $CP; CP=; PAUSED=guard; return 1
    fi
    n=$((n + 1)); [ $((n % 30)) = 0 ] && glog "compile running (pid $CP)"
    sleep 20
  done
  wait $CP || { [ -e $STOPF ] && { CP=; PAUSED=guard; return 1; }; glog "compile failed (see $L/v8_${LT}_compile_*.log)"; exit 1; }
  CP=; glog "compile done: $(grep 'done in' $L/v8_${LT}_compile_$CF.log | tail -1 | cut -c1-120)"
}
compile() {
  compile_wait || return
  [ -n "$PAUSED" ] && return
  if (( $(free_gib) < NEED )); then
    glog "compile of packages 0..$1 NOT started: needs >= ${NEED} GiB free (guard ${GUARD} + 3 for one chunk)"; PAUSED=guard; return
  fi
  reviewer_wait
  $PY $W/dec8_manifest.py $O || exit 1
  glog "compiling packages 0..$1"; CF=$1
  MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 nice -n 10 $PY $W/journal_compile.py $O > $L/v8_${LT}_compile_$1.log 2>&1 & CP=$!
  $WD $CP 1024 $L/watchdog_v8_${LT}_compile_$1.log >/dev/null 2>&1 &
  compile_wait   # no overlap with the next export
}
# 22 Nov 02:38 fix: the guard used to be polled only while waiting for a compile, so a compile overlapping the next
# export ran unwatched below 15 GiB. Now a watcher polls every 15 s for the whole run and stops this build's compile
# and export below GUARD (STOPF), and export / compile no longer overlap (a lower transient peak).
STOPF=$R/guard.stop; rm -f $STOPF
guard_watch() {
  while kill -0 $1 2>/dev/null; do
    if (( $(free_gib) < GUARD )); then
      glog "below the ${GUARD} GiB guard: stopping this build's compile / export"; touch $STOPF
      for p in $(pgrep -f "coreai_compile.py --build $O|build_dec8t_chunk.py $O|journal_compile.py $O"); do kill -TERM $p 2>/dev/null; done
      return
    fi
    sleep 15
  done
}
guard_watch $$ &
glog "start: OUT $O, EXPORT_DIR $EXPORT_DIR, TREE_FORM $TREE_FORM, TREE_SLICED $TREE_SLICED, guard $GUARD GiB"
for f in $(seq ${FIRST:-0} ${LAST:-7}); do
  a=$((8*f)); n=$(printf "chunk_L%02d-%02d.aimodel" $a $((a+7)))
  if [ ! -f $O/$n.info.json ]; then
    reviewer_wait
    env A=$a OUT=$O nice -n 10 zsh $W/build_dec8t_one.sh > $L/v8_${LT}_build_$f.log 2>&1 & BP=$!
    $WD $BP 1024 $L/watchdog_v8_${LT}_build_$f.log >/dev/null 2>&1 &
    wait $BP || { [ -e $STOPF ] && { glog "PAUSED at the disk guard during export $f"; exit 3; }; glog "export $f failed (see $L/v8_${LT}_build_$f.log)"; exit 1; }
    glog "exported $f: $(tail -1 $L/v8_${LT}_build_$f.log)"
    grep -E "v8s_|t8s_" $L/v8_${LT}_build_$f.log >> $R/sliced_entries.log
  fi
  compile $f
done
compile_wait
if [ -n "$PAUSED" ]; then glog "PAUSED at the disk guard: exports done to $(ls $O/*.info.json | wc -l | tr -d ' ') packages; compile incomplete"; exit 3; fi
glog "all packages compiled; GPU tables -> $GWD"
mkdir -p $GWD
cd $P && PREP_NO_FFN=1 EXPORT_DIR=$EXPORT_DIR MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
  nice -n 10 $BPY prep_engine_par.py $O $GWD 2048 4 > $L/v8_prep_${LT}.log 2>&1 & BP=$!   # 22 Nov: 4 parts, = serial (sliced/prep_par_check.txt)
$WD $BP 1024 $L/watchdog_v8_prep_${LT}.log >/dev/null 2>&1 &
wait $BP || { glog "GPU tables failed (see $L/v8_prep_${LT}.log)"; exit 1; }
cd $P && EXPORT_DIR=$EXPORT_DIR MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
  $BPY prep_kv8.py $GWD/manifest.json $GWD/manifest_best.json > $L/v8_prep_kv8_${LT}.log 2>&1 || { glog "kv8 tables failed"; exit 1; }
$PY $W/dec8_srv_manifest.py $O || exit 1
GWD=$GWD $PY - <<'Y' || exit 1
import json, re, os
s = open(os.environ["GWD"] + "/manifest_best.json").read()
p = sorted(set(re.findall(r'"(/[^"]*model\.hwx)"', s)))
missing = [x for x in p if not os.path.exists(x)]
print(f"{os.environ['GWD']}: {len(p)} HWX files referenced, missing {len(missing)}")
assert len(p) == 8 and not missing
Y
glog "dec8ts build done: $(du -sh $O | cut -f1) build, $(du -sh $GWD | cut -f1) GPU tables"
