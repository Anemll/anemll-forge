#!/bin/zsh
# decode-split build (17 Nov 2026): 8 packages of 8 layers (build_dec8_chunk.py) into v8_dec8. Each package is compiled
# (journal_compile.py -> forge.py compile, one compile at a time) as soon as it is exported, alongside the next export;
# then the V8 engine's GPU tables from the new HWX (pipe/w_dec8). Swap watchdog on every process.
#   [FIRST=0 LAST=7 OUT=...] build_dec8_all.sh
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; O=${OUT:-/Volumes/SSD4TB/anemll-forge-research/v8_dec8}; W=${0:a:h}
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
mkdir -p $O; CP=
compile() {   # wait for the running compile, then compile the packages exported so far (in the background)
  if [ -n "$CP" ]; then wait $CP || { echo "$(date +%T) compile failed (see $L/v8_dec8_compile_*.log)"; exit 1; }; fi
  $PY $W/dec8_manifest.py $O || exit 1
  MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $PY $W/journal_compile.py $O > $L/v8_dec8_compile_$1.log 2>&1 & CP=$!
  $WD $CP 1024 $L/watchdog_v8_dec8_compile_$1.log >/dev/null 2>&1 &
  echo "$(date +%T) compiling packages 0..$1 (pid $CP)"
}
for f in $(seq ${FIRST:-0} ${LAST:-7}); do
  a=$((8*f)); n=$(printf "chunk_L%02d-%02d.aimodel" $a $((a+7)))
  if [ ! -f $O/$n.info.json ]; then
    env A=$a OUT=$O zsh $W/build_dec8_one.sh > $L/v8_dec8_build_$f.log 2>&1 & BP=$!
    $WD $BP 1024 $L/watchdog_v8_dec8_build_$f.log >/dev/null 2>&1 &
    wait $BP || { echo "$(date +%T) export $f failed (see $L/v8_dec8_build_$f.log)"; exit 1; }
    echo "$(date +%T) $(tail -1 $L/v8_dec8_build_$f.log)"
  fi
  compile $f
done
wait $CP || { echo "$(date +%T) compile failed"; exit 1; }
echo "$(date +%T) $(grep 'done in' $L/v8_dec8_compile_${LAST:-7}.log | tail -1)"
[ "${LAST:-7}" = 7 ] || exit 0
cd $P && PREP_NO_FFN=1 EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
  /Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python prep_engine.py $O 0 64 $P/w_dec8 2048 \
  > $L/v8_prep_dec8.log 2>&1 & BP=$!
$WD $BP 1024 $L/watchdog_v8_prep_dec8.log >/dev/null 2>&1 &
wait $BP || { echo "$(date +%T) GPU tables failed (see $L/v8_prep_dec8.log)"; exit 1; }
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python - <<'X' || exit 1
import json, re, os
s = open("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_dec8/manifest.json").read()
p = sorted(set(re.findall(r'"(/[^"]*model\.hwx)"', s)))
missing = [x for x in p if not os.path.exists(x)]
print(f"w_dec8: {len(p)} HWX files referenced, missing {len(missing)}")
assert len(p) == 8 and not missing
X
echo "$(date +%T) dec8 build done: $(du -sh $O | cut -f1)"
