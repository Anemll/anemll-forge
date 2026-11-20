#!/bin/zsh
# 20 Nov 2026 tree study: the decode-split build with tree verify entries (build_dec8t_chunk.py via build_dec8t_one.sh)
# into v8_dec8t, exactly build_dec8_all.sh's flow: each package compiled (journal_compile.py -> forge.py compile, one
# compile at a time) as soon as it is exported; then the V8 engine's GPU tables from the new HWX (pipe/w_dec8t, plus
# prep_kv8.py's INT8 K / V tables -> manifest_best.json) and the server manifest (dec8_srv_manifest.py). Swap watchdog
# on every process. TREE_FORM (ctap | gather) passes through.
#   [FIRST=0 LAST=7 OUT=...] build_dec8t_all.sh
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; O=${OUT:-/Volumes/SSD4TB/anemll-forge-research/v8_dec8t}; W=${0:a:h}
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
BPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "forge listens on 8765: stop it first (one model at a time)"; exit 1; }
mkdir -p $O; CP=
compile() {
  if [ -n "$CP" ]; then wait $CP || { echo "$(date +%T) compile failed (see $L/v8_dec8t_compile_*.log)"; exit 1; }; fi
  $PY $W/dec8_manifest.py $O || exit 1
  MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $PY $W/journal_compile.py $O > $L/v8_dec8t_compile_$1.log 2>&1 & CP=$!
  $WD $CP 1024 $L/watchdog_v8_dec8t_compile_$1.log >/dev/null 2>&1 &
  echo "$(date +%T) compiling packages 0..$1 (pid $CP)"
}
for f in $(seq ${FIRST:-0} ${LAST:-7}); do
  a=$((8*f)); n=$(printf "chunk_L%02d-%02d.aimodel" $a $((a+7)))
  if [ ! -f $O/$n.info.json ]; then
    env A=$a OUT=$O zsh $W/build_dec8t_one.sh > $L/v8_dec8t_build_$f.log 2>&1 & BP=$!
    $WD $BP 1024 $L/watchdog_v8_dec8t_build_$f.log >/dev/null 2>&1 &
    wait $BP || { echo "$(date +%T) export $f failed (see $L/v8_dec8t_build_$f.log)"; exit 1; }
    echo "$(date +%T) $(tail -1 $L/v8_dec8t_build_$f.log)"
  fi
  compile $f
done
wait $CP || { echo "$(date +%T) compile failed"; exit 1; }
echo "$(date +%T) $(grep 'done in' $L/v8_dec8t_compile_${LAST:-7}.log | tail -1)"
[ "${LAST:-7}" = 7 ] || exit 0
mkdir -p $P/w_dec8t
cd $P && PREP_NO_FFN=1 EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
  $BPY prep_engine.py $O 0 64 $P/w_dec8t 2048 > $L/v8_prep_dec8t.log 2>&1 & BP=$!
$WD $BP 1024 $L/watchdog_v8_prep_dec8t.log >/dev/null 2>&1 &
wait $BP || { echo "$(date +%T) GPU tables failed (see $L/v8_prep_dec8t.log)"; exit 1; }
cd $P && EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
  $BPY prep_kv8.py $P/w_dec8t/manifest.json $P/w_dec8t/manifest_best.json > $L/v8_prep_kv8_dec8t.log 2>&1 || { echo "kv8 tables failed"; exit 1; }
$PY $W/dec8_srv_manifest.py $O || exit 1
$PY - <<'Y' || exit 1
import json, re, os
s = open("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_dec8t/manifest_best.json").read()
p = sorted(set(re.findall(r'"(/[^"]*model\.hwx)"', s)))
missing = [x for x in p if not os.path.exists(x)]
print(f"w_dec8t: {len(p)} HWX files referenced, missing {len(missing)}")
assert len(p) == 8 and not missing
Y
echo "$(date +%T) dec8t build done: $(du -sh $O | cut -f1)"
