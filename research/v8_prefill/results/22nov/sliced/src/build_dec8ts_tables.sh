#!/bin/zsh
# 22 Nov 2026: the GPU-table tail of build_dec8ts_all.sh, rerun after the 4-part table step was stopped by the swap
# watchdog at 07:01 (swap 204 -> 1391 MB with forge serving). NPART parts (default 2), same watchdog, then kv8 tables,
# the server manifest and the 8-HWX check.
set -u
W=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work; P=$W/pipe; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs
O=/Volumes/SSD4TB/anemll-forge-research/v8_dec8ts; GWD=/Volumes/SSD4TB/anemll-forge-research/v8_dec8ts_w; LT=dec8ts
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/22nov/sliced
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
BPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh; NPART=${NPART:-1}
glog() { echo "$(date +%T) internal free $(df -k / | awk 'NR==2{printf "%.1f", $4/1048576}') GiB | $*" | tee -a $R/disk_guard.log; }
glog "GPU tables rerun: $NPART parts (the 4-part run hit the swap watchdog at 07:01)"
rm -rf $GWD; mkdir -p $GWD
cd $P && PREP_NO_FFN=1 EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
  nice -n 10 $BPY prep_engine_par.py $O $GWD 2048 $NPART > $L/v8_prep_${LT}.log 2>&1 & BP=$!
$WD $BP 512 $L/watchdog_v8_prep_${LT}_r.log >/dev/null 2>&1 &
wait $BP || { glog "GPU tables failed again (see $L/v8_prep_${LT}.log, $L/watchdog_v8_prep_${LT}_r.log)"; exit 1; }
cd $P && EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
  $BPY prep_kv8.py $GWD/manifest.json $GWD/manifest_best.json > $L/v8_prep_kv8_${LT}.log 2>&1 || { glog "kv8 tables failed"; exit 1; }
$PY $W/v8/dec8_srv_manifest.py $O || exit 1
GWD=$GWD $PY - <<'Y' || exit 1
import json, re, os
s = open(os.environ["GWD"] + "/manifest_best.json").read()
p = sorted(set(re.findall(r'"(/[^"]*model\.hwx)"', s)))
missing = [x for x in p if not os.path.exists(x)]
print(f"{os.environ['GWD']}: {len(p)} HWX files referenced, missing {len(missing)}")
assert len(p) == 8 and not missing
Y
glog "dec8ts build done: $(du -sh $O | cut -f1) build, $(du -sh $GWD | cut -f1) GPU tables"
