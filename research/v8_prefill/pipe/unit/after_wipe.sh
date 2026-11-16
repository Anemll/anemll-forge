#!/bin/zsh
# after the 15 Nov aned cache wipe: recompile v8_shared2 (forge.py), regenerate its GPU tables (w_shared2r, zero-copy
# from the NEW HWX), check they reference only freshly compiled HWX, then the position sweep.
S=${0:a:h}; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; W=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
O=/Volumes/SSD4TB/anemll-forge-research/v8_shared2
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge && MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 .venv/bin/python forge.py compile --build $O > $L/v8_shared2_recompile.log 2>&1 & CP=$!
/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $CP 1024 $L/watchdog_v8_shared2_recompile.log >/dev/null 2>&1 &
wait $CP; rc=$?; echo "compile rc=$rc"; grep "done in" $L/v8_shared2_recompile.log | tail -1; [ $rc = 0 ] || exit 1
cd $W && PREP_NO_FFN=1 EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
  /Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python prep_engine.py $O 0 64 $W/w_shared2r 2048 > $L/v8_prep_shared2r.log 2>&1 & BP=$!
/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $BP 1024 $L/watchdog_v8_prep_shared2r.log >/dev/null 2>&1 &
wait $BP; rc=$?; echo "prep rc=$rc"; tail -1 $L/v8_prep_shared2r.log; [ $rc = 0 ] || exit 1
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python - <<'P' || exit 1
import json, re, os
s = open("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared2r/manifest.json").read()
p = sorted(set(re.findall(r'"(/[^"]*model\.hwx)"', s)))
missing = [x for x in p if not os.path.exists(x)]
print(f"w_shared2r: {len(p)} HWX files referenced, missing {len(missing)}")
assert len(p) == 16 and not missing
P
sleep 60
MAN=$W/w_shared2r/manifest.json $S/pos_sweep.sh
