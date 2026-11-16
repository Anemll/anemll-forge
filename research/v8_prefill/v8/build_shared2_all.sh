#!/bin/zsh
# v8_shared2 = v8_shared + ffnq2 / offn2 (attention projections fused into the FFN entries); all 16 shared-weight chunks (target entries with the chunked FFN + V8 FFN / mixer entries) into v8_shared, then the
# manifest (the seg2 manifest with these chunks), then `forge.py compile`
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; O=/Volumes/SSD4TB/anemll-forge-research/v8_shared2; W=${0:a:h}
mkdir -p $O
for c in $(seq 0 15); do
  a=$((4*c)); b=$((4*c+3)); n=$(printf "chunk_L%02d-%02d.aimodel" $a $b)
  [ -f $O/$n.info.json ] && continue
  env A=$a B=$b zsh $W/build_shared2_one.sh > $L/v8_shared2_build_$c.log 2>&1 & BP=$!
  /Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $BP 1024 $L/watchdog_v8_shared2_build_$c.log >/dev/null 2>&1 &
  wait $BP || { echo "build $c failed"; exit 1; }
  tail -1 $L/v8_shared2_build_$c.log
done
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python - <<'PY'
import json
from pathlib import Path
O = Path("/Volumes/SSD4TB/anemll-forge-research/v8_shared2")
S2 = Path("/Volumes/SSD4TB/anemll-forge-research/coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8")
m = json.loads((S2 / "manifest.json").read_text())
for c in m["chunks"]:
    i = json.loads((O / (c["file"] + ".info.json")).read_text())
    assert i["layers"] == c["layers"] and i["gdn_j"] == c["gdn_j"] and i["att_j"] == c["att_j"], c["file"]
    c["entries"], c["mb"] = i["entries"], i["mb"]
    c["numerics"] = dict(c["numerics"], V8_SPLIT=i["split"])
    c.pop("compiled", None)
m["head"] = dict(m["head"], file=str(S2 / m["head"]["file"]))
m["v8_shared"] = {"split": i["split"], "from": str(S2), "attproj": True, "note": "target entries with the chunked FFN + ffn/ffn2/mix/mix2 + ffnq2/offn2 entries on the same weights"}
(O / "manifest.json").write_text(json.dumps(m, indent=1))
print("manifest", len(m["chunks"]), "chunks")
PY
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge && MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 .venv/bin/python forge.py compile --build $O > $L/v8_shared2_compile.log 2>&1 & CP=$!
/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $CP 1024 $L/watchdog_v8_shared2_compile.log >/dev/null 2>&1 &
wait $CP; echo "compile rc=$?"; grep "done in" $L/v8_shared2_compile.log | tail -1; du -sh $O
