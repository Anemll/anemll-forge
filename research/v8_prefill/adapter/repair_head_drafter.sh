#!/bin/zsh
# 24 Nov 2026: second repair after repair_dec8ts.sh. The served head (coreai_seg2/..._kv8/head_T8.aimodel, aned
# 2CBF43E551A6) and the all3 drafter (dflash2_lut4_gptq_nohead.aimodel, aned C0A30F694542) lost their model.hwx in the
# same 23 Nov purge (Core AI entries kept: load = BridgeError nilError). Delete the two entries, recompile them with
# forge.py compile (journaled), then list every ANE program both A/B arms load and require all present.
set -u
R=/Volumes/SSD4TB/anemll-forge-research; VP=${0:a:h:h}; RES=$VP/results/24nov/adapter/repair; mkdir -p $RES
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python; FORGE=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge
HEAD=$R/coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8/head_T8.aimodel
DRAFT=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/drafter/all3/dflash2_lut4_gptq_nohead.aimodel
log() { echo "$(date +%T) $*" | tee -a $RES/repair.log; }
lsof -nP -iTCP:8765 -iTCP:8767 -sTCP:LISTEN >/dev/null 2>&1 && { log "a server is up: stop"; exit 4; }
log "delete the Core AI entries of the head and the drafter (HWX gone)"
$PY $FORGE/scripts/coreai_cache_delete.py --delete $HEAD $DRAFT > $RES/delete2.log 2>&1 || { log "delete failed: $(tail -3 $RES/delete2.log)"; exit 1; }
J() { DEPLOY_JOURNAL_ROOT=/Users/anemll/Documents/Codex/ios-prefill-20261030 PYTHONPATH=$VP/tools $PY - "$@" <<'EOF'
import sys, deploy_journal as dj
ev, aid = sys.argv[1:3]; pk = [{"path": p, "main_hash": open(p + "/main.hash", "rb").read().hex(), "label": "repair: served head / drafter"} for p in sys.argv[4:]]
j = dj.Journal({"mode": "forge_compile_repair", "out": "repair_head_drafter.sh", "driver_sha256": "repair_head_drafter.sh"}, aid.split(":")[0])
for p in pk:
    print(j.attempt(dj.m6_device(), p, request={"compile_mode": 2}) if ev == "attempt" else j.result(dj.m6_device(), aid, p, sys.argv[3]))
EOF
}
AID=$(J attempt rep2-$$:1 - $HEAD $DRAFT | head -1)
log "recompile: forge.py compile --build v8_dec8ts --draft all3 nohead"
cd $FORGE && MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $PY forge.py compile --build $R/v8_dec8ts --draft $DRAFT > $RES/compile2.log 2>&1; CS=$?
J result $AID $([ $CS = 0 ] && echo ok || echo failed) $HEAD $DRAFT >/dev/null
[ $CS = 0 ] || { log "compile failed: $(tail -3 $RES/compile2.log)"; exit 1; }
log "$(grep 'done in' $RES/compile2.log | tail -1 | cut -c1-160)"
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02/scripts && $PY - $R $HEAD $DRAFT <<'EOF' | tee -a $RES/repair.log
import sys, json
from pathlib import Path
import m6_hwx_inspect as H
R, head, draft = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
bad = 0
for b in ("v8_dec8ts", "v8_dec8ts_sa"):
    man = json.loads((R / b / "manifest.json").read_text())
    pk = sorted({R / b / c["file"] for c in man["chunks"]}) + [Path(man["head"]["file"])]
    for p in pk + [Path(draft)]:
        hs = H.hwx_paths(p.resolve()); ok = bool(hs) and hs[0].exists()   # newest specialization first
        bad += not ok
        print(f"{b:13s} {p.name:36s} {'ok' if ok else 'MISSING'} {[str(h).split('/')[-3][:12] for h in hs]}")
print("all ANE programs present" if not bad else f"{bad} MISSING"); sys.exit(1 if bad else 0)
EOF
