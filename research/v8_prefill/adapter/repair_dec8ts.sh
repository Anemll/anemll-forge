#!/bin/zsh
# 24 Nov 2026: repair the served tree build v8_dec8ts. On 23 Nov 17:13-17:15 (low internal disk) aned's model.hwx of 4
# packages (L16-23, L32-39, L40-47, L48-55) disappeared (model.src stubs left) while their Core AI entries stayed: loads
# fail with BridgeError nilError and forge cannot start. Delete those 4 entries (coreai_cache_delete.py, same python),
# recompile (journal_compile.py = journaled forge.py compile), then prove the served GPU tables w_dec8ts still match:
# gpu_tables_kit.py install of the HF kit (weight-section sha256 per program) into a scratch dir + verify against w_dec8ts.
set -u
R=/Volumes/SSD4TB/anemll-forge-research; B=$R/v8_dec8ts; VP=${0:a:h:h}; RES=$VP/results/24nov/adapter/repair; mkdir -p $RES
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
log() { echo "$(date +%T) $*" | tee -a $RES/repair.log; }
lsof -nP -iTCP:8765 -iTCP:8767 -sTCP:LISTEN >/dev/null 2>&1 && { log "a server is up: stop"; exit 4; }
PK=($B/chunk_L16-23.aimodel $B/chunk_L32-39.aimodel $B/chunk_L40-47.aimodel $B/chunk_L48-55.aimodel)
log "delete the Core AI entries of ${#PK} packages whose HWX is gone"
$PY /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/scripts/coreai_cache_delete.py --delete $PK > $RES/delete.log 2>&1 \
  || { log "delete failed: $(tail -3 $RES/delete.log)"; exit 1; }
log "recompile (journaled)"
MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $PY $VP/v8/journal_compile.py $B > $RES/compile.log 2>&1 || { log "compile failed: $(tail -3 $RES/compile.log)"; exit 1; }
log "$(grep 'done in' $RES/compile.log | tail -1 | cut -c1-160)"
rm -rf $RES/w_check
$PY $P/gpu_tables_kit.py install $R/hf_stage_v8_96k/gpu/w_dec8ts $B $RES/w_check > $RES/kit_install.log 2>&1; KS=$?
log "kit install exit $KS: $(grep -cE 'MATCH' $RES/kit_install.log) MATCH lines; $(tail -1 $RES/kit_install.log | cut -c1-160)"
[ $KS = 0 ] || { log "compiled programs differ from the kit: the served tables need a rebuild (build_gpu_tables.sh)"; exit 2; }
$PY $P/gpu_tables_kit.py verify $RES/w_check $P/w_dec8ts > $RES/kit_verify.log 2>&1; VS=$?
log "verify vs served w_dec8ts exit $VS: $(tail -1 $RES/kit_verify.log | cut -c1-160)"
python3 -I -c "
import json,re,os
s=open('$P/w_dec8ts/manifest_best.json').read(); p=sorted(set(re.findall(r'\"(/[^\"]*model\.hwx)\"',s)))
print('served w_dec8ts:', len(p), 'HWX referenced, missing', sum(not os.path.exists(x) for x in p))" | tee -a $RES/repair.log
exit $VS
