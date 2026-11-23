#!/bin/zsh
# Full GPU-table build for a V8 build (22 Nov 2026), the path when gpu_tables_kit.py install reports that this Mac's
# compiled programs differ from the kit's (another OS build or compiler). The same steps as v8/build_dec8ts_tables.sh
# with every machine path a variable:
#   1. prep_engine_par.py: the mixer matrices as tables into each package's compiled HWX (found through the Core AI
#      cache, so the build must be compiled first) + norms, rotations, adapters, DeltaNet constants from the export
#   2. prep_kv8.py: the attention k / v projections as the export's own INT8 weights -> manifest_best.json
#   3. check: every HWX the manifest names exists, one per package
# Needs the builder python (torch + the Core AI Python tools: requirements-conversion.txt), the release 0.2 export and
# the checkpoint directory with the small tensors (both from anemll/anemll-quantized-qwen3.8-27b-for-CoreAI:
# export/release_vq3pA_mixh_s600_k1_mat/ and model/), the hwx_parsing tool (research/v8_prefill/hwx, `make`), read
# access to aned's cache. About 25 to 40 min, up to ~6 GB of memory per part (NPART=1 on a 32 GB Mac with a server up).
#   BUILD=<build dir> OUT=<table dir> EXPORT_DIR=<export> MODEL=<checkpoint dir> BPY=<builder python> \
#   [PY=<runtime python>] [NPART=1] [LMAX=2048] zsh pipe/build_gpu_tables.sh
# Then compare with the kit's tables (should be identical when only the compiled layout moved):
#   python pipe/gpu_tables_kit.py verify $OUT <kit dir>    (reports which files differ)
set -eu
P=${0:a:h}
: ${BUILD:?} ${OUT:?} ${EXPORT_DIR:?} ${MODEL:?} ${BPY:?}
PY=${PY:-python3}; NPART=${NPART:-1}; LMAX=${LMAX:-2048}
mkdir -p $OUT
echo "$(date +%T) tables: $BUILD -> $OUT ($NPART parts, Lmax $LMAX)"
cd $P
PREP_NO_FFN=1 EXPORT_DIR=$EXPORT_DIR MODEL=$MODEL nice -n 10 $BPY prep_engine_par.py $BUILD $OUT $LMAX $NPART
EXPORT_DIR=$EXPORT_DIR MODEL=$MODEL $BPY prep_kv8.py $OUT/manifest.json $OUT/manifest_best.json
OUT=$OUT BUILD=$BUILD $PY - <<'Y'
import json, os, re
s = open(os.environ["OUT"] + "/manifest_best.json").read()
p = sorted(set(re.findall(r'"(/[^"]*model\.hwx)"', s)))
n = len({c["file"] for c in json.load(open(os.environ["BUILD"] + "/manifest.json"))["chunks"]})
missing = [x for x in p if not os.path.exists(x)]
print(f"{os.environ['OUT']}: {len(p)} HWX files referenced ({n} packages), missing {len(missing)}")
assert len(p) == n and not missing
Y
echo "$(date +%T) tables done: $(du -sh $OUT | cut -f1); server: GM=$OUT/manifest_best.json"
