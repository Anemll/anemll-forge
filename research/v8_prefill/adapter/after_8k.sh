#!/bin/zsh
# 24 Nov 2026 (yolo run, Anemll): after build_8k_sa.sh: KL-512 gate of the 8K super-adapter build (same trace as the
# base gate_kl512_tierA_s600), KL(adapter || base) on the ANE, a pass / fail gate, cleanup of this test's Core AI
# entries (coreai_cache_delete.py, same python; packages whose main.hash a kept build shares are skipped), then the 96K
# tree build with the adapter (v8/build_dec8ts_all.sh, env-driven since 24 Nov). Exit 2: the 8K gate failed (no 96K).
set -u
R=/Volumes/SSD4TB/anemll-forge-research; V02=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
H=${0:a:h}; VP=${H:h}; T=$VP/tools; L=$R/coreai_seg_logs; RES=$VP/results/24nov/adapter; mkdir -p $RES
DEST=$R/sa_test/release_vq3pA_sa_kvv8; KL=$R/gate_kl512_sa8k; BASE=$R/gate_kl512_tierA_s600
REF=$R/moved_internal_20261112/gate/ref
log() { echo "$(date +%T) $*" | tee -a $RES/run.log; }
while pgrep -f "build_8k_sa.sh" >/dev/null || pgrep -f "qwen38_coreai_build.py all" >/dev/null; do sleep 30; done
[ -e $DEST/head_T8.aimodel ] && grep -q "head" $DEST/manifest.json || { log "8K build incomplete (no head): stop"; exit 2; }
grep -q "done in" $L/sa8k_compile.log || { log "8K compile did not finish: $(tail -2 $L/sa8k_compile.log)"; exit 2; }
log "8K build + compile done: $(grep 'done in' $L/sa8k_compile.log | tail -1 | cut -c1-140)"
# --- KL-512 gate (journaled load of the same packages)
J() { DEPLOY_JOURNAL_ROOT=/Users/anemll/Documents/Codex/ios-prefill-20261030 PYTHONPATH=$T $PY - "$@" <<'EOF'
import sys, deploy_journal as dj
ev, aid, dest, label = sys.argv[1:5]
pkg = {"path": dest, "label": label}
j = dj.Journal({"mode": "kl512_eval", "out": dest, "driver_sha256": "after_8k.sh"}, aid.split(":")[0])
print(j.attempt(dj.m6_device(), pkg) if ev == "attempt" else j.result(dj.m6_device(), aid, pkg, sys.argv[5]))
EOF
}
AID=$(J attempt sa8k-kl-$$:1 $DEST "super-adapter 8K build: KL-512 eval (v8_8k entries)")
cd $V02 && env MODEL=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model \
    EMBED_NPY=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model/embed_tokens_fp16.npy \
  $PY scripts/m6_kl512_eval.py run --build $DEST --format v8 --ref-dir $REF --out $KL > $L/sa8k_kl.log 2>&1
KS=$?; J result $AID $DEST "super-adapter 8K build: KL-512 eval" $([ $KS = 0 ] && echo ok || echo failed) >/dev/null
[ $KS = 0 ] || { log "KL eval failed ($KS): $(tail -3 $L/sa8k_kl.log)"; exit 2; }
cd $V02 && $PY scripts/m6_kl512_eval.py compare --a $KL --b $BASE > $RES/kl_adapter_vs_base.json 2>$L/sa8k_kl_compare.log
cp $KL/summary.json $RES/kl512_sa8k_summary.json; cp $BASE/summary.json $RES/kl512_base8k_summary.json
$PY - $KL/summary.json $BASE/summary.json $RES/kl_adapter_vs_base.json <<'EOF' | tee -a $RES/run.log
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
c = open(sys.argv[3]).read()
print(f"8K KL-512 vs BF16 base: adapter mean {a['mean_kl']:.4f} median {a['median_kl']:.4f} top1 {a['top1_agree']:.4f} "
      f"ppl {a['ppl']:.3f} | base build mean {b['mean_kl']:.4f} top1 {b['top1_agree']:.4f} ppl {b['ppl']:.3f}")
print("KL(adapter || base build):", c.strip()[:600])
ok = a["mean_kl"] < 0.3 and a["top1_agree"] > 0.8 and a["ppl"] < 1.5 * b["ppl"]
print("GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 2)
EOF
[ ${pipestatus[1]} = 0 ] || { log "8K gate FAILED: no 96K build (fix / iterate)"; exit 2; }
# --- cleanup of this test's Core AI entries (Anemll: "cleanup coreai after test"), shared hashes kept
KEEP=$($PY - $DEST <<'EOF'
import sys, hashlib
from pathlib import Path
def h(p):
    f = Path(p) / "main.hash"
    return f.read_bytes().hex() if f.exists() else None
keep = set()
for b in ("/Volumes/SSD4TB/anemll-forge-research/v8_dec8ts", "/Volumes/SSD4TB/anemll-forge-research/coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8",
          "/Volumes/SSD4TB/anemll-forge-research/hf_stage_v8_96k"):
    for p in Path(b).rglob("*.aimodel"):
        keep.add(h(p.resolve()))
for p in sorted(Path(sys.argv[1]).glob("*.aimodel")):
    print(("SKIP " if h(p) in keep else "DEL ") + str(p))
EOF
)
echo "$KEEP" > $RES/cleanup_8k_plan.txt
DEL=(${(f)"$(echo "$KEEP" | sed -n 's/^DEL //p')"})
log "cleanup: deleting Core AI entries of ${#DEL} packages, skipping $(echo "$KEEP" | grep -c '^SKIP') shared"
$PY /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/scripts/coreai_cache_delete.py --delete $DEL > $RES/cleanup_8k.log 2>&1 \
  && log "cleanup done" || log "cleanup FAILED: $(tail -2 $RES/cleanup_8k.log)"
# --- 96K tree build with the adapter
log "96K tree build: v8_dec8ts_sa (export release_vq3pA_sa_p64)"
env OUT=$R/v8_dec8ts_sa GW=$R/v8_dec8ts_sa_w EXPORT_DIR=$R/exports/release_vq3pA_sa_p64 LT=dec8tssa RDIR=$RES/build96 \
  zsh $VP/v8/build_dec8ts_all.sh > $L/v8_dec8tssa_all.log 2>&1
BS=$?; log "96K build exit $BS: $(tail -2 $L/v8_dec8tssa_all.log)"
exit $BS
