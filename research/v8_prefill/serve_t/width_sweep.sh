#!/bin/zsh
# 21 Nov 2026 (outside review: verify widths 8-16). Forge DOWN. Per width (8, 9, 12, 15, 16 first, then 10, 11, 13, 14):
# v8/build_treew_probe.py (one target group, chain / tree entries at 8K and 32K, P > 8: 8-row block solve with a partial
# last block + a full-inverse chain reference at 8K), journal_compile.py (forge.py compile, journaled), hwx_mode.py,
# pipe/unit/power/tree_width_time.py (median of 30; checks), then the package's cache entries deleted
# (coreai_cache_delete.py) and the scratch package removed. A failed width is logged and skipped. Ends with forge
# restored on the served config (launcher defaults) and the greedy smoke. Per width two independent timing blocks (a, b:
# repeat spread; review TREE_LITERATURE.md: a cliff only when the step beats it) and width 8 built and timed again at
# the end (8r: drift over the sweep).
set -u
V8=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/v8; PW=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power
PIPE=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
SPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
BPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python
RES=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/21nov
R=$RES/widths; mkdir -p $R
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
L=$R/run.log
echo "$(date +%T) width sweep: start; stopping forge (pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN))" | tee -a $L
for i in $(seq 1 60); do [ "$(lsof -nP -iTCP:8765 -sTCP:ESTABLISHED 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')" = 0 ] && break; sleep 5; done
( cd $PIPE && PORT=8765 zsh srv_stop.sh ) 2>&1 | tail -2 | tee -a $L
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "forge still listens on 8765: abort" | tee -a $L; exit 1; }
for lab in 8 9 12 15 16 10 11 13 14 8r; do
  p=${lab%r}
  O=/Volumes/SSD4TB/anemll-forge-research/twprobe_p$lab; mkdir -p $O
  echo "$(date +%T) build P=$p ($lab)" | tee -a $L
  ( cd $V8 && env KV_CACHE_DTYPE=kv8 KV_KEYS_T=1 ATT_INT8MM=s8,s8b,sm8,pvf8 ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_PF8_UNIT=0.015625 \
    ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096 ATT_BLOCK_LONG=4096 ATT_LONG_FROM=131072 TPS=64 TPS_LONG=32 PREFILL_LONG_FROM=98304 \
    ATT_BLOCK_PREFILL_LONG=8192 GDN_FAST=1 QCONV_INT8=0 KV_SEG=8192,8192,16384,32768,32768 V8_SPLIT=8192,9216 V8_ATTPROJ=1 \
    EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B TREE_P=$p \
    $BPY -u build_treew_probe.py $O ) > $R/build_p$lab.log 2>&1 & BP=$!
  $WD $BP 1024 $R/build_p$lab.watchdog.log >/dev/null 2>&1 &
  if ! wait $BP; then echo "build P=$p FAILED" | tee -a $L; tail -6 $R/build_p$lab.log | tee -a $L; rm -rf "$O"; continue; fi
  tail -1 $R/build_p$lab.log | tee -a $L
  MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $V8/journal_compile.py $O > $R/compile_p$lab.log 2>&1 & CP=$!
  $WD $CP 1024 $R/compile_p$lab.watchdog.log >/dev/null 2>&1 &
  if ! wait $CP; then echo "compile P=$p FAILED" | tee -a $L; tail -6 $R/compile_p$lab.log | tee -a $L; fi
  grep -i "done in" $R/compile_p$lab.log | tail -1 | tee -a $L
  $SPY $PW/hwx_mode.py $O/*.aimodel > $R/hwx_p$lab.log 2>&1; cut -c1-300 $R/hwx_p$lab.log | tee -a $L
  for blk in a b; do
    MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $PW/tree_width_time.py $O $R/timing_p$lab$blk.json 2>&1 | grep -v -i warn | sed "s/^/[$lab$blk] /" | tee -a $L
  done
  DEL=$(cd /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge && $SPY scripts/coreai_cache_delete.py --delete "$O" 2>&1 | tail -1)
  echo "$DEL" | tee -a $L
  if [[ "$DEL" == "deleted the entries of 1 of 1 packages"* ]]; then rm -rf "$O"; else echo "cache delete incomplete: KEPT $O" | tee -a $L; fi
done
$SPY - <<PYEOF | tee -a $L
import json, os
R = "$R"
rows, blocks = {}, {}
for lab in [str(p) for p in range(8, 17)] + ["8r"]:
    bl = [json.load(open(f"{R}/timing_p{lab}{x}.json"))["entries"] for x in "ab" if os.path.exists(f"{R}/timing_p{lab}{x}.json")]
    if not bl:
        continue
    blocks[lab] = bl
    if lab != "8r":   # the mean of the two blocks' medians per entry; the spread kept
        p = int(lab); E = {}
        for e in bl[0]:
            v = [b_[e] for b_ in bl if e in b_]
            E[e] = dict(v[0])
            for k in ("ms_median", "ms_median_tree_mask"):
                xs = [x[k] for x in v if k in x]
                if xs:
                    E[e][k] = sum(xs) / len(xs); E[e][k + "_spread"] = max(xs) - min(xs)
        rows[p] = E
b = rows.get(8, {})
def ms(e, k="ms_median"):
    return e.get(k) if e else None
for lab in ("8", "8r"):
    for bi, bl_ in enumerate(blocks.get(lab, [])):
        print(f"drift control P={lab} block {'ab'[bi]}: tree8 8K {ms(bl_.get('tree8_8k'), 'ms_median_tree_mask')}, 32K {ms(bl_.get('tree8_32k'), 'ms_median_tree_mask')}")
print("width | chain 8K  tree 8K (branched)  chain 32K  tree 32K (branched) | vs width 8: tree 8K, tree 32K | tree=chain (chain mask) 8K/32K | block solve vs inverse rel")
for p in sorted(rows):
    E = rows[p]
    c8, t8, c32, t32 = (E.get(f"{k}{p}_{c}k") for k, c in (("chain", 8), ("tree", 8), ("chain", 32), ("tree", 32)))
    rt8 = ms(t8, "ms_median_tree_mask"); rt32 = ms(t32, "ms_median_tree_mask")
    b8 = ms(b.get("tree8_8k"), "ms_median_tree_mask"); b32 = ms(b.get("tree8_32k"), "ms_median_tree_mask")
    inv = (c8 or {}).get("y_vs_inv", {})
    f = lambda x: f"{x:6.3f}" if x is not None else "   n/a"
    g = lambda x, y: f"{x / y - 1:+6.1%}" if x and y else "   n/a"
    sp = max((t8 or {}).get("ms_median_tree_mask_spread", 0), (t32 or {}).get("ms_median_tree_mask_spread", 0))
    print(f"{p:5d} | {f(ms(c8))} {f(rt8)} {f(ms(c32))} {f(rt32)} | {g(rt8, b8)} {g(rt32, b32)} (block spread <= {sp:.3f} ms) | "
          f"{(t8 or {}).get('y_rel_vs_chain8', (t8 or {}).get('y_rel_vs_chain', 'n/a'))} {(t32 or {}).get('y_rel_vs_chain8', 'n/a')} | {inv.get('rel', 'n/a')}")
json.dump({"merged": rows, "blocks": blocks}, open(f"{R}/widths_summary.json", "w"), indent=1)
PYEOF
echo "$(date +%T) width sweep: restore forge (served config)" | tee -a $L
EX="v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8 tree=policy=branched tree=v8_dec8tx tree=sampled=on lookup=off"
( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED -u COLD_SCRUB \
    RDIR=$RES LAUNCHER=v8_server_dec8_qk.sh ARM=served_after_widths RUNGS=2k TASKS=1,8 EXPECT="$EX" timeout 900 zsh $T/run_arm.sh ) > $R/restore.driver.log 2>&1
grep -E "drafter ring snapshot:|dec8: decode GPU engine|tree: BANNER|serving OpenAI" $SL | cut -c1-230 | tee -a $L
grep -E "SUMMARY|ABORT" $RES/bench/served_after_widths.log | tee -a $L
echo "$(date +%T) width sweep: done, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $L
