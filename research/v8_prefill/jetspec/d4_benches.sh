#!/bin/zsh
# D4 identity benches on the corrx0 build (v8_dec8tx), realistic_v1 9 prompts x 2K-96K, then the served forge restored.
#   cap3     the served chain entry only, at most 3 drafts accepted per cycle (a different cycle split: the control)
#   topc1    every cycle through the tree entry, the tree always the chain (exactness of the tree entry end to end)
#   branched the tree entry only when the tree is not the chain (tree8 as it would serve)
set -u
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree; P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
export BUILD=/Volumes/SSD4TB/anemll-forge-research/v8_dec8tx GM=$P/w_dec8tx/manifest_best.json
for arm in "d4_cap3:cap:7:3" "d4_topc1:always:1:7" "d4_branched:branched:4:7"; do
  a=${arm%%:*}; rest=${arm#*:}; pol=${rest%%:*}; rest=${rest#*:}; topc=${rest%%:*}; cap=${rest#*:}
  echo "STATE=working ITEM=D4-identity-bench NOTE=arm $a (TREE_POLICY=$pol TOPC=$topc CAP=$cap) on v8_dec8tx" > /tmp/m6_yolo_status.txt
  ( cd $T && LAUNCHER=v8_server_tree.sh ARM=$a TREE=1 TREE_POLICY=$pol TREE_TOPC=$topc TREE_CAP=$cap \
      EXPECT="tree=policy=$pol tree=topc=$topc v8=qk=attn_qk8isk_tg dec8=libdec_gpu_tree.dylib" timeout 3600 zsh run_arm.sh > $R/d4/bench_$a.driver.log 2>&1 )
  rc=$?; echo "$a rc $rc"; [ $pol = cap ] && { grep -q "policy=cap cap=$cap" $R/bench/$a.log || echo "$a: banner lacks cap=$cap"; }
  $PY $T/compare.py $R/bench/chain_v1.json $R/bench/$a.json > $R/d4/compare_$a.txt 2>&1
  tail -12 $R/d4/compare_$a.txt
done
echo "STATE=working ITEM=restore-forge NOTE=D4 benches done; restoring the served forge" > /tmp/m6_yolo_status.txt
( cd $T && LAUNCHER=v8_server_dec8_qk.sh ARM=final_served3 RUNGS=2k TASKS=1,8 EXPECT="v8=qk=attn_qk8isk_tg dec8=libdec_gpu_clk.dylib" \
    timeout 600 zsh run_arm.sh > $R/d4/bench_final_served3.driver.log 2>&1 )
grep -o "sha [0-9a-f]\{16\}" $R/bench/final_served.log $R/bench/final_served3.log
echo ALLDONE
