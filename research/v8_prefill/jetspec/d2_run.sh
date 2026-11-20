#!/bin/zsh
# D2: LUT4 (Hadamard + GPTQ) JetSpec++ adapter on the ANE, quiet machine (forge down): 3 x 4 blocks (8K) and 4 x 3 blocks
# (8K + 32K); timing (tree_decomp_time.py), then resident memory per package (ane_pkg_mem.py, one package per process)
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov/d2
B=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/v8/build_jsp_adapter_all.sh
PW=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power
SPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
S=/Volumes/SSD4TB/anemll-forge-research/tree20
LUT=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/adapter_lut4hg
for cfg in "b4:4:8192" "b3:3:8192,32768"; do
  t=${cfg%%:*}; rest=${cfg#*:}; n=${rest%%:*}; c=${rest#*:}
  echo "STATE=working ITEM=D2-lut4-ane-quiet NOTE=build+time $t ($n blocks per package, ctx $c)" > /tmp/m6_yolo_status.txt
  mkdir -p $R/$t
  LUT=$LUT JSP_ROT=1 COUNT=$n CTXS=$c zsh $B $S/jsp_lut4_$t $R/$t > $R/$t/driver.log 2>&1 || { echo "$t failed"; continue; }
  for p in $S/jsp_lut4_$t/*.aimodel; do
    fns=$(python3 -c "import json,sys; m=json.load(open('$S/jsp_lut4_$t/manifest.json')); print(','.join(next(c['entries'] for c in m['chunks'] if c['file']=='$(basename $p)')))")
    echo "STATE=working ITEM=D2-lut4-ane-quiet NOTE=memory $t $(basename $p)" > /tmp/m6_yolo_status.txt
    PKGS=$p FNS=$fns MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $PW/ane_pkg_mem.py $R/$t/mem_$(basename $p .aimodel).json > $R/$t/mem_$(basename $p .aimodel).log 2>&1 & MP=$!
    $WD $MP 1024 $R/$t/mem.watchdog.log >/dev/null 2>&1 &
    wait $MP; echo "mem $(basename $p) rc $?"
  done
done
echo ALLDONE
