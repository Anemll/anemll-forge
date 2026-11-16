#!/bin/zsh
# ANE DeltaNet mixers with 8 streams (4 pairs in flight), shared build, prefill only, rotating layer sets per rep
S=${0:a:h}; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
while pgrep -f run96.sh >/dev/null; do sleep 30; done
sleep 60
export BUILD=/Volumes/SSD4TB/anemll-forge-research/v8_shared MAN=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared/manifest.json
M16="1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61"; M24="$M16,2,10,18,26,34,42,50,58"
for NR in 32768:ref_32k.npz 65536:ref_64k.npz; do
  N=${NR%%:*}; R=${NR##*:}
  $S/run_wd.sh $N $R amix8_$N --reps 6 --shared --streams 8 --heads-after --cool-s 120 --ane-mix-alt ";$M16;$M24"
  $PY $S/alt_summ.py $L/amix8_$N.json 2>&1 | tail -4
  sleep 60
done
