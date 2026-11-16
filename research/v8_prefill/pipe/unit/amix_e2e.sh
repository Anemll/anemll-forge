#!/bin/zsh
S=${0:a:h}; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs
export BUILD=/Volumes/SSD4TB/anemll-forge-research/v8_shared MAN=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared/manifest.json
M16="1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61"
$S/run_wd.sh 65536 ref_64k.npz amixe2e_65536 --reps 1 --shared --streams 8 --ane-mix $M16 --handover --ho-overlap --heads-after --cool-s 120
grep -E '^\{"ane_only_same|^\{"e2e' $L/amixe2e_65536.log | cut -c1-400
grep -E '^\{"rep"' $L/amixe2e_65536.log | /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python $S/summ.py
