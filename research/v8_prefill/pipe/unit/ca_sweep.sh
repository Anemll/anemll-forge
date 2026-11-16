#!/bin/zsh
# Long-context rebalance: first CA target chunks (4 layers each, incl. one attention layer) run whole on the ANE as a
# prefix (--ane-chunks), the rest V8 (8 streams + attproj). Timing-only 4K slices at position P.
# usage: ca_sweep.sh "P..." "CA..."
S=${0:a:h}; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs
export BUILD=/Volumes/SSD4TB/anemll-forge-research/v8_shared2 MAN=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared2r/manifest.json
for P in ${=1:-94208 61440}; do
  for CA in ${=2:-0 2 4}; do
    sleep 20
    $S/run_wd.sh 4096 ref_2k.npz ca${CA}_$P --start $P --reps 2 --shared --streams 8 --attproj --ane-chunks $CA --dylib libgpu_v8_attcore.dylib --cool-s 30
    grep -E "Traceback|Error" $L/ca${CA}_$P.log | tail -2
  done
done
for P in ${=1:-94208 61440}; do
  for CA in ${=2:-0 2 4}; do
    /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python $S/alt_table.py ane_chunks $L/ca${CA}_$P.json 2>&1 | tail -1
  done
done
