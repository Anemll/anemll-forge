#!/bin/zsh
# FUSE / REC / LUTP variants of libgpu_v8_fuse.dylib vs the current libgpu_v8_attcore.dylib: per layer, median GPU ms of
# 128-row calls and the outputs of 4 calls from a fresh state compared with the baseline (bit-same or max abs diff).
#   fuse_test.sh [CALLS]
S=${0:a:h}; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
O=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/fuse_test; mkdir -p $O; N=${1:-300}
run() {   # run LAYER P0 TAG DYLIB [ENV...]
  local L=$1 P=$2 T=$3 D=$4; shift 4
  env P0=$P DYLIB=$D SAVE=$O/${T}_L$L.npy "$@" $PY $S/gdn_prof.py $N $L $T 2>&1 | grep median
}
for L P in 0 0 60 0 3 1024; do
  run $L $P base libgpu_v8_attcore.dylib
  run $L $P same libgpu_v8_fuse.dylib
  if [ $L != 3 ]; then
    run $L $P rec_pf libgpu_v8_fuse.dylib REC=pf
    run $L $P rec_px libgpu_v8_fuse.dylib REC=px
    run $L $P conv libgpu_v8_fuse.dylib FUSE=conv
    run $L $P norm libgpu_v8_fuse.dylib FUSE=norm
  fi
  [ $L = 60 ] && run $L $P lutp libgpu_v8_fuse.dylib LUTP=1
  [ $L != 3 ] && run $L $P i8 libgpu_v8_fuse.dylib I8=1
  [ $L != 3 ] && run $L $P all_i8 libgpu_v8_fuse.dylib FUSE=all I8=1 REC=px
  run $L $P rms libgpu_v8_fuse.dylib FUSE=rms
  run $L $P add libgpu_v8_fuse.dylib FUSE=add
  run $L $P enc libgpu_v8_fuse.dylib FUSE=enc
  run $L $P all libgpu_v8_fuse.dylib FUSE=all LUTP=1
  run $L $P all_px libgpu_v8_fuse.dylib FUSE=all LUTP=1 REC=px
  run $L $P base2 libgpu_v8_attcore.dylib
done
$PY - $O <<'P'
import sys, glob, os, numpy as np
O = sys.argv[1]
for f in sorted(glob.glob(f"{O}/*_L*.npy")):
    t, L = os.path.basename(f)[:-4].rsplit("_L", 1)
    if t == "base": continue
    a = np.load(f"{O}/base_L{L}.npy").astype(np.float32); b = np.load(f).astype(np.float32)
    x = (np.random.default_rng(5).standard_normal(5120 * 128) * 0.5).astype(np.float16).astype(np.float32)   # gdn_prof's input
    d = np.abs(a - b); rel = np.linalg.norm(a - b) / np.linalg.norm(a - x)   # vs the mixer's own output (out - in)
    print(f"L{L:>2} {t:8s} " + ("bit-same" if not d.any() else f"max abs {d.max():.3g}  rel to mixer output {rel:.2e}  differing {np.mean(d > 0) * 100:.3f}%"))
P
