#!/bin/zsh
# Decode-order variants: LUTC (fp16, K column fastest) vs the current kernel; I8T (channel-major int8 tile, vector
# stores) vs I8. Outputs compared with the named reference variant of the same run.   fuse_test2.sh [CALLS]
S=${0:a:h}; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
O=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/fuse_test2; mkdir -p $O; N=${1:-300}
run() { local L=$1 T=$2; shift 2; env DYLIB=libgpu_v8_fuse.dylib SAVE=$O/${T}_L$L.npy "$@" $PY $S/gdn_prof.py $N $L $T 2>&1 | grep median; }
for L in 0 60; do
  run $L base
  run $L lutc LUTC=1
  run $L fuse_lutc FUSE=all LUTC=1
  run $L i8 I8=1
  run $L i8t I8=1 I8T=1
  run $L fuse_i8t FUSE=all I8=1 I8T=1
  run $L base2
done
$PY - $O <<'P'
import sys, numpy as np
O = sys.argv[1]
x = (np.random.default_rng(5).standard_normal(5120 * 128) * 0.5).astype(np.float16).astype(np.float32)
for L in (0, 60):
    for t, ref in (("lutc", "base"), ("base2", "base"), ("i8t", "i8"), ("i8", "base"), ("fuse_lutc", "base")):
        a = np.load(f"{O}/{ref}_L{L}.npy").astype(np.float32); b = np.load(f"{O}/{t}_L{L}.npy").astype(np.float32)
        d = np.abs(a - b); rel = np.linalg.norm(a - b) / np.linalg.norm(a - x)
        print(f"L{L:>2} {t:9s} vs {ref:5s} " + ("bit-same" if not d.any() else f"max abs {d.max():.3g}  rel to mixer output {rel:.2e}"))
P
