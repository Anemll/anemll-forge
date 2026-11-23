#!/bin/zsh
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; R=/Volumes/SSD4TB/anemll-forge-research/contention/gdn/g256_debug; mkdir -p $R
PORT=8765 $P/srv_stop.sh; sleep 20
env V8_DYLIB=libgpu_v8_f8q.dylib F8=x QKE=tg EXPECT_F8=x V8_DUMP=$R/dump_s $D/v8_trace.sh $R/tr_ds 512 >/dev/null; grep -E "^run|Traceback" $R/tr_ds.log
env V8_DYLIB=libgpu_v8_f8q.dylib F8=x QKE=tg V8_ENGINE_MOD=v8_engine_trace_g256 V8G_I8TV=_k256 V8_INFLIGHT=1 EXPECT_G256=1 EXPECT_F8=x V8_DUMP=$R/dump_g $D/v8_trace.sh $R/tr_dg 512 >/dev/null; grep -E "^run|Traceback" $R/tr_dg.log
cd $P; PORT=8765 nohup $P/v8_server_dec8_qk.sh >/dev/null 2>&1 &
for i in $(seq 1 90); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done; echo "$(date +%T) forge ready"
$PY - <<PYE
import numpy as np
a, b = np.load("$R/dump_s_512.npz"), np.load("$R/dump_g_512.npz")
def key(k):
    io = 0 if k.startswith("in") else 1; L, blk = k.replace("in", "").split("_"); return (int(L), io, int(blk))
out = []
for k in sorted(a.files, key=key):
    L, io, blk = key(k)
    if L < 20 or L > 25: continue
    same = k in b.files and np.array_equal(a[k], b[k])
    out.append(f"{'in ' if io == 0 else 'out'} L{L} b{blk}:{'=' if same else 'X'}")
print(" ".join(out))
PYE
