#!/bin/zsh
# 9b debug: two 256-row runs and two served runs at 512 tokens with dumps (determinism), forge restarted.
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; R=/Volumes/SSD4TB/anemll-forge-research/contention/gdn/g256_debug; mkdir -p $R
PORT=8765 $P/srv_stop.sh; sleep 20
for r in 1 2; do
  env V8_DYLIB=libgpu_v8_f8q.dylib F8=x QKE=tg EXPECT_F8=x V8_DUMP=$R/d4_s$r $D/v8_trace.sh $R/tr_4s$r 512 >/dev/null; grep -E "^run|Traceback" $R/tr_4s$r.log
  env V8_DYLIB=libgpu_v8_f8q.dylib F8=x QKE=tg V8_ENGINE_MOD=v8_engine_trace_g256 V8G_I8TV=_k256 V8_INFLIGHT=1 V8_ANE_LOCK=1 EXPECT_G256=1 EXPECT_F8=x V8_DUMP=$R/d4_g$r $D/v8_trace.sh $R/tr_4g$r 512 >/dev/null; grep -E "^run|in flight|Traceback" $R/tr_4g$r.log
done
cd $P; PORT=8765 nohup $P/v8_server_dec8_qk.sh >/dev/null 2>&1 &
for i in $(seq 1 90); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done; echo "$(date +%T) forge ready"
$PY - <<PYE
import numpy as np
R = "$R"
def first(a, b, pre=""):
    ks = [k for k in a.files if k.startswith("in") == (pre == "in")]
    key = lambda k: tuple(map(int, k.replace("in", "").split("_")))
    f = {}
    for k in sorted(ks, key=key):
        L, blk = key(k)
        if blk not in f and (k not in b.files or not np.array_equal(a[k], b[k])): f[blk] = L
    return dict(sorted(f.items()))
s1, s2, g1, g2 = (np.load(f"{R}/d4_{t}_512.npz") for t in ("s1", "s2", "g1", "g2"))
for nm, a, b in (("s1 vs s2", s1, s2), ("g1 vs g2", g1, g2), ("s1 vs g1", s1, g1), ("s1 vs g2", s1, g2)):
    print(nm, "| outputs first diff layer per block:", first(a, b), "| inputs:", first(a, b, "in"))
PYE
