#!/bin/zsh
# 9b debug (19 Nov 2026): 1024-token smoke, served path vs 256-row path at 1 and 2 GPU jobs in flight; forge restarted.
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; R=/Volumes/SSD4TB/anemll-forge-research/contention/gdn/g256_debug; mkdir -p $R
PORT=8765 $P/srv_stop.sh; sleep 20
env V8_DYLIB=libgpu_v8_f8q.dylib F8=x QKE=tg EXPECT_F8=x $D/v8_trace.sh $R/tr_s 1024 >/dev/null; grep "^run" $R/tr_s.log
for IF in 1 2; do env V8_DYLIB=libgpu_v8_f8q.dylib F8=x QKE=tg V8_ENGINE_MOD=v8_engine_trace_g256 V8G_I8TV=_k256 V8_INFLIGHT=$IF EXPECT_G256=1 EXPECT_F8=x $D/v8_trace.sh $R/tr_g$IF 1024 >/dev/null; grep -E "^run|in flight|Traceback" $R/tr_g$IF.log; done
$PY $D/logits_cmp.py $R/tr_s $R/tr_g1 $R/tr_g2
cd $P; PORT=8765 nohup $P/v8_server_dec8_qk.sh >/dev/null 2>&1 &
for i in $(seq 1 90); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done; echo "$(date +%T) forge ready"
