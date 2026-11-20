#!/bin/zsh
# Deploy QKE=tg (19 Nov 2026): pre-deploy reference on the plain launcher, deploy launcher, the same request, compare,
# then the deploy launcher without the logits dump (the final serving state).
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; O=/Volumes/SSD4TB/anemll-forge-research/contention/qkf8/deploy; mkdir -p $O
LOG=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
up() { for i in $(seq 1 90); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && return 0; sleep 5; done; return 1; }
start() { cd $P; env "$@" PORT=8765 nohup $P/$L >/dev/null 2>&1 & up; echo "$(date +%T) up: $L $*"; grep -E "v8: variant|dec8: decode GPU" $LOG | cut -c1-400; }
echo "$(date +%T) restart plain launcher with the logits dump (pre-deploy reference)"; PORT=8765 $P/srv_stop.sh; sleep 5
L=v8_server_dec8.sh start PREFILL_LOGITS_DIR=$O/pre
$PY $D/deploy_check.py $O/pre.json
echo "$(date +%T) deploy launcher (QKE=tg) with the logits dump"; PORT=8765 $P/srv_stop.sh; sleep 5
L=v8_server_dec8_qk.sh start PREFILL_LOGITS_DIR=$O/post
$PY $D/deploy_check.py $O/post.json
$PY $D/deploy_check.py --compare $O/pre.json $O/post.json | tee $O/compare.json
echo "$(date +%T) final: deploy launcher without the dump"; PORT=8765 $P/srv_stop.sh; sleep 5
L=v8_server_dec8_qk.sh start DEPLOY=qk
echo "$(date +%T) done"
