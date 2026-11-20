#!/bin/zsh
# 18 Nov: after the coupling campaign: ANE on-chip memory probe (stops / restarts forge), then real-prefill telemetry.
until grep -q "forge ready" /Volumes/SSD4TB/anemll-forge-research/contention/runs/c18/campaign.log 2>/dev/null; do sleep 15; done
echo "$(date +%T) campaign done"
/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power/anecache_run.sh /Volumes/SSD4TB/anemll-forge-research/contention/anecache
sleep 90
echo "$(date +%T) prefill telemetry"
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python -u /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power/prefill_tel.py /Volumes/SSD4TB/anemll-forge-research/contention/runs/prefill1 --port 8765 > /Volumes/SSD4TB/anemll-forge-research/contention/runs/prefill1.log 2>&1
echo "$(date +%T) all done"
