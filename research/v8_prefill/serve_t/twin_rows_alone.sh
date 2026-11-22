#!/bin/zsh
# 22 Nov 2026 (paper-sweep A1): GPU drafter cost twin alone at 8 / 16 / 32 / 64 rows (draft-ahead guesses B = 1 / 2 / 4 / 8),
# 8 s each, results/22nov/probes/twin_rows_alone.txt. GPU only; run only while Pi is idle (it competes for DRAM).
W=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power
O=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/22nov/probes/twin_rows_alone.txt
: > $O
for r in 8 16 32 64; do echo "== rows $r" | tee -a $O; env V2=1 $W/drafter_gpu_twin $r 8 0 2>&1 | grep -E "drafter twin|check:" | tee -a $O; sleep 3; done
