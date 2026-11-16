#!/bin/zsh
# follow-up turns after the aligned-feed fix (defaults now --v8-k8v16 1 --v8-min 256), 6K / 64K, then a 95K session
U=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit; O=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/srv_matrix.jsonl
FOLLOW=300,700,1500 STEP=12000 $U/srv_matrix.sh $O "final|--v8-min 256|6100 64500"
MAXTOK=200 FOLLOW=400,1200,3000 $U/srv_matrix.sh $O "final96|--v8-min 256|95000"
