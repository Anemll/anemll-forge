#!/bin/zsh
# follow-ups with the 64-aligned V8 start (fewer pairs first), same requests as srv_follow_run.sh
U=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit; O=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/srv_matrix.jsonl
FOLLOW=300,700,1500 STEP=12000 $U/srv_matrix.sh $O "final_a64||6100 64500"
MAXTOK=200 FOLLOW=400,1200 $U/srv_matrix.sh $O "final96_a64||95000"
