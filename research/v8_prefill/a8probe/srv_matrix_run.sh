#!/bin/zsh
# the V8 server matrix of 16 Nov (KV8 / K8V16 / fp16 KV / attproj / --v8-min for follow-ups); results srv_matrix.jsonl
U=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit; O=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/srv_matrix.jsonl
$U/srv_matrix.sh $O "k8v16|--v8-k8v16 1|2000 6100 16200 32600 64500 95000" \
                    "kv8||6100 16200 64500 95000" \
                    "fp16|--v8-kv8 0|2000 6100 16200 32600|CTX=32768" \
                    "k8v16_ap32k|--v8-k8v16 1 --v8-attproj-from 32768|64500 95000"
FOLLOW=300,700,1500 STEP=12000 $U/srv_matrix.sh $O "k8v16_min1024|--v8-k8v16 1|6100 64500" "k8v16_min256|--v8-k8v16 1 --v8-min 256|6100 64500"
