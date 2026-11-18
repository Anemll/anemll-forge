#!/bin/zsh
# sync bridges (build_sync_bridge.py), the shared build environment
cd /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/v8
OUT=${OUT:-/Volumes/SSD4TB/anemll-forge-research/v8_sync_bridge}
exec env KV_CACHE_DTYPE=kv8 KV_KEYS_T=1 ATT_INT8MM=s8,s8b,sm8,pvf8 ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_PF8_UNIT=0.015625 \
  ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096 ATT_BLOCK_LONG=4096 ATT_LONG_FROM=131072 TPS=64 TPS_LONG=32 PREFILL_LONG_FROM=98304 \
  ATT_BLOCK_PREFILL_LONG=8192 GDN_FAST=1 QCONV_INT8=0 KV_SEG=8192,8192,16384,32768,32768 V8_SPLIT=8192,9216 \
  EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
  /Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python -u build_sync_bridge.py $OUT
