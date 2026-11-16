#!/bin/zsh
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02
exec env KV_CACHE_DTYPE=kv8 KV_KEYS_T=1 ATT_INT8MM=s8,s8b,sm8,pvf8 ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_PF8_UNIT=0.015625 \
  ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096 ATT_BLOCK_LONG=4096 ATT_LONG_FROM=131072 TPS=64 TPS_LONG=32 PREFILL_LONG_FROM=98304 \
  ATT_BLOCK_PREFILL_LONG=8192 GDN_FAST=1 QCONV_INT8=0 KV_SEG=8192,8192,16384,32768,32768 \
  EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat OUT=/Volumes/SSD4TB/anemll-forge-research/coreai_seg2 \
  /Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python -u coreai/qwen38_coreai_build.py all --ctx 8192,16384,32768,65536,98304 --pctx 8192,16384,32768,65536,98304 --kv-cache-dtype kv8
