#!/bin/zsh
# Queue 16c (19 Nov 2026): what caps the decode attention's K side (105-110 GB/s)? One synthetic layer,
# unit/dec_gpu_steady.py (steady state, 50 copies per command buffer, 2 repeats), served dylib libdec_gpu_clk.dylib,
# research kernels copy dec_kernels_ks.metal (DEC_ATTN=dec_attn_r4ks; served dec_kernels.metal untouched):
#   VAR (DEC_SM % 20): 0 full | 4 no P.V | 8 Q.K only (no softmax, no P.V) | 9 softmax only (no Q.K, no P.V)
#   KM (DEC_SM >= 20, DEC_KMAJ=1): K read key-major (dense 64 KB tiles) instead of the ANE's dims-major layout
#   DEC_CK 1024 / 512: 2x / 4x the threadgroups (chunks of keys). (DEC_M48 was measured 17 Nov: 2.8x slower.)
# Checksums against the served kernel's output (same random cache). Forge on 8765 stays down.
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python; R=/Volumes/SSD4TB/anemll-forge-research/contention/dec_kside; mkdir -p $R
cd /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit
export DEC_DYLIB=libdec_gpu_clk.dylib ITERS=50
run() { env "$@" $PY dec_gpu_steady.py $P 2>&1 | grep -E "BANNER|steady|vs o_|error|rror:" | sed -E 's/\| dylib libdec_gpu_clk.dylib sha 2b64e979a91a414f //'; }
for P in ${=PLIST:-65536 98304}; do
  for rep in 1 2; do
    run TAG="served r4 P=$P" DEC_SAVE=$R/o_$P.npy
    run TAG="ks full P=$P" DEC_METAL=dec_kernels_ks.metal DEC_ATTN=dec_attn_r4ks DEC_SM=0 DEC_REF=$R/o_$P.npy
    run TAG="ks noPV P=$P" DEC_METAL=dec_kernels_ks.metal DEC_ATTN=dec_attn_r4ks DEC_SM=4
    run TAG="ks QKonly P=$P" DEC_METAL=dec_kernels_ks.metal DEC_ATTN=dec_attn_r4ks DEC_SM=8
    run TAG="ks softmax-only P=$P" DEC_METAL=dec_kernels_ks.metal DEC_ATTN=dec_attn_r4ks DEC_SM=9
    run TAG="ks keymajor full P=$P" DEC_METAL=dec_kernels_ks.metal DEC_ATTN=dec_attn_r4ks DEC_SM=20 DEC_KMAJ=1 DEC_REF=$R/o_$P.npy
    run TAG="ks keymajor noPV P=$P" DEC_METAL=dec_kernels_ks.metal DEC_ATTN=dec_attn_r4ks DEC_SM=24 DEC_KMAJ=1
    run TAG="ks keymajor QKonly P=$P" DEC_METAL=dec_kernels_ks.metal DEC_ATTN=dec_attn_r4ks DEC_SM=28 DEC_KMAJ=1
    run TAG="served r4 CK1024 P=$P" DEC_CK=1024 DEC_REF=$R/o_$P.npy
    run TAG="served r4 CK512 P=$P" DEC_CK=512 DEC_REF=$R/o_$P.npy
    run TAG="ks noPV CK1024 P=$P" DEC_METAL=dec_kernels_ks.metal DEC_ATTN=dec_attn_r4ks DEC_SM=4 DEC_CK=1024
    run TAG="ks QKonly CK1024 P=$P" DEC_METAL=dec_kernels_ks.metal DEC_ATTN=dec_attn_r4ks DEC_SM=8 DEC_CK=1024
  done
done
rm -f $R/o_*.npy; echo "$(date +%T) done"
