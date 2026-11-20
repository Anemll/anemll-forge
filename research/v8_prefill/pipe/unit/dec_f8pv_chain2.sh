#!/bin/zsh
# Queue 16 (19 Nov 2026): REAL FP8 x FP8 P.V in decode GPU attention, one synthetic layer (dec_gpu_steady.py), steady
# state (ITERS copies per command buffer) at 32K / 64K / 96K (PLIST): served dec_attn_r4 (FP16 P x INT8 V) vs dec_attn_r4f8
# (e4m3 P x e4m3 V copy, research copy dec_kernels_f8d.metal); each also without P.V (DEC_SM=4: the V side's cost =
# full - no-P.V). Error of r4f8 against r4's output (same random cache), F8VS 1 and 16. Forge is down.
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python; R=/Volumes/SSD4TB/anemll-forge-research/contention/dec_f8pv
cd /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit
export DEC_DYLIB=libdec_gpu_clk.dylib ITERS=50
for P in ${=PLIST:-32768 65536 98304}; do
  for rep in 1 2; do
    DEC_SAVE=$R/o_r4_$P.npy TAG="r4 P=$P" $PY dec_gpu_steady.py $P 2>&1 | grep -E "BANNER|steady"
    DEC_METAL=dec_kernels_f8d.metal DEC_SM=4 TAG="r4 noPV P=$P" $PY dec_gpu_steady.py $P 2>&1 | grep -E "steady"
    DEC_METAL=dec_kernels_f8d.metal DEC_ATTN=dec_attn_r4f8 DEC_F8V=1.0 DEC_REF=$R/o_r4_$P.npy TAG="r4f8 P=$P" $PY dec_gpu_steady.py $P 2>&1 | grep -E "BANNER|steady|vs o_"
    DEC_METAL=dec_kernels_f8d.metal DEC_ATTN=dec_attn_r4f8 DEC_SM=4 DEC_F8V=1.0 TAG="r4f8 noPV P=$P" $PY dec_gpu_steady.py $P 2>&1 | grep -E "steady"
  done
  DEC_ATTN=dec_attn DEC_SM=2 DEC_REF=$R/o_r4_$P.npy TAG="dec_attn DEC_SM=2 (P rounded to FP8, FP16 x INT8) P=$P" $PY dec_gpu_steady.py $P 2>&1 | grep -E "steady|vs o_"
done
echo "$(date +%T) done"
