# Operand types of the served V8 prefill pipeline (18 Nov 2026)

From a read of the code, nothing run: the served GPU engine is `libgpu_v8_srv3.dylib` (`v8_server.py` default
`--v8-dylib`; the server log says "GPU engine libgpu_v8_srv3.dylib"), built from the current `gpu_v8.swift`, its
kernels compiled at load from the current `kernels.metal`. Served flags: FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1
I8KV=1 LRM=1 QK8=1 TKV=2048, engine KV8=1 K8V16=1 NT=128. In prefill the GPU runs `v8_submit_pair` /
`v8_submit_pair_nv`; the ANE runs `ffn2_L` (attention projections on the ANE are off; `ffnq2_L` / `offn2_L` exist in
the dec8 packages but are not run in prefill). SILU, MLP_SILU, GDN_SQ, GDN_SV, MLP_DS are ANE builder numerics, not
read by the GPU engine; LUTC only affects FP16 fallback paths not reached with I8=1 I8A=1.

File key: S = v8_work/pipe/engine/gpu_v8.swift, K = v8_work/pipe/engine/kernels.metal, B =
anemll-forge-v02/coreai/qwen38_coreai_build.py, C = v8_work/v8/chunkmlp.py. MAC per token from the shapes; MM = Metal 4
`matmul2d`, 4 simdgroups (no simdgroup_matrix anywhere).

| # | step (MAC / token) | unit | kernel | weights stored | A x B -> acc | engine | flag | notes |
|---|---|---|---|---|---|---|---|---|
| D2 | RMSNorm + mix_in Hadamard | GPU | rms_partial K:909, rms_fwht4 K:955 | fp32 signs | fp32 | ALU | FUSE | |
| D3 | quantize activations | GPU | quant8_blk K:367 | | fp16 -> int8 | ALU | I8 | scale max/127 per (1024-channel block, token) |
| D4 | qkv 5120 -> 10240 (52.4M) | GPU | lut_gemm32_i8t_k256 K:330 (body K:247-317) | 4-bit palette indices + fp16 palette per subgroup + row scale | int8 x int8 -> int32 -> fp32 | MM 32x128x256 | I8, I8T, I8TV | palette re-quantized to int8 in the kernel |
| D5 | z 5120 -> 6144 (31.5M) | GPU | same | 4-bit | int8 x int8 | MM | I8 | |
| D6 | low-rank qkv + z, rank 64 (1.7M) | GPU | dense_splitk K:720, dense_gemm K:425 | fp16 | fp16 x fp16 -> fp32 | MM | LRM | |
| D7 | a / b 5120 -> 128 (0.66M) | GPU | dense_splitk | fp16 | fp16 x fp16 -> fp32 | MM | | |
| D8-D11 | conv1d + SiLU + L2 norm, gates, recurrence (about 3M FMA, serial), gated norm + Hadamard | GPU | gdn_conv_qkv K:1025, gdn_gates K:480, gdn_rec K:489, gdn_norm_fwht4 K:986 | fp32 | fp32 | ALU | | not matrix work |
| D12 | out 6144 -> 5120 (31.5M) | GPU | quant8_blk + lut_gemm32_i8t_k256 | 4-bit | int8 x int8 | MM | I8 | |
| D13 | low-rank out (0.7M) | GPU | lowrank S:172 | fp16 | fp16 x fp16 | MM | | |
| A4 | q + gate 5120 -> 12288 (62.9M) | GPU | lut_gemm32_i8t_k256 | 4-bit | int8 x int8 | MM | I8 + I8A | |
| A6 | k, v 5120 -> 1024 each (10.5M) | GPU | dense_i8_splitk K:349 | INT8 export + row scale | int8 x int8 -> int32 | MM 32x128x1024 | I8KV | |
| A8 | q / k norms, RoPE, quantize Q / K / V | GPU | attn_prep8qv K:1771 | | fp32 | ALU | KV8 QK8 K8V16 | K, V int8 into the ANE cache + fp16 V copy |
| A9 | Q.K (6144 x L) | GPU | attn_qk8isk K:2069 | ANE INT8 K cache | int8 x int8 -> int32 -> fp16 | MM 64x64x256 | QK8 | |
| A10 | softmax | GPU | attn_softmax8n K:2090 | | fp32 math, fp16 P | ALU | | |
| A11 | P.V (6144 x L) | GPU | attn_pv_w K:1526 | fp16 V copy | fp16 x fp16 -> fp32 | MM 64x128xn | K8V16 | |
| A14 | o 6144 -> 5120 (31.5M) | GPU | quant8_blk + lut_gemm32_i8t_k256 | 4-bit | int8 x int8 | MM | I8A | |
| A5, A7, A15 | low-rank q, k, v, o | GPU | lowrank | fp16 | fp16 x fp16 | MM | | |
| F1 | RMSNorm | ANE | rms_hidden B:343 | fp16 | fp16 | elementwise | | |
| F2 | input Hadamard (5.2M) | ANE | grouped 1x1 conv B:292 | +-1/32 palettized | fp16 x fp16 | NE conv | | |
| F3 | gate, up 5120 -> 17408 in 8192 + 9216 chunks (178M) | ANE | QConv B:246 (C:38) | vector 2x64 + per-channel scale: 6-bit index per weight pair (3 bits per weight), 64x2 fp16 palette | fp16 x fp16 (palette expanded) | NE conv | V8_SPLIT | |
| F4 | silu(gate) x up | ANE | B:385 | | fp16 tanh form | elementwise | MLP_SILU | |
| F5 | mid Hadamard (17.8M) | ANE | grouped conv C:42 | +-1/32 | fp16 x fp16 | NE conv | | |
| F6 | down 17408 -> 5120 (89M) | ANE | QConv C:40 | as F3 | fp16 x fp16 | NE conv | | |

Totals: GPU projections about 7.2G MAC per token (48 DeltaNet x 115M + 16 attention x 105M, about 27%), all INT8 x
INT8 except the rank-64 corrections and a / b; the FFN about 18.6G (64 x 290M, about 71%), all FP16 on the ANE; the
attention core grows with the context (Q.K INT8, P.V FP16). No INT4 product anywhere today.
