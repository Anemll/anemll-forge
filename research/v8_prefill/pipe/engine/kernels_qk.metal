// ---- Q.K with a staged epilogue (QKE=tg, 19 Nov 2026; appended after kernels_f8.metal by gpu_v8_f8q.swift) ----
// attn_qk8isk (INT8 Q x INT8 K from the ANE cache layout, S = acc x QS x KS, fp16 scores) with the accumulators stored
// to threadgroup memory as a [row][key] int32 tile and the scaled scores written as coalesced half4 rows, instead of
// one scattered 2-byte store per cooperative-tensor element. The same arithmetic in the same order ((acc x QS) x KS,
// fp32, then half): the scores are bit-identical. Same arguments, grid (n / 64, 6T / 64, 4) and 128 threads.
kernel void attn_qk8isk_tg(device int8_t *Q8 [[buffer(0)]], device int8_t *K8 [[buffer(1)]], device half *S [[buffer(2)]],
                           constant uint2 &kst [[buffer(3)]], constant uint &TK [[buffer(4)]], device const float *QS [[buffer(5)]],
                           device const half *KS [[buffer(6)]], constant uint &ssh [[buffer(7)]],
                           uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int tile[64 * 64];
    uint kt = tg.x, rt = tg.y, gh = tg.z, r0 = gh * 6 * T + rt * 64;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tQ(Q8 + r0 * 256, dextents<int32_t, 2>(256, 64));
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK(K8 + gh * kst.x + kt * 64, dextents<int32_t, 2>(64, 256),
                                                                  array<int32_t, 2>{1, int(kst.y)});
    constexpr auto d = matmul2d_descriptor(64, 64, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    auto cI = op.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
    op.run(tQ, tK, cI);
    tensor<threadgroup int32_t, dextents<int32_t, 2>, tensor_inline> tT(tile, dextents<int32_t, 2>(64, 64));
    cI.store(tT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const half *ks = KS + gh * ssh + kt * 64;
    for (uint i = tid; i < 64 * 16; i += 128) {
        uint row = i >> 4, c = (i & 15) * 4;
        float4 v = float4(*(threadgroup int4 *)(tile + row * 64 + c)) * QS[r0 + row];
        v *= float4(*(device const half4 *)(ks + c));
        *(device half4 *)(S + (r0 + row) * TK + kt * 64 + c) = half4(v);
    }
}
