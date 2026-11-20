// ---- F8=x with the softmax folded into P.V (F8F=w64|w64d|w128, 18 Nov 2026; appended after kernels_f8.metal by
// gpu_v8_f8f.swift). attn_qk8iskm (kernels_fold.metal) writes the scores and each 64-key tile's masked row maximum;
// attn_pv_f8m_* reads the scores, computes exp(s - m) in half precision on load (as attn_softmax8n_f8x), scales by 256,
// converts to e4m3 into threadgroup memory and runs FP8 x FP8 against the F8=v V copy (e4m3 of v x vsc, layout
// [kv head][dim half][Lmax][128]); sums in fp32; stats (m, 256 vsc sum) as attn_softmax8n_f8x, for attn_combine.
// The separate softmax pass (score read for the max, score read for exp, P8 write) and P.V's P8 read are gone.
// Wide: one threadgroup = 64 rows x both 128-dim halves (8 simdgroups, 256 threads; two matmuls per key chunk on one P
// chunk), so each score is read and exponentiated once. Grid (1, 6T / 64, 4). n must be a multiple of KC.

// exp(s - m) x 256 -> e4m3 of this thread's quarter of a KC-key chunk (row r, keys j0 + kq KC / 4 ...); returns sum(e)
template <uint KC>
inline float f8m_fill(device const half *row, threadgroup uchar *P, uint r, uint kq, uint j0, int lim, half m16, bool none) {
    constexpr uint Q = KC / 4;
    device const half4 *src = (device const half4 *)(row + j0 + kq * Q);
    threadgroup uchar4 *dst = (threadgroup uchar4 *)(P + r * KC + kq * Q);
    int b = int(j0 + kq * Q);
    if (none || b >= lim) { for (uint c = 0; c < Q / 4; c++) dst[c] = uchar4(0); return 0.0f; }
    bool full = b + int(Q) <= lim;
    float s = 0;
    for (uint c = 0; c < Q / 4; c++) {
        half4 e = exp(src[c] - m16);
        if (!full) {
            int j = b + int(4 * c);
            e = half4(j < lim ? e.x : 0.0h, j + 1 < lim ? e.y : 0.0h, j + 2 < lim ? e.z : 0.0h, j + 3 < lim ? e.w : 0.0h);
        }
        half4 e2 = e * 256.0h;
        dst[c] = uchar4(e4m3_h(e2.x), e4m3_h(e2.y), e4m3_h(e2.z), e4m3_h(e2.w));
        s += float(e.x) + float(e.y) + float(e.z) + float(e.w);
    }
    return s;
}

template <uint KC, bool DB>
inline void pv_f8m_wide(device half *S, device uchar *Vf, device float *O, device const float *MX, device float *st,
                        uint Lmax, uint a, uint TK, uint n, uint p0, float vsc, uint3 tg, uint tid, threadgroup uchar *Pt) {
    uint rt = tg.y, gh = tg.z;
    uint r = tid >> 2, kq = tid & 3, R = gh * 6 * T + rt * 64 + r, nt = n / 64;
    float m = -INFINITY;
    for (uint j = kq; j < nt; j += 4) m = max(m, MX[R * (TK / 64) + j]);
    m = max(m, simd_shuffle_xor(m, 1)); m = max(m, simd_shuffle_xor(m, 2));
    bool none = (m == -INFINITY);
    half m16 = none ? 0.0h : half(m);
    int lim = int(p0 + (R % T) + 1) - int(a);
    using TF = tensor<device metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline>;
    using TT = tensor<threadgroup metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline>;
    TT tP0((typename TT::data_handle_type)Pt, dextents<int32_t, 2>(KC, 64));
    TT tP1((typename TT::data_handle_type)(Pt + (DB ? 64 * KC : 0)), dextents<int32_t, 2>(KC, 64));
    constexpr auto d = matmul2d_descriptor(64, 128, KC, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<8>> op;
    auto c0 = op.template get_destination_cooperative_tensor<TT, TF, float>();
    auto c1 = op.template get_destination_cooperative_tensor<TT, TF, float>();
    for (uint16_t i = 0; i < c0.get_capacity(); ++i) if (c0.is_valid_element(i)) c0[i] = 0;
    for (uint16_t i = 0; i < c1.get_capacity(); ++i) if (c1.is_valid_element(i)) c1[i] = 0;
    device const half *row = S + R * TK;
    device uchar *v0 = Vf + ((gh * 2) * Lmax + a) * 128, *v1 = Vf + ((gh * 2 + 1) * Lmax + a) * 128;
    uint nc = n / KC;
    float sum = f8m_fill<KC>(row, Pt, r, kq, 0, lim, m16, none);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = 0; c < nc; c++) {
        TF tV0((typename TF::data_handle_type)(v0 + c * KC * 128), dextents<int32_t, 2>(128, KC));
        TF tV1((typename TF::data_handle_type)(v1 + c * KC * 128), dextents<int32_t, 2>(128, KC));
        if (DB) {
            if (c & 1) { op.run(tP1, tV0, c0); op.run(tP1, tV1, c1); } else { op.run(tP0, tV0, c0); op.run(tP0, tV1, c1); }
            if (c + 1 < nc) sum += f8m_fill<KC>(row, Pt + ((c + 1) & 1) * 64 * KC, r, kq, (c + 1) * KC, lim, m16, none);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            op.run(tP0, tV0, c0); op.run(tP0, tV1, c1);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (c + 1 < nc) { sum += f8m_fill<KC>(row, Pt, r, kq, (c + 1) * KC, lim, m16, none); threadgroup_barrier(mem_flags::mem_threadgroup); }
        }
    }
    for (uint16_t i = 0; i < c0.get_capacity(); ++i) {
        if (!c0.is_valid_element(i)) continue;
        auto ix = c0.get_multidimensional_index(i);
        O[(gh * 6 * T + rt * 64 + ix[1]) * 256 + ix[0]] = c0[i];
    }
    for (uint16_t i = 0; i < c1.get_capacity(); ++i) {
        if (!c1.is_valid_element(i)) continue;
        auto ix = c1.get_multidimensional_index(i);
        O[(gh * 6 * T + rt * 64 + ix[1]) * 256 + 128 + ix[0]] = c1[i];
    }
    sum += simd_shuffle_xor(sum, 1); sum += simd_shuffle_xor(sum, 2);
    if (kq == 0) { st[2 * R] = m; st[2 * R + 1] = sum * 256.0f * vsc; }
}

#define F8M_ARGS device half *S [[buffer(0)]], device uchar *Vf [[buffer(1)]], device float *O [[buffer(2)]], \
    device const float *MX [[buffer(3)]], device float *st [[buffer(4)]], constant uint &Lmax [[buffer(5)]], \
    constant uint &a [[buffer(6)]], constant uint &TK [[buffer(7)]], constant uint &n [[buffer(8)]], \
    constant uint &p0 [[buffer(9)]], constant float &vsc [[buffer(10)]], \
    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
kernel void attn_pv_f8m_w64(F8M_ARGS) { threadgroup uchar Pt[64 * 64]; pv_f8m_wide<64, false>(S, Vf, O, MX, st, Lmax, a, TK, n, p0, vsc, tg, tid, Pt); }
kernel void attn_pv_f8m_w64d(F8M_ARGS) { threadgroup uchar Pt[2 * 64 * 64]; pv_f8m_wide<64, true>(S, Vf, O, MX, st, Lmax, a, TK, n, p0, vsc, tg, tid, Pt); }
kernel void attn_pv_f8m_w128(F8M_ARGS) { threadgroup uchar Pt[64 * 128]; pv_f8m_wide<128, false>(S, Vf, O, MX, st, Lmax, a, TK, n, p0, vsc, tg, tid, Pt); }
kernel void attn_pv_f8m_w128d(F8M_ARGS) { threadgroup uchar Pt[2 * 64 * 128]; pv_f8m_wide<128, true>(S, Vf, O, MX, st, Lmax, a, TK, n, p0, vsc, tg, tid, Pt); }
