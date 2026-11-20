
// ---- fold v2 (draft, 18 Nov 2026; appended to kernels_fold.metal after the stage B/C chain): wide PV, 64 rows x all
// 256 dims per threadgroup (8 simdgroups), so each score is read and exponentiated once (the 128-dim forms do it twice).
// Grid (1, 6T / 64, 4), 256 threads. n must be a multiple of KC.
template <uint KC>
inline float fold_fill4(device const half *row, threadgroup half *P, uint r, uint kq, uint j0, int lim, float m) {
    constexpr uint Q = KC / 4;
    device const half4 *src = (device const half4 *)(row + j0 + kq * Q);
    threadgroup half4 *dst = (threadgroup half4 *)(P + r * KC + kq * Q);
    int b = int(j0 + kq * Q);
    float s = 0;
    for (uint c = 0; c < Q / 4; c++) {
        float4 x = float4(src[c]) - m;
        int j = b + int(4 * c);
        float4 e = float4(j < lim ? exp(x.x) : 0.0f, j + 1 < lim ? exp(x.y) : 0.0f, j + 2 < lim ? exp(x.z) : 0.0f, j + 3 < lim ? exp(x.w) : 0.0f);
        dst[c] = half4(e); s += e.x + e.y + e.z + e.w;
    }
    return s;
}
template <uint KC>
inline void fold_pv_wide(device half *S, device half *Vc, device float *O, device const float *MX, device float *st,
                         uint Lmax, uint a, uint TK, uint n, uint p0, uint3 tg, uint tid, threadgroup half *Pt) {
    uint rt = tg.y, gh = tg.z;
    uint r = tid >> 2, kq = tid & 3, R = gh * 6 * T + rt * 64 + r, nt = n / 64;
    float m = -INFINITY;
    for (uint j = kq; j < nt; j += 4) m = max(m, MX[R * (TK / 64) + j]);
    m = max(m, simd_shuffle_xor(m, 1)); m = max(m, simd_shuffle_xor(m, 2));
    int lim = int(p0 + (R % T) + 1) - int(a);
    constexpr auto d = matmul2d_descriptor(64, 256, KC, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<8>> op;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(Pt, dextents<int32_t, 2>(KC, 64));
    auto cO = op.template get_destination_cooperative_tensor<decltype(tP), tensor<device half, dextents<int32_t, 2>, tensor_inline>, float>();
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0;
    device const half *row = S + R * TK;
    uint nc = n / KC;
    float sum = 0;
    for (uint c = 0; c < nc; c++) {
        sum += fold_fill4<KC>(row, Pt, r, kq, c * KC, lim, m);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tV(Vc + (gh * Lmax + a + c * KC) * 256, dextents<int32_t, 2>(256, KC),
                                                                    array<int32_t, 2>{1, 256});
        op.run(tP, tV, cO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        O[(gh * 6 * T + rt * 64 + ix[1]) * 256 + ix[0]] = cO[i];
    }
    sum += simd_shuffle_xor(sum, 1); sum += simd_shuffle_xor(sum, 2);
    if (kq == 0) { st[2 * R] = m; st[2 * R + 1] = sum; }
}
kernel void attn_pv_wm8_w64(FOLD_PV_ARGS) { threadgroup half Pt[64 * 64]; fold_pv_wide<64>(S, Vc, O, MX, st, Lmax, a, TK, n, p0, tg, tid, Pt); }
kernel void attn_pv_wm8_w128(FOLD_PV_ARGS) { threadgroup half Pt[64 * 128]; fold_pv_wide<128>(S, Vc, O, MX, st, Lmax, a, TK, n, p0, tg, tid, Pt); }

// online form (draft): QK unchanged (attn_qk8isk, no tile maxima); PV keeps a running row max per KC-key chunk and
// rescales its accumulators and sum when the max grows (alpha per row through threadgroup memory). Grid (2, 6T / 64, 4),
// 128 threads, KC = 64.
kernel void attn_pv_om8_64(device half *S [[buffer(0)]], device half *Vc [[buffer(1)]], device float *O [[buffer(2)]],
                           device float *st [[buffer(4)]], constant uint &Lmax [[buffer(5)]], constant uint &a [[buffer(6)]],
                           constant uint &TK [[buffer(7)]], constant uint &n [[buffer(8)]], constant uint &p0 [[buffer(9)]],
                           uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half Pt[64 * 64];
    threadgroup float A[64];
    uint dt = tg.x, rt = tg.y, gh = tg.z;
    uint r = tid >> 1, kh = tid & 1, R = gh * 6 * T + rt * 64 + r;
    int lim = int(p0 + (R % T) + 1) - int(a);
    constexpr auto d = matmul2d_descriptor(64, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(Pt, dextents<int32_t, 2>(64, 64));
    auto cO = op.template get_destination_cooperative_tensor<decltype(tP), tensor<device half, dextents<int32_t, 2>, tensor_inline>, float>();
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0;
    device const half *row = S + R * TK;
    float m = -INFINITY, sum = 0;
    for (uint j0 = 0; j0 < n; j0 += 64) {
        device const half4 *src = (device const half4 *)(row + j0 + kh * 32);
        int b = int(j0 + kh * 32);
        float4 x[8];
        float cm = -INFINITY;
        for (int c = 0; c < 8; c++) {
            float4 v = float4(src[c]); int j = b + 4 * c;
            v = float4(j < lim ? v.x : -INFINITY, j + 1 < lim ? v.y : -INFINITY, j + 2 < lim ? v.z : -INFINITY, j + 3 < lim ? v.w : -INFINITY);
            x[c] = v; cm = max(cm, max(max(v.x, v.y), max(v.z, v.w)));
        }
        cm = max(cm, simd_shuffle_xor(cm, 1));
        float mn = max(m, cm);
        float al = (mn == -INFINITY || m == mn) ? 1.0f : exp(m - mn);   // m = -inf: nothing accumulated yet, any factor
        threadgroup half4 *dst = (threadgroup half4 *)(Pt + r * 64 + kh * 32);
        float s = 0;
        for (int c = 0; c < 8; c++) {
            float4 e = (mn == -INFINITY) ? float4(0.0f) : exp(x[c] - mn);
            dst[c] = half4(e); s += e.x + e.y + e.z + e.w;
        }
        sum = sum * al + s; m = mn;
        if (kh == 0) A[r] = al;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
            if (!cO.is_valid_element(i)) continue;
            auto ix = cO.get_multidimensional_index(i);
            cO[i] *= A[ix[1]];
        }
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tV(Vc + (gh * Lmax + a + j0) * 256 + dt * 128, dextents<int32_t, 2>(128, 64),
                                                                    array<int32_t, 2>{1, 256});
        op.run(tP, tV, cO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        O[(gh * 6 * T + rt * 64 + ix[1]) * 256 + dt * 128 + ix[0]] = cO[i];
    }
    sum += simd_shuffle_xor(sum, 1);
    if (dt == 0 && kh == 0) { st[2 * R] = m; st[2 * R + 1] = sum; }
}
