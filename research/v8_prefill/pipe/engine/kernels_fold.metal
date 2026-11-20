
// ---- softmax folded into QK / PV on the served KV8 + K8V16 path (FOLD=..., 18 Nov 2026; appended to kernels.metal
// by gpu_v8_fold.swift). The plain path is attn_qk8isk -> attn_softmax8n (reads the scores, writes P) -> attn_pv_w
// (reads P). Folded: attn_qk8iskm also writes each 64-key tile's masked row maximum; attn_pv_wm8_* reads the scores,
// computes exp(s - m) while loading each key chunk into threadgroup memory and accumulates the row sums. Same maths as
// attn_softmax8n (m = the row max over the tile's valid keys; P rounded to fp16; sums in fp32), two score passes fewer.

// attn_qk8iskm = attn_qk8isk + MX[row][tile] (row max of the fp16-rounded scaled scores over the valid keys of the
// 64-key tile); am = (a: absolute first key of the 2048-key tile, p0). Grid (n / 64, 6T / 64, 4), 128 threads.
kernel void attn_qk8iskm(device int8_t *Q8 [[buffer(0)]], device int8_t *K8 [[buffer(1)]], device half *S [[buffer(2)]],
                         constant uint2 &kst [[buffer(3)]], constant uint &TK [[buffer(4)]], device const float *QS [[buffer(5)]],
                         device const half *KS [[buffer(6)]], constant uint &ssh [[buffer(7)]], device float *MX [[buffer(8)]],
                         constant uint2 &am [[buffer(9)]],
                         uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half St[64 * 64];
    uint kt = tg.x, rt = tg.y, gh = tg.z, r0 = gh * 6 * T + rt * 64;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tQ(Q8 + r0 * 256, dextents<int32_t, 2>(256, 64));
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK(K8 + gh * kst.x + kt * 64, dextents<int32_t, 2>(64, 256),
                                                                  array<int32_t, 2>{1, int(kst.y)});
    constexpr auto d = matmul2d_descriptor(64, 64, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    auto cI = op.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
    op.run(tQ, tK, cI);
    device const half *ks = KS + gh * ssh + kt * 64;
    for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
        if (!cI.is_valid_element(i)) continue;
        auto ix = cI.get_multidimensional_index(i);
        uint key = ix[0], row = ix[1];
        half s = half(float(cI[i]) * QS[r0 + row] * float(ks[key]));
        S[(r0 + row) * TK + kt * 64 + key] = s;
        St[row * 64 + key] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = tid >> 1, kh = tid & 1, R = r0 + r;
    int lim = int(am.y + (R % T) + 1) - int(am.x + kt * 64 + kh * 32);   // valid keys among this thread's 32
    threadgroup const half4 *row = (threadgroup const half4 *)(St + r * 64 + kh * 32);
    float m = -INFINITY;
    for (int c = 0; c < 8; c++) {
        float4 x = float4(row[c]);
        int j = 4 * c;
        m = max(m, max(max(j < lim ? x.x : -INFINITY, j + 1 < lim ? x.y : -INFINITY), max(j + 2 < lim ? x.z : -INFINITY, j + 3 < lim ? x.w : -INFINITY)));
    }
    m = max(m, simd_shuffle_xor(m, 1));
    if (kh == 0) MX[R * (TK / 64) + kt] = m;
}

// exp(s - m) of this thread's half of a KC-key chunk (row r, keys j0 + kh * KC / 2 ...) into P[r][..]; returns the sum
template <uint KC>
inline float fold_fill(device const half *row, threadgroup half *P, uint r, uint kh, uint j0, int lim, float m) {
    constexpr uint H = KC / 2;
    device const half4 *src = (device const half4 *)(row + j0 + kh * H);
    threadgroup half4 *dst = (threadgroup half4 *)(P + r * KC + kh * H);
    int b = int(j0 + kh * H);
    float s = 0;
    for (uint c = 0; c < H / 4; c++) {
        float4 x = float4(src[c]) - m;
        int j = b + int(4 * c);
        float4 e = float4(j < lim ? exp(x.x) : 0.0f, j + 1 < lim ? exp(x.y) : 0.0f, j + 2 < lim ? exp(x.z) : 0.0f, j + 3 < lim ? exp(x.w) : 0.0f);
        dst[c] = half4(e); s += e.x + e.y + e.z + e.w;
    }
    return s;
}

// O tile (64 rows x 128 dims) = exp(S - m) x V over the tile's n keys in KC-key chunks; DB: two chunk buffers (one
// barrier per chunk), else one (two barriers). Stats (m, sum) for attn_combine. n must be a multiple of KC.
template <uint KC, bool DB>
inline void fold_pv(device half *S, device half *Vc, device float *O, device const float *MX, device float *st,
                    uint Lmax, uint a, uint TK, uint n, uint p0, uint3 tg, uint tid, threadgroup half *Pt) {
    uint dt = tg.x, rt = tg.y, gh = tg.z;
    uint r = tid >> 1, kh = tid & 1, R = gh * 6 * T + rt * 64 + r, nt = n / 64;
    float m = -INFINITY;
    for (uint j = kh; j < nt; j += 2) m = max(m, MX[R * (TK / 64) + j]);
    m = max(m, simd_shuffle_xor(m, 1));
    int lim = int(p0 + (R % T) + 1) - int(a);
    constexpr auto d = matmul2d_descriptor(64, 128, KC, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP0(Pt, dextents<int32_t, 2>(KC, 64));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP1(Pt + (DB ? 64 * KC : 0), dextents<int32_t, 2>(KC, 64));
    auto cO = op.template get_destination_cooperative_tensor<decltype(tP0), tensor<device half, dextents<int32_t, 2>, tensor_inline>, float>();
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0;
    device const half *row = S + R * TK;
    uint nc = n / KC;
    float sum = fold_fill<KC>(row, Pt, r, kh, 0, lim, m);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = 0; c < nc; c++) {
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tV(Vc + (gh * Lmax + a + c * KC) * 256 + dt * 128, dextents<int32_t, 2>(128, KC),
                                                                    array<int32_t, 2>{1, 256});
        if (DB) {
            if (c & 1) op.run(tP1, tV, cO); else op.run(tP0, tV, cO);
            if (c + 1 < nc) sum += fold_fill<KC>(row, Pt + ((c + 1) & 1) * 64 * KC, r, kh, (c + 1) * KC, lim, m);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            op.run(tP0, tV, cO);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (c + 1 < nc) { sum += fold_fill<KC>(row, Pt, r, kh, (c + 1) * KC, lim, m); threadgroup_barrier(mem_flags::mem_threadgroup); }
        }
    }
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        O[(gh * 6 * T + rt * 64 + ix[1]) * 256 + dt * 128 + ix[0]] = cO[i];
    }
    sum += simd_shuffle_xor(sum, 1);
    if (dt == 0 && kh == 0) { st[2 * R] = m; st[2 * R + 1] = sum; }
}

#define FOLD_PV_ARGS device half *S [[buffer(0)]], device half *Vc [[buffer(1)]], device float *O [[buffer(2)]], \
    device const float *MX [[buffer(3)]], device float *st [[buffer(4)]], constant uint &Lmax [[buffer(5)]], \
    constant uint &a [[buffer(6)]], constant uint &TK [[buffer(7)]], constant uint &n [[buffer(8)]], \
    constant uint &p0 [[buffer(9)]], uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
// grid (2, 6T / 64, 4), 128 threads
kernel void attn_pv_wm8_64(FOLD_PV_ARGS) { threadgroup half Pt[2 * 64 * 64]; fold_pv<64, true>(S, Vc, O, MX, st, Lmax, a, TK, n, p0, tg, tid, Pt); }
kernel void attn_pv_wm8_64s(FOLD_PV_ARGS) { threadgroup half Pt[64 * 64]; fold_pv<64, false>(S, Vc, O, MX, st, Lmax, a, TK, n, p0, tg, tid, Pt); }
kernel void attn_pv_wm8_128(FOLD_PV_ARGS) { threadgroup half Pt[64 * 128]; fold_pv<128, false>(S, Vc, O, MX, st, Lmax, a, TK, n, p0, tg, tid, Pt); }
