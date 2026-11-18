// Decode-verify attention on the GPU (16 Nov 2026): dec_prep (INT8 Q, the 8 new K / V rows into the ANE cache segments),
// dec_attn (flash decoding over the segmented INT8 cache, causal among the 8 rows: pipe/unit/decode_attn_seg.swift),
// dec_combine_gate (chunks merged, x sigmoid(gate), fp16 channel-major for part B). Used by dec_gpu.swift.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#define NSEG 5
#ifndef DEC_SM
#define DEC_SM 0   // softmax research switch (DEC_SM env of libdec_gpu): 0 FP32 exp (served), 1 no exp (timing bound), 2 FP8 probabilities
#endif
struct DPrm { uint n, p0, ck, nc; uint start[NSEG]; uint rows[NSEG]; };
// the ANE buffers' strides in halves: q, kt, vt, g (head, row; d dense), o (channel; t dense)
struct DSt { uint qh, qt, kh, kt, vh, vt, gh, gt, oc; };
#ifndef DEC_M
#define DEC_M 16
#define DEC_NSG 4
#define DEC_QSP 3
#endif
#ifndef DEC_SUB
#define DEC_SUB 256
#endif
constant constexpr int NSG = DEC_NSG, SUB = DEC_SUB, M = DEC_M, QSP = DEC_QSP, NT = 32 * NSG, TPR = NT / M, KPT = SUB / TPR;   // research copy: overridable

kernel void dec_attn(device const int8_t *Q8 [[buffer(0)]], device const float *QS [[buffer(1)]],
                     device const int8_t *K0 [[buffer(2)]], device const int8_t *K1 [[buffer(3)]], device const int8_t *K2 [[buffer(4)]],
                     device const int8_t *K3 [[buffer(5)]], device const int8_t *K4 [[buffer(6)]],
                     device const half *KS0 [[buffer(7)]], device const half *KS1 [[buffer(8)]], device const half *KS2 [[buffer(9)]],
                     device const half *KS3 [[buffer(10)]], device const half *KS4 [[buffer(11)]],
                     device const int8_t *V0 [[buffer(12)]], device const int8_t *V1 [[buffer(13)]], device const int8_t *V2 [[buffer(14)]],
                     device const int8_t *V3 [[buffer(15)]], device const int8_t *V4 [[buffer(16)]],
                     device const half *VS0 [[buffer(17)]], device const half *VS1 [[buffer(18)]], device const half *VS2 [[buffer(19)]],
                     device const half *VS3 [[buffer(20)]], device const half *VS4 [[buffer(21)]],
                     device float *OP [[buffer(22)]], device float *MP [[buffer(23)]], device float *LP [[buffer(24)]],
                     constant DPrm &p [[buffer(25)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    threadgroup half S[M * SUB]; threadgroup float alpha[M], qs[M], ks[SUB], vs[SUB];
    uint c = tg.x, qi = tg.y, gh = tg.z, r0 = qi * M;
    uint k0 = p.start[0] + c * p.ck, k1 = min(k0 + p.ck, p.n);   // from the first registered segment (0: all keys)
    uint sg = 0;
    for (uint i = 1; i < NSEG; ++i) if (p.rows[i] && k0 >= p.start[i]) sg = i;   // a chunk never crosses a segment
    uint R = p.rows[sg], s0 = p.start[sg];
    device const int8_t *K = sg == 0 ? K0 : sg == 1 ? K1 : sg == 2 ? K2 : sg == 3 ? K3 : K4;
    device const half *KS = sg == 0 ? KS0 : sg == 1 ? KS1 : sg == 2 ? KS2 : sg == 3 ? KS3 : KS4;
    device const int8_t *V = sg == 0 ? V0 : sg == 1 ? V1 : sg == 2 ? V2 : sg == 3 ? V3 : V4;
    device const half *VS = sg == 0 ? VS0 : sg == 1 ? VS1 : sg == 2 ? VS2 : sg == 3 ? VS3 : VS4;
    uint slot = (gh * p.nc + c) * QSP + qi;
    if (t < M) qs[t] = QS[gh * 48 + r0 + t];
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tQ((device int8_t *)Q8 + (gh * 48 + r0) * 256, dextents<int32_t, 2>(256, M));
    constexpr auto dqk = matmul2d_descriptor(M, SUB, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<dqk, execution_simdgroups<NSG>> qk;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(S, dextents<int32_t, 2>(SUB, M), array<int32_t, 2>{1, SUB});
    constexpr auto dpv = matmul2d_descriptor(M, 256, SUB, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dpv, execution_simdgroups<NSG>> pv;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tV0((device int8_t *)V + gh * R * 256, dextents<int32_t, 2>(256, SUB),
                                                                   array<int32_t, 2>{1, 256});
    auto cO = pv.template get_destination_cooperative_tensor<decltype(tP), decltype(tV0), float>();
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0.0f;
    uint row = t / TPR, part = t % TPR;
    float m = -INFINITY, l = 0.0f;
#if DEC_SM == 7   // research bound: both matmuls (all K and V reads) with a fixed P tile: no score store, no softmax
    for (uint j = t; j < M * SUB; j += NT) S[j] = half(0.001f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int dummy = 0;
    for (uint kb = k0; kb < k1; kb += SUB) {
        uint lb = kb - s0;
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK((device int8_t *)K + gh * 256 * R + lb, dextents<int32_t, 2>(SUB, 256),
                                                                      array<int32_t, 2>{1, int(R)});
        auto cI = qk.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
        qk.run(tQ, tK, cI);
        for (uint16_t i = 0; i < cI.get_capacity(); ++i) if (cI.is_valid_element(i)) dummy += cI[i];
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tV((device int8_t *)V + (gh * R + lb) * 256, dextents<int32_t, 2>(256, SUB),
                                                                      array<int32_t, 2>{1, 256});
        pv.run(tP, tV, cO);
    }
    if (dummy == 123456789) l = 1.0f;
    m = 0.0f; l += 1.0f;
#else
    for (uint kb = k0; kb < k1; kb += SUB) {
        uint lb = kb - s0;                                    // segment-local row of the sub-tile
        for (uint j = t; j < SUB; j += NT) {
            bool in = kb + j < p.n;
            ks[j] = in ? float(KS[gh * R + lb + j]) : 0.0f; vs[j] = in ? float(VS[gh * R + lb + j]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK((device int8_t *)K + gh * 256 * R + lb, dextents<int32_t, 2>(SUB, 256),
                                                                      array<int32_t, 2>{1, int(R)});
        auto cI = qk.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
#if DEC_SM == 5   // research, timing only: no QK (no K reads), zero scores
        for (uint16_t i = 0; i < cI.get_capacity(); ++i) if (cI.is_valid_element(i)) cI[i] = 0;
#else
        qk.run(tQ, tK, cI);
#endif
        for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
            if (!cI.is_valid_element(i)) continue;
            auto ix = cI.get_multidimensional_index(i);
            uint key = kb + ix[0], tok = (r0 + ix[1]) & 7;    // the query's verify row
            bool vis = key < p.n && key <= p.p0 + tok;         // keys before p0 are all visible
            S[ix[1] * SUB + ix[0]] = vis ? half(float(cI[i]) * qs[ix[1]] * ks[ix[0]]) : half(-INFINITY);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half *Sr = S + row * SUB + part * KPT;
        float tm = -INFINITY;
        for (int j = 0; j < KPT; ++j) tm = max(tm, float(Sr[j]));
        for (int o = 1; o < TPR; o <<= 1) tm = max(tm, simd_shuffle_xor(tm, o));
        float mn = max(m, tm), a = (mn == -INFINITY) ? 1.0f : exp(m - mn), sum = 0.0f;
#if DEC_SM == 2   // research: FP8 e4m3 softmax probabilities (the ANE's sm8 step: e x 64 -> FP8 -> / 64), the sum over them
        for (int j = 0; j < KPT; j += 4) {
            float4 e4;
            for (int u = 0; u < 4; ++u) { float sv = float(Sr[j + u]); e4[u] = (sv == -INFINITY) ? 0.0f : exp(sv - mn); }
            e4 = unpack<float>(pack<metal_fp8_e4m3_format>(e4 * 64.0f)) * (1.0f / 64.0f);
            for (int u = 0; u < 4; ++u) { sum += e4[u]; Sr[j + u] = half(e4[u] * vs[part * KPT + j + u]); }
        }
#else
        for (int j = 0; j < KPT; ++j) {
            float sv = float(Sr[j]);
#if DEC_SM == 1   // research, timing bound only: no exp (wrong results)
            float e = (sv == -INFINITY) ? 0.0f : max(0.0f, 1.0f + (sv - mn)); sum += e;
#else
            float e = (sv == -INFINITY) ? 0.0f : exp(sv - mn); sum += e;
#endif
            Sr[j] = half(e * vs[part * KPT + j]);
        }
#endif
        for (int o = 1; o < TPR; o <<= 1) sum += simd_shuffle_xor(sum, o);
        l = l * a + sum; m = mn;
        if (part == 0) alpha[row] = a;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
            if (!cO.is_valid_element(i)) continue;
            auto ix = cO.get_multidimensional_index(i);
            cO[i] *= alpha[ix[1]];
        }
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tV((device int8_t *)V + (gh * R + lb) * 256, dextents<int32_t, 2>(256, SUB),
                                                                      array<int32_t, 2>{1, 256});
#if DEC_SM != 4   // 4: research, timing only: no PV (no V reads)
        pv.run(tP, tV, cO);
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#endif
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        OP[(slot * M + ix[1]) * 256 + ix[0]] = cO[i];
    }
    if (part == 0) { MP[slot * M + row] = m; LP[slot * M + row] = l; }
}

// decode prep: part A's fp16 outputs -> INT8 Q + scale (q / 16 folded into the scale), and the 8 new K / V rows as INT8
// codes + fp16 scales into their cache segments (each row picks its segment: a window may cross a boundary). The K / V
// rounding is the host's (qwen38_kv_cache.quantize_values): scale = fp16(max(max|x| / 127, 1e-6)), code = rint(x / scale).
// q, g: [24][8][256]; kt, vt: [4][8][256]. Grid (32 heads: 24 q + 4 k + 4 v, 8 rows) x 256 threads.
kernel void dec_prep(device const half *qh [[buffer(0)]], device const half *kt [[buffer(1)]], device const half *vt [[buffer(2)]],
                     device int8_t *Q8 [[buffer(3)]], device float *QS [[buffer(4)]],
                     device int8_t *K0 [[buffer(5)]], device int8_t *K1 [[buffer(6)]], device int8_t *K2 [[buffer(7)]],
                     device int8_t *K3 [[buffer(8)]], device int8_t *K4 [[buffer(9)]],
                     device half *KS0 [[buffer(10)]], device half *KS1 [[buffer(11)]], device half *KS2 [[buffer(12)]],
                     device half *KS3 [[buffer(13)]], device half *KS4 [[buffer(14)]],
                     device int8_t *V0 [[buffer(15)]], device int8_t *V1 [[buffer(16)]], device int8_t *V2 [[buffer(17)]],
                     device int8_t *V3 [[buffer(18)]], device int8_t *V4 [[buffer(19)]],
                     device half *VS0 [[buffer(20)]], device half *VS1 [[buffer(21)]], device half *VS2 [[buffer(22)]],
                     device half *VS3 [[buffer(23)]], device half *VS4 [[buffer(24)]],
                     constant DPrm &p [[buffer(25)]], constant DSt &st [[buffer(26)]],
                     uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                     uint sl [[thread_index_in_simdgroup]], uint sgi [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    uint h = g.x, t = g.y;
    float v = h < 24 ? float(qh[h * st.qh + t * st.qt + d]) : h < 28 ? float(kt[(h - 24) * st.kh + t * st.kt + d]) : float(vt[(h - 28) * st.vh + t * st.vt + d]);
    float am = simd_max(fabs(v));
    if (sl == 0) part[sgi] = am;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    if (h < 24) {
        float sq = m > 0 ? m / 127.0f : 1.0f;
        Q8[(h * 8 + t) * 256 + d] = int8_t(clamp(rint(v / sq), -127.0f, 127.0f));
        if (d == 0) QS[h * 8 + t] = sq * 0.0625f;
        return;
    }
    uint pos = p.p0 + t, sg = 0;
    for (uint i = 1; i < NSEG; ++i) if (p.rows[i] && pos >= p.start[i]) sg = i;
    uint R = p.rows[sg], r = pos - p.start[sg];
    half sc = half(max(m / 127.0f, 1e-6f));
    int8_t code = int8_t(clamp(rint(precise::divide(v, float(sc))), -127.0f, 127.0f));   // exact division: the host quantize bit for bit
    if (h < 28) {
        uint kh = h - 24;
        device int8_t *K = sg == 0 ? K0 : sg == 1 ? K1 : sg == 2 ? K2 : sg == 3 ? K3 : K4;
        device half *KS = sg == 0 ? KS0 : sg == 1 ? KS1 : sg == 2 ? KS2 : sg == 3 ? KS3 : KS4;
        K[kh * 256 * R + d * R + r] = code; if (d == 0) KS[kh * R + r] = sc;
    } else {
        uint vh = h - 28;
        device int8_t *V = sg == 0 ? V0 : sg == 1 ? V1 : sg == 2 ? V2 : sg == 3 ? V3 : V4;
        device half *VS = sg == 0 ? VS0 : sg == 1 ? VS1 : sg == 2 ? VS2 : sg == 3 ? VS3 : VS4;
        V[(vh * R + r) * 256 + d] = code; if (d == 0) VS[vh * R + r] = sc;
    }
}
// merge the chunks, x sigmoid(gate) -> o (fp16, channel-major [h * 256 + d][8]: part B's input). g: [24][8][256] fp16.
kernel void dec_combine_gate(device const float *OP [[buffer(0)]], device const float *MP [[buffer(1)]], device const float *LP [[buffer(2)]],
                             device half *o [[buffer(3)]], constant DPrm &p [[buffer(4)]], device const half *gt [[buffer(5)]], constant DSt &st [[buffer(6)]],
                             uint3 gid [[thread_position_in_grid]]) {
    uint d = gid.x, j = gid.y, gh = gid.z, qi = j / M, r = j % M, h = gh * 6 + j / 8, t = j % 8;
    float Mx = -INFINITY;
    for (uint c = 0; c < p.nc; ++c) Mx = max(Mx, MP[((gh * p.nc + c) * QSP + qi) * M + r]);
    float num = 0.0f, den = 0.0f;
    for (uint c = 0; c < p.nc; ++c) {
        uint s = (gh * p.nc + c) * QSP + qi;
        float mc = MP[s * M + r];
        if (mc == -INFINITY) continue;
        float w = exp(mc - Mx);
        num += w * OP[(s * M + r) * 256 + d]; den += w * LP[s * M + r];
    }
    float gv = float(gt[h * st.gh + t * st.gt + d]);
    o[(h * 256 + d) * st.oc + t] = half(num / den / (1.0f + exp(-gv)));
}
// Partial result for a split attention (the ANE takes the first keys): this GPU part's softmax-normalized output and
// its log-sum-exp per (query head, row), FP32, no gate: PO[(h * 8 + t) * 256 + d], PO[24 * 8 * 256 + h * 8 + t].
kernel void dec_combine_part(device const float *OP [[buffer(0)]], device const float *MP [[buffer(1)]], device const float *LP [[buffer(2)]],
                             device float *PO [[buffer(3)]], constant DPrm &p [[buffer(4)]],
                             uint3 gid [[thread_position_in_grid]]) {
    uint d = gid.x, j = gid.y, gh = gid.z, qi = j / M, r = j % M, h = gh * 6 + j / 8, t = j % 8;
    float Mx = -INFINITY;
    for (uint c = 0; c < p.nc; ++c) Mx = max(Mx, MP[((gh * p.nc + c) * QSP + qi) * M + r]);
    float num = 0.0f, den = 0.0f;
    for (uint c = 0; c < p.nc; ++c) {
        uint s = (gh * p.nc + c) * QSP + qi;
        float mc = MP[s * M + r];
        if (mc == -INFINITY) continue;
        float w = exp(mc - Mx);
        num += w * OP[(s * M + r) * 256 + d]; den += w * LP[s * M + r];
    }
    PO[(h * 8 + t) * 256 + d] = num / den;
    if (d == 0) PO[24 * 8 * 256 + h * 8 + t] = Mx + log(den);
}
// GPU-side wait for a CPU flag (the armed work starts as soon as the CPU stores v; the GPU never idles in between),
// bounded so a missed store cannot hang the GPU (res[0] = loop count, res[1] = 1 on timeout).
kernel void dec_flag_wait(device atomic_uint *f [[buffer(0)]], constant uint &v [[buffer(1)]], device uint *res [[buffer(2)]],
                          uint t [[thread_index_in_threadgroup]]) {
    if (t != 0) return;
    uint n = 0;
    while (atomic_load_explicit(f, memory_order_relaxed) < v && n < (1u << 27)) n++;
    res[0] = n; res[1] = n >= (1u << 27) ? 1u : 0u;
}
kernel void dec_flag_set(device atomic_uint *f [[buffer(0)]], constant uint &v [[buffer(1)]]) {
    atomic_store_explicit(f, v, memory_order_relaxed);
}

// ---- research (17 Nov 2026): the same attention with the softmax kept in registers (no score / P tile in threadgroup
// memory): QK into a cooperative tensor, row max / sum with mpp reduce_rows, P handed to the PV matmul as its
// cooperative left input. SUBR keys per matmul step (no threadgroup tile limits it any more). Same buffers and partial
// outputs as dec_attn (dec_combine_gate / dec_combine_part unchanged).
#ifndef SUBR
#define SUBR 256
#endif
kernel void dec_attn_sg(device const int8_t *Q8 [[buffer(0)]], device const float *QS [[buffer(1)]],
                     device const int8_t *K0 [[buffer(2)]], device const int8_t *K1 [[buffer(3)]], device const int8_t *K2 [[buffer(4)]],
                     device const int8_t *K3 [[buffer(5)]], device const int8_t *K4 [[buffer(6)]],
                     device const half *KS0 [[buffer(7)]], device const half *KS1 [[buffer(8)]], device const half *KS2 [[buffer(9)]],
                     device const half *KS3 [[buffer(10)]], device const half *KS4 [[buffer(11)]],
                     device const int8_t *V0 [[buffer(12)]], device const int8_t *V1 [[buffer(13)]], device const int8_t *V2 [[buffer(14)]],
                     device const int8_t *V3 [[buffer(15)]], device const int8_t *V4 [[buffer(16)]],
                     device const half *VS0 [[buffer(17)]], device const half *VS1 [[buffer(18)]], device const half *VS2 [[buffer(19)]],
                     device const half *VS3 [[buffer(20)]], device const half *VS4 [[buffer(21)]],
                     device float *OP [[buffer(22)]], device float *MP [[buffer(23)]], device float *LP [[buffer(24)]],
                     constant DPrm &p [[buffer(25)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                     uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    // every simdgroup: its own flash-attention stream over a quarter of the chunk (own QK / PV matmuls, softmax in
    // registers via reduce_rows, own m / l / O); its partial goes to slot c * NSG + sgi (the combine merges nc * NSG)
    threadgroup float qs[M], ksb[NSG][SUBR], vsb[NSG][SUBR], alpha_s[NSG][M];
    uint c = tg.x, qi = tg.y, gh = tg.z, r0 = qi * M;
    uint k0 = p.start[0] + c * p.ck, k1 = min(k0 + p.ck, p.n);
    uint sg = 0;
    for (uint i = 1; i < NSEG; ++i) if (p.rows[i] && k0 >= p.start[i]) sg = i;
    uint R = p.rows[sg], s0 = p.start[sg];
    device const int8_t *K = sg == 0 ? K0 : sg == 1 ? K1 : sg == 2 ? K2 : sg == 3 ? K3 : K4;
    device const half *KS = sg == 0 ? KS0 : sg == 1 ? KS1 : sg == 2 ? KS2 : sg == 3 ? KS3 : KS4;
    device const int8_t *V = sg == 0 ? V0 : sg == 1 ? V1 : sg == 2 ? V2 : sg == 3 ? V3 : V4;
    device const half *VS = sg == 0 ? VS0 : sg == 1 ? VS1 : sg == 2 ? VS2 : sg == 3 ? VS3 : VS4;
    if (t < M) qs[t] = QS[gh * 48 + r0 + t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint ncs = p.nc * NSG, slot = (gh * ncs + c * NSG + sgi) * QSP + qi;
    uint part = p.ck / NSG, a0 = k0 + sgi * part, a1 = min(a0 + part, k1);
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tQ((device int8_t *)Q8 + (gh * 48 + r0) * 256, dextents<int32_t, 2>(256, M));
    constexpr auto dqk = matmul2d_descriptor(M, SUBR, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<dqk, execution_simdgroups<1>> qk;
    constexpr auto dpv = matmul2d_descriptor(M, 256, SUBR, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dpv, execution_simdgroups<1>> pv;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK0((device int8_t *)K + gh * 256 * R, dextents<int32_t, 2>(SUBR, 256),
                                                                   array<int32_t, 2>{1, int(R)});
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tV0((device int8_t *)V + gh * R * 256, dextents<int32_t, 2>(256, SUBR),
                                                                   array<int32_t, 2>{1, 256});
    using PL = typename decltype(pv)::template cooperative_tensor_left_input_t<half, int8_t, float>;
    auto cO = pv.template get_destination_cooperative_tensor<PL, decltype(tV0), float>();
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0.0f;
    auto mrow = qk.template get_row_reduction_destination_cooperative_tensor<decltype(tQ), decltype(tK0), float>();
    auto lrow = qk.template get_row_reduction_destination_cooperative_tensor<decltype(tQ), decltype(tK0), float>();
    for (uint16_t j = 0; j < mrow.get_capacity(); ++j) if (mrow.is_valid_element(j)) { mrow[j] = -INFINITY; lrow[j] = 0.0f; }
    for (uint kb = a0; kb < a1; kb += SUBR) {
        uint lb = kb - s0;
        for (uint j = lane; j < SUBR; j += 32) {
            bool in = kb + j < p.n;
            ksb[sgi][j] = in ? float(KS[gh * R + lb + j]) : 0.0f; vsb[sgi][j] = in ? float(VS[gh * R + lb + j]) : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK((device int8_t *)K + gh * 256 * R + lb, dextents<int32_t, 2>(SUBR, 256),
                                                                      array<int32_t, 2>{1, int(R)});
        auto cI = qk.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
        qk.run(tQ, tK, cI);
        auto cS = qk.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), float>();
        for (uint16_t i = 0; i < cS.get_capacity(); ++i) {
            if (!cS.is_valid_element(i)) continue;
            auto ix = cS.get_multidimensional_index(i);
            uint key = kb + ix[0], tok = (r0 + ix[1]) & 7;
            bool vis = key < p.n && key <= p.p0 + tok;
            cS[i] = vis ? float(cI[i]) * qs[ix[1]] * ksb[sgi][ix[0]] : -INFINITY;
        }
        auto tmx = qk.template get_row_reduction_destination_cooperative_tensor<decltype(tQ), decltype(tK), float>();
        reduce_rows(cS, tmx, reduction_operation::max, -INFINITY);
        auto arow = qk.template get_row_reduction_destination_cooperative_tensor<decltype(tQ), decltype(tK), float>();
        for (uint16_t j = 0; j < mrow.get_capacity(); ++j) {
            if (!mrow.is_valid_element(j)) continue;
            float mn = max(mrow[j], tmx[j]);
            arow[j] = (mn == -INFINITY) ? 1.0f : exp(mrow[j] - mn);
            mrow[j] = mn;
        }
        for (auto it = cS.begin(); it != cS.end(); ++it) {
            if (!cS.is_valid_element(it)) continue;
            auto mi = mrow.map_iterator(it);
            float sv = *it;
            *it = (sv == -INFINITY) ? 0.0f : exp(sv - *mi);
        }
        auto srow = qk.template get_row_reduction_destination_cooperative_tensor<decltype(tQ), decltype(tK), float>();
        reduce_rows(cS, srow, reduction_operation::sum, 0.0f);
        for (uint16_t j = 0; j < lrow.get_capacity(); ++j) if (lrow.is_valid_element(j)) lrow[j] = lrow[j] * arow[j] + srow[j];
        auto cE = qk.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), half>();
        for (uint16_t i = 0; i < cE.get_capacity(); ++i) {
            if (!cE.is_valid_element(i)) continue;
            auto ix = cE.get_multidimensional_index(i);
            cE[i] = half(cS[i] * vsb[sgi][ix[0]]);
        }
        auto cP = pv.template get_left_input_cooperative_tensor<half, int8_t, float>(cE);
        for (uint16_t j = 0; j < arow.get_capacity(); ++j) {
            if (!arow.is_valid_element(j)) continue;
            auto ix = arow.get_multidimensional_index(j);
            alpha_s[sgi][ix[0]] = arow[j];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
            if (!cO.is_valid_element(i)) continue;
            auto ix = cO.get_multidimensional_index(i);
            cO[i] *= alpha_s[sgi][ix[1]];
        }
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tV((device int8_t *)V + (gh * R + lb) * 256, dextents<int32_t, 2>(256, SUBR),
                                                                      array<int32_t, 2>{1, 256});
        pv.run(cP, tV, cO);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        OP[(slot * M + ix[1]) * 256 + ix[0]] = cO[i];
    }
    for (uint16_t j = 0; j < mrow.get_capacity(); ++j) {
        if (!mrow.is_valid_element(j)) continue;
        auto ix = mrow.get_multidimensional_index(j);
        MP[slot * M + ix[0]] = mrow[j]; LP[slot * M + ix[0]] = lrow[j];
    }
}

// research (17 Nov 2026): the served dec_attn with the softmax in registers. QK scores stay in the QK destination
// cooperative tensor (probed layout: each lane holds rows r and r + 8, r = (lane / 2) % 8; the 4 lanes of a row in a
// simdgroup are lane ^ 1, ^ 16, ^ 17; the 4 simdgroups split the columns), row max / sum by simd shuffles + a 4 x 16
// float exchange (red, red2) instead of the 8 KB score tile, P handed to PV as its cooperative left input.
kernel void dec_attn_r4(device const int8_t *Q8 [[buffer(0)]], device const float *QS [[buffer(1)]],
                     device const int8_t *K0 [[buffer(2)]], device const int8_t *K1 [[buffer(3)]], device const int8_t *K2 [[buffer(4)]],
                     device const int8_t *K3 [[buffer(5)]], device const int8_t *K4 [[buffer(6)]],
                     device const half *KS0 [[buffer(7)]], device const half *KS1 [[buffer(8)]], device const half *KS2 [[buffer(9)]],
                     device const half *KS3 [[buffer(10)]], device const half *KS4 [[buffer(11)]],
                     device const int8_t *V0 [[buffer(12)]], device const int8_t *V1 [[buffer(13)]], device const int8_t *V2 [[buffer(14)]],
                     device const int8_t *V3 [[buffer(15)]], device const int8_t *V4 [[buffer(16)]],
                     device const half *VS0 [[buffer(17)]], device const half *VS1 [[buffer(18)]], device const half *VS2 [[buffer(19)]],
                     device const half *VS3 [[buffer(20)]], device const half *VS4 [[buffer(21)]],
                     device float *OP [[buffer(22)]], device float *MP [[buffer(23)]], device float *LP [[buffer(24)]],
                     constant DPrm &p [[buffer(25)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    threadgroup half S[M * SUB]; threadgroup float qs[M], ks[SUB], vs[SUB], red[NSG][M], red2[NSG][M], alpha_tg[M];
    uint c = tg.x, qi = tg.y, gh = tg.z, r0 = qi * M, lane = t & 31, sgi = t >> 5;
    uint k0 = p.start[0] + c * p.ck, k1 = min(k0 + p.ck, p.n);
    uint sg = 0;
    for (uint i = 1; i < NSEG; ++i) if (p.rows[i] && k0 >= p.start[i]) sg = i;
    uint R = p.rows[sg], s0 = p.start[sg];
    device const int8_t *K = sg == 0 ? K0 : sg == 1 ? K1 : sg == 2 ? K2 : sg == 3 ? K3 : K4;
    device const half *KS = sg == 0 ? KS0 : sg == 1 ? KS1 : sg == 2 ? KS2 : sg == 3 ? KS3 : KS4;
    device const int8_t *V = sg == 0 ? V0 : sg == 1 ? V1 : sg == 2 ? V2 : sg == 3 ? V3 : V4;
    device const half *VS = sg == 0 ? VS0 : sg == 1 ? VS1 : sg == 2 ? VS2 : sg == 3 ? VS3 : VS4;
    if (t < M) qs[t] = QS[gh * 48 + r0 + t];
    uint slot = (gh * p.nc + c) * QSP + qi;
    uint rA = ((lane >> 1) & 3) | ((lane >> 2) & 4), rB = rA + 8;   // probed: lanes l, l ^ 1, l ^ 8, l ^ 9 share rows
    bool wr = (lane & 9) == 0;                                   // one writer per row and simdgroup
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tQ((device int8_t *)Q8 + (gh * 48 + r0) * 256, dextents<int32_t, 2>(256, M));
    constexpr auto dqk = matmul2d_descriptor(M, SUB, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<dqk, execution_simdgroups<NSG>> qk;
    constexpr auto dpv = matmul2d_descriptor(M, 256, SUB, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dpv, execution_simdgroups<NSG>> pv;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tV0((device int8_t *)V + gh * R * 256, dextents<int32_t, 2>(256, SUB),
                                                                   array<int32_t, 2>{1, 256});
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(S, dextents<int32_t, 2>(SUB, M), array<int32_t, 2>{1, SUB});
    auto cO = pv.template get_destination_cooperative_tensor<decltype(tP), decltype(tV0), float>();
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0.0f;
    float mA = -INFINITY, mB = -INFINITY, lA = 0.0f, lB = 0.0f;
    for (uint kb = k0; kb < k1; kb += SUB) {
        uint lb = kb - s0;
        for (uint j = t; j < SUB; j += NT) {
            bool in = kb + j < p.n;
            ks[j] = in ? float(KS[gh * R + lb + j]) : 0.0f; vs[j] = in ? float(VS[gh * R + lb + j]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK((device int8_t *)K + gh * 256 * R + lb, dextents<int32_t, 2>(SUB, 256),
                                                                      array<int32_t, 2>{1, int(R)});
        auto cI = qk.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
        qk.run(tQ, tK, cI);
        float sc[32];                                            // this lane's 32 scores (capacity 32, probed)
        float xA = -INFINITY, xB = -INFINITY;
#pragma unroll
        for (uint16_t i = 0; i < 32; ++i) {
            sc[i] = -INFINITY;
            if (!cI.is_valid_element(i)) continue;
            auto ix = cI.get_multidimensional_index(i);
            uint key = kb + ix[0], tok = (r0 + ix[1]) & 7;
            bool vis = key < p.n && key <= p.p0 + tok;
            float s = vis ? float(cI[i]) * qs[ix[1]] * ks[ix[0]] : -INFINITY;
            sc[i] = s;
            if (uint(ix[1]) < 8) xA = max(xA, s); else xB = max(xB, s);
        }
        xA = max(xA, simd_shuffle_xor(xA, 1)); xA = max(xA, simd_shuffle_xor(xA, 8));
        xB = max(xB, simd_shuffle_xor(xB, 1)); xB = max(xB, simd_shuffle_xor(xB, 8));
        if (wr) { red[sgi][rA] = xA; red[sgi][rB] = xB; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tA = red[0][rA], tB = red[0][rB];
        for (int s_ = 1; s_ < NSG; ++s_) { tA = max(tA, red[s_][rA]); tB = max(tB, red[s_][rB]); }
        float nA = max(mA, tA), nB = max(mB, tB);
        float aA = (nA == -INFINITY) ? 1.0f : exp(mA - nA), aB = (nB == -INFINITY) ? 1.0f : exp(mB - nB);
        mA = nA; mB = nB;
        float sA = 0.0f, sB = 0.0f;
#pragma unroll
        for (uint16_t i = 0; i < 32; ++i) {
            if (!cI.is_valid_element(i)) continue;
            auto ix = cI.get_multidimensional_index(i);
            bool isA = uint(ix[1]) < 8;
            float sv = sc[i], mn = isA ? mA : mB;
            float e = (sv == -INFINITY) ? 0.0f : exp(sv - mn);
            if (isA) sA += e; else sB += e;
            S[ix[1] * SUB + ix[0]] = half(e * vs[ix[0]]);   // P written once (the 4-simdgroup PV needs a tensor operand)
        }
        sA += simd_shuffle_xor(sA, 1); sA += simd_shuffle_xor(sA, 8);
        sB += simd_shuffle_xor(sB, 1); sB += simd_shuffle_xor(sB, 8);
        if (wr) { red2[sgi][rA] = sA; red2[sgi][rB] = sB; if (sgi == 0) { alpha_tg[rA] = aA; alpha_tg[rB] = aB; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float uA = 0.0f, uB = 0.0f;
        for (int s_ = 0; s_ < NSG; ++s_) { uA += red2[s_][rA]; uB += red2[s_][rB]; }
        lA = lA * aA + uA; lB = lB * aB + uB;
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
            if (!cO.is_valid_element(i)) continue;
            auto ix = cO.get_multidimensional_index(i);
            cO[i] *= alpha_tg[ix[1]];
        }
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tV((device int8_t *)V + (gh * R + lb) * 256, dextents<int32_t, 2>(256, SUB),
                                                                      array<int32_t, 2>{1, 256});
        pv.run(tP, tV, cO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        OP[(slot * M + ix[1]) * 256 + ix[0]] = cO[i];
    }
    if (wr && sgi == 0) { MP[slot * M + rA] = mA; MP[slot * M + rB] = mB; LP[slot * M + rA] = lA; LP[slot * M + rB] = lB; }
}

#ifndef PF_MODE
#define PF_MODE 3   // 1: K, 2: V, 3: both
#endif
#ifndef PF_DIST
#define PF_DIST 1   // tiles ahead
#endif
// research (17 Nov 2026): dec_attn_r4 + a cache-warming prefetch of the next K / V tile (plain loads of every line)
#ifndef PF_AT
#define PF_AT 0     // where the prefetch is issued: 0 tile start, 1 after QK (K and V), 2 K after QK and V after PV
#endif
#ifndef PF_CUR
#define PF_CUR 0    // 1: touch the CURRENT tile's lines (no extra DRAM traffic: the control for register / issue cost)
#endif
#define PF_TILE(KBX) ((KBX) + (PF_CUR ? 0u : uint(PF_DIST * SUB)))
#define PF_K(KBX) { if (PF_MODE & 1) { uint nb = PF_TILE(KBX); if (nb < k1) { uint lbn = nb - s0; for (int j = 0; j < 4; ++j) { \
    uint L = t + j * NT; dummy ^= pfv[j]; pfv[j] = *(device const uint *)(K + gh * 256 * R + (L >> 1) * R + lbn + (L & 1) * 128); } } } }
#define PF_V(KBX) { if (PF_MODE & 2) { uint nb = PF_TILE(KBX); if (nb < k1) { uint lbn = nb - s0; for (int j = 0; j < 4; ++j) { \
    uint L = t + j * NT; dummy ^= pfv[4 + j]; pfv[4 + j] = *(device const uint *)(V + (gh * R + lbn) * 256 + L * 128); } } } }
kernel void dec_attn_pf(device const int8_t *Q8 [[buffer(0)]], device const float *QS [[buffer(1)]],
                     device const int8_t *K0 [[buffer(2)]], device const int8_t *K1 [[buffer(3)]], device const int8_t *K2 [[buffer(4)]],
                     device const int8_t *K3 [[buffer(5)]], device const int8_t *K4 [[buffer(6)]],
                     device const half *KS0 [[buffer(7)]], device const half *KS1 [[buffer(8)]], device const half *KS2 [[buffer(9)]],
                     device const half *KS3 [[buffer(10)]], device const half *KS4 [[buffer(11)]],
                     device const int8_t *V0 [[buffer(12)]], device const int8_t *V1 [[buffer(13)]], device const int8_t *V2 [[buffer(14)]],
                     device const int8_t *V3 [[buffer(15)]], device const int8_t *V4 [[buffer(16)]],
                     device const half *VS0 [[buffer(17)]], device const half *VS1 [[buffer(18)]], device const half *VS2 [[buffer(19)]],
                     device const half *VS3 [[buffer(20)]], device const half *VS4 [[buffer(21)]],
                     device float *OP [[buffer(22)]], device float *MP [[buffer(23)]], device float *LP [[buffer(24)]],
                     constant DPrm &p [[buffer(25)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    threadgroup half S[M * SUB]; threadgroup float qs[M], ks[SUB], vs[SUB], red[NSG][M], red2[NSG][M], alpha_tg[M];
    uint c = tg.x, qi = tg.y, gh = tg.z, r0 = qi * M, lane = t & 31, sgi = t >> 5;
    uint k0 = p.start[0] + c * p.ck, k1 = min(k0 + p.ck, p.n);
    uint sg = 0;
    for (uint i = 1; i < NSEG; ++i) if (p.rows[i] && k0 >= p.start[i]) sg = i;
    uint R = p.rows[sg], s0 = p.start[sg];
    device const int8_t *K = sg == 0 ? K0 : sg == 1 ? K1 : sg == 2 ? K2 : sg == 3 ? K3 : K4;
    device const half *KS = sg == 0 ? KS0 : sg == 1 ? KS1 : sg == 2 ? KS2 : sg == 3 ? KS3 : KS4;
    device const int8_t *V = sg == 0 ? V0 : sg == 1 ? V1 : sg == 2 ? V2 : sg == 3 ? V3 : V4;
    device const half *VS = sg == 0 ? VS0 : sg == 1 ? VS1 : sg == 2 ? VS2 : sg == 3 ? VS3 : VS4;
    if (t < M) qs[t] = QS[gh * 48 + r0 + t];
    uint slot = (gh * p.nc + c) * QSP + qi;
    uint rA = ((lane >> 1) & 3) | ((lane >> 2) & 4), rB = rA + 8;   // probed: lanes l, l ^ 1, l ^ 8, l ^ 9 share rows
    bool wr = (lane & 9) == 0;                                   // one writer per row and simdgroup
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tQ((device int8_t *)Q8 + (gh * 48 + r0) * 256, dextents<int32_t, 2>(256, M));
    constexpr auto dqk = matmul2d_descriptor(M, SUB, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<dqk, execution_simdgroups<NSG>> qk;
    constexpr auto dpv = matmul2d_descriptor(M, 256, SUB, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dpv, execution_simdgroups<NSG>> pv;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tV0((device int8_t *)V + gh * R * 256, dextents<int32_t, 2>(256, SUB),
                                                                   array<int32_t, 2>{1, 256});
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(S, dextents<int32_t, 2>(SUB, M), array<int32_t, 2>{1, SUB});
    auto cO = pv.template get_destination_cooperative_tensor<decltype(tP), decltype(tV0), float>();
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0.0f;
    float mA = -INFINITY, mB = -INFINITY, lA = 0.0f, lB = 0.0f;
    uint pfv[8] = {0, 0, 0, 0, 0, 0, 0, 0}, dummy = 0;
    for (uint kb = k0; kb < k1; kb += SUB) {
        uint lb = kb - s0;
#if PF_AT == 0
        PF_K(kb); PF_V(kb);
#endif
        for (uint j = t; j < SUB; j += NT) {
            bool in = kb + j < p.n;
            ks[j] = in ? float(KS[gh * R + lb + j]) : 0.0f; vs[j] = in ? float(VS[gh * R + lb + j]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK((device int8_t *)K + gh * 256 * R + lb, dextents<int32_t, 2>(SUB, 256),
                                                                      array<int32_t, 2>{1, int(R)});
        auto cI = qk.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
        qk.run(tQ, tK, cI);
#if PF_AT == 1
        PF_K(kb); PF_V(kb);                                      // after the current K is consumed
#elif PF_AT == 2
        PF_K(kb);
#endif
        float sc[32];                                            // this lane's 32 scores (capacity 32, probed)
        float xA = -INFINITY, xB = -INFINITY;
#pragma unroll
        for (uint16_t i = 0; i < 32; ++i) {
            sc[i] = -INFINITY;
            if (!cI.is_valid_element(i)) continue;
            auto ix = cI.get_multidimensional_index(i);
            uint key = kb + ix[0], tok = (r0 + ix[1]) & 7;
            bool vis = key < p.n && key <= p.p0 + tok;
            float s = vis ? float(cI[i]) * qs[ix[1]] * ks[ix[0]] : -INFINITY;
            sc[i] = s;
            if (uint(ix[1]) < 8) xA = max(xA, s); else xB = max(xB, s);
        }
        xA = max(xA, simd_shuffle_xor(xA, 1)); xA = max(xA, simd_shuffle_xor(xA, 8));
        xB = max(xB, simd_shuffle_xor(xB, 1)); xB = max(xB, simd_shuffle_xor(xB, 8));
        if (wr) { red[sgi][rA] = xA; red[sgi][rB] = xB; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tA = red[0][rA], tB = red[0][rB];
        for (int s_ = 1; s_ < NSG; ++s_) { tA = max(tA, red[s_][rA]); tB = max(tB, red[s_][rB]); }
        float nA = max(mA, tA), nB = max(mB, tB);
        float aA = (nA == -INFINITY) ? 1.0f : exp(mA - nA), aB = (nB == -INFINITY) ? 1.0f : exp(mB - nB);
        mA = nA; mB = nB;
        float sA = 0.0f, sB = 0.0f;
#pragma unroll
        for (uint16_t i = 0; i < 32; ++i) {
            if (!cI.is_valid_element(i)) continue;
            auto ix = cI.get_multidimensional_index(i);
            bool isA = uint(ix[1]) < 8;
            float sv = sc[i], mn = isA ? mA : mB;
            float e = (sv == -INFINITY) ? 0.0f : exp(sv - mn);
            if (isA) sA += e; else sB += e;
            S[ix[1] * SUB + ix[0]] = half(e * vs[ix[0]]);   // P written once (the 4-simdgroup PV needs a tensor operand)
        }
        sA += simd_shuffle_xor(sA, 1); sA += simd_shuffle_xor(sA, 8);
        sB += simd_shuffle_xor(sB, 1); sB += simd_shuffle_xor(sB, 8);
        if (wr) { red2[sgi][rA] = sA; red2[sgi][rB] = sB; if (sgi == 0) { alpha_tg[rA] = aA; alpha_tg[rB] = aB; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float uA = 0.0f, uB = 0.0f;
        for (int s_ = 0; s_ < NSG; ++s_) { uA += red2[s_][rA]; uB += red2[s_][rB]; }
        lA = lA * aA + uA; lB = lB * aB + uB;
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
            if (!cO.is_valid_element(i)) continue;
            auto ix = cO.get_multidimensional_index(i);
            cO[i] *= alpha_tg[ix[1]];
        }
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tV((device int8_t *)V + (gh * R + lb) * 256, dextents<int32_t, 2>(256, SUB),
                                                                      array<int32_t, 2>{1, 256});
        pv.run(tP, tV, cO);
#if PF_AT == 2
        PF_V(kb);                                                // after the current V is consumed
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        OP[(slot * M + ix[1]) * 256 + ix[0]] = cO[i];
    }
    if (dummy == 0x9e3779b9u) OP[0] = 1.0f;                   // keeps the prefetch loads alive
    if (wr && sgi == 0) { MP[slot * M + rA] = mA; MP[slot * M + rB] = mB; LP[slot * M + rA] = lA; LP[slot * M + rB] = lB; }
}

// research (17 Nov 2026): double-buffered K / V through threadgroup memory. Each thread holds the NEXT 32-key tile in
// 8 x uint4 registers (loads issued right after the current tile is stored, so they overlap its compute); the QK / PV
// matmuls read K / V from threadgroup memory (Kt [dim][key], Vt [key][dim]); softmax as dec_attn (score tile S).
#ifndef TB
#define TB 32
#endif
kernel void dec_attn_db(device const int8_t *Q8 [[buffer(0)]], device const float *QS [[buffer(1)]],
                     device const int8_t *K0 [[buffer(2)]], device const int8_t *K1 [[buffer(3)]], device const int8_t *K2 [[buffer(4)]],
                     device const int8_t *K3 [[buffer(5)]], device const int8_t *K4 [[buffer(6)]],
                     device const half *KS0 [[buffer(7)]], device const half *KS1 [[buffer(8)]], device const half *KS2 [[buffer(9)]],
                     device const half *KS3 [[buffer(10)]], device const half *KS4 [[buffer(11)]],
                     device const int8_t *V0 [[buffer(12)]], device const int8_t *V1 [[buffer(13)]], device const int8_t *V2 [[buffer(14)]],
                     device const int8_t *V3 [[buffer(15)]], device const int8_t *V4 [[buffer(16)]],
                     device const half *VS0 [[buffer(17)]], device const half *VS1 [[buffer(18)]], device const half *VS2 [[buffer(19)]],
                     device const half *VS3 [[buffer(20)]], device const half *VS4 [[buffer(21)]],
                     device float *OP [[buffer(22)]], device float *MP [[buffer(23)]], device float *LP [[buffer(24)]],
                     constant DPrm &p [[buffer(25)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    constexpr int TPR2 = NT / M, KPT2 = TB / TPR2, NU = 256 * TB / 16 / NT;   // uint4 per thread for each of K, V
    threadgroup int8_t Kt[256 * TB];
    threadgroup int8_t Vt[TB * 256];
    threadgroup half S[M * TB];
    threadgroup float alpha[M], qs[M], ks[TB], vs[TB];
    uint c = tg.x, qi = tg.y, gh = tg.z, r0 = qi * M;
    uint k0 = p.start[0] + c * p.ck, k1 = min(k0 + p.ck, p.n);
    uint sg = 0;
    for (uint i = 1; i < NSEG; ++i) if (p.rows[i] && k0 >= p.start[i]) sg = i;
    uint R = p.rows[sg], s0 = p.start[sg];
    device const int8_t *K = sg == 0 ? K0 : sg == 1 ? K1 : sg == 2 ? K2 : sg == 3 ? K3 : K4;
    device const half *KS = sg == 0 ? KS0 : sg == 1 ? KS1 : sg == 2 ? KS2 : sg == 3 ? KS3 : KS4;
    device const int8_t *V = sg == 0 ? V0 : sg == 1 ? V1 : sg == 2 ? V2 : sg == 3 ? V3 : V4;
    device const half *VS = sg == 0 ? VS0 : sg == 1 ? VS1 : sg == 2 ? VS2 : sg == 3 ? VS3 : VS4;
    if (t < M) qs[t] = QS[gh * 48 + r0 + t];
    uint slot = (gh * p.nc + c) * QSP + qi;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tQ((device int8_t *)Q8 + (gh * 48 + r0) * 256, dextents<int32_t, 2>(256, M));
    constexpr auto dqk = matmul2d_descriptor(M, TB, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<dqk, execution_simdgroups<NSG>> qk;
    constexpr auto dpv = matmul2d_descriptor(M, 256, TB, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dpv, execution_simdgroups<NSG>> pv;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(S, dextents<int32_t, 2>(TB, M), array<int32_t, 2>{1, TB});
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tKt(Kt, dextents<int32_t, 2>(TB, 256), array<int32_t, 2>{1, TB});
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tVt(Vt, dextents<int32_t, 2>(256, TB), array<int32_t, 2>{1, 256});
    auto cO = pv.template get_destination_cooperative_tensor<decltype(tP), decltype(tVt), float>();
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0.0f;
    uint row = t / TPR2, part = t % TPR2;
    float m = -INFINITY, l = 0.0f;
    uint4 rk[NU], rv[NU];
    constexpr uint KPU = TB / 16;                                  // uint4 per dim row of the K tile
#define DB_LOAD(KBX) { uint lbx = (KBX) - s0; for (int j = 0; j < NU; ++j) { uint u = t + j * NT; \
        rk[j] = *(device const uint4 *)(K + gh * 256 * R + (u / KPU) * R + lbx + (u % KPU) * 16); \
        rv[j] = *(device const uint4 *)(V + (gh * R + lbx) * 256 + u * 16); } }
    DB_LOAD(k0);
    for (uint kb = k0; kb < k1; kb += TB) {
        uint lb = kb - s0;
        threadgroup_barrier(mem_flags::mem_threadgroup);           // the previous tile's matmuls are done with Kt / Vt / S
        for (int j = 0; j < NU; ++j) {
            uint u = t + j * NT;
            *(threadgroup uint4 *)(Kt + (u / KPU) * TB + (u % KPU) * 16) = rk[j];
            *(threadgroup uint4 *)(Vt + u * 16) = rv[j];
        }
        for (uint j = t; j < TB; j += NT) {
            bool in = kb + j < p.n;
            ks[j] = in ? float(KS[gh * R + lb + j]) : 0.0f; vs[j] = in ? float(VS[gh * R + lb + j]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kb + TB < k1) DB_LOAD(kb + TB);                        // the next tile's loads overlap this tile's compute
        auto cI = qk.template get_destination_cooperative_tensor<decltype(tQ), decltype(tKt), int32_t>();
        qk.run(tQ, tKt, cI);
        for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
            if (!cI.is_valid_element(i)) continue;
            auto ix = cI.get_multidimensional_index(i);
            uint key = kb + ix[0], tok = (r0 + ix[1]) & 7;
            bool vis = key < p.n && key <= p.p0 + tok;
            S[ix[1] * TB + ix[0]] = vis ? half(float(cI[i]) * qs[ix[1]] * ks[ix[0]]) : half(-INFINITY);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half *Sr = S + row * TB + part * KPT2;
        float tm = -INFINITY;
        for (int j = 0; j < KPT2; ++j) tm = max(tm, float(Sr[j]));
        for (int o = 1; o < TPR2; o <<= 1) tm = max(tm, simd_shuffle_xor(tm, o));
        float mn = max(m, tm), a = (mn == -INFINITY) ? 1.0f : exp(m - mn), sum = 0.0f;
        for (int j = 0; j < KPT2; ++j) {
            float sv = float(Sr[j]);
            float e = (sv == -INFINITY) ? 0.0f : exp(sv - mn); sum += e;
            Sr[j] = half(e * vs[part * KPT2 + j]);
        }
        for (int o = 1; o < TPR2; o <<= 1) sum += simd_shuffle_xor(sum, o);
        l = l * a + sum; m = mn;
        if (part == 0) alpha[row] = a;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
            if (!cO.is_valid_element(i)) continue;
            auto ix = cO.get_multidimensional_index(i);
            cO[i] *= alpha[ix[1]];
        }
        pv.run(tP, tVt, cO);
    }
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        OP[(slot * M + ix[1]) * 256 + ix[0]] = cO[i];
    }
    if (part == 0) { MP[slot * M + row] = m; LP[slot * M + row] = l; }
}
#undef DB_LOAD
