
// ---- FP8 (e4m3) probabilities x FP8 V on the served KV8 + K8V16 path (F8=c|v, 18 Nov 2026; appended to kernels.metal
// by gpu_v8_f8.swift). The M6 matmul2d runs FP8 x FP8 at about twice the FP16 rate (no FP8 x INT8 or fast FP16 x FP8
// form), so both P.V operands are e4m3:
//   F8=c  V copy = e4m3(INT8 code) (exact for |code| <= 16), P' = e4m3(exp(s - m) x vs_j x c_t) with c_t = 256 / the
//         tile's largest value scale of that KV head; the P.V epilogue divides by c_t; the sum is of exp(s - m) (fp32)
//   F8=v  V copy = e4m3(v x vsc) (the fp16 value, vsc = F8VS), P = e4m3(exp(s - m) x 256), sum = 256 vsc sum(exp)
// The V copy is one byte per value, laid out [kv head][dim half][Lmax][128] so each P.V operand is a dense tile (the
// FP8 tensors take implied strides); P is written per tile with row stride n (dense [rows][n]).

// float -> e4m3 (bias 7, max 448 = 0x7E, round to nearest even, saturating; no NaN produced)
inline uchar f2e4m3(float x) {
    uint s = (as_type<uint>(x) >> 24) & 0x80u;
    float a = min(fabs(x), 448.0f);
    if (a < 0.015625f) return uchar(s | uint(rint(a * 512.0f)));   // subnormal m x 2^-9 (8 -> 0x08 = 2^-6)
    uint b = as_type<uint>(a);
    b += 0x7FFFFu + ((b >> 20) & 1u);
    uint r = ((((b >> 23) - 120u) << 3) | ((b >> 20) & 7u));
    return uchar(s | min(r, 0x7Eu));
}
inline uint f8idx(uint h, uint d, uint Lmax, uint row) { return ((h * 2 + (d >> 7)) * Lmax + row) * 128 + (d & 127); }

// attn_prep8qv with the FP8 V copy (buffer 16: uchar) instead of the fp16 one; vsc (buffer 18) for F8=v
template <bool CODES>
inline void prep8qv_f8(device const float *qg, device const float *kp, device const float *vp, device const float *qn,
                       device const float *kn, device half *Q, device int8_t *K8, device int8_t *V8, device half *KS,
                       device half *VS, constant uint4 &st, constant uint2 &st2, float eps, device const float *cs,
                       device int8_t *Q8, device float *QS, device uchar *Vf, constant uint2 &vl, float vsc,
                       uint2 g, uint d, uint sl, uint sg, threadgroup float *part, threadgroup float *buf) {
    uint h = g.x, t = g.y, r = st2.y + t;
    float v;
    if (h < 24) v = qg[(h * 512 + d) * T + t];
    else if (h < 28) v = kp[((h - 24) * 256 + d) * T + t];
    else v = vp[((h - 28) * 256 + d) * T + t];
    if (h < 28) {
        float s = simd_sum(v * v);
        if (sl == 0) part[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0;
        for (uint i = 0; i < 8; i++) tot += part[i];
        v = v * rsqrt(tot / 256.0f + eps) * (h < 24 ? qn[d] : kn[d]);
        buf[d] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d < 64) {
            float c = cs[(t * 64 + d) * 2], sn = cs[(t * 64 + d) * 2 + 1];
            float rot = d < 32 ? -buf[d + 32] : buf[d - 32];
            v = buf[d] * c + rot * sn;
        }
        if (h < 24) {
            Q[(h * T + t) * 256 + d] = half(v * 0.0625f);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float am = simd_max(fabs(v));
            if (sl == 0) part[sg] = am;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float mq = part[0];
            for (uint i = 1; i < 8; i++) mq = max(mq, part[i]);
            float sq = mq > 0 ? mq / 127.0f : 1.0f;
            Q8[(h * T + t) * 256 + d] = int8_t(clamp(rint(v / sq), -127.0f, 127.0f));
            if (d == 0) QS[h * T + t] = sq * 0.0625f;
            return;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float hv = float(half(v));
    float m = simd_max(fabs(hv));
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    half sc = half(max(m / 127.0f, 1e-6f));
    int8_t code = int8_t(clamp(rint(hv / float(sc)), -127.0f, 127.0f));
    if (h < 28) { uint kh = h - 24; K8[kh * st.x + d * st.y + r] = code; if (d == 0) KS[kh * st2.x + r] = sc; }
    else { uint vh = h - 28; V8[vh * st.z + r * st.w + d] = code; if (d == 0) VS[vh * st2.x + r] = sc;
           Vf[f8idx(vh, d, vl.x, vl.y + t)] = CODES ? f2e4m3(float(code)) : f2e4m3(hv * vsc); }
}
#define PREP_F8_ARGS device const float *qg [[buffer(0)]], device const float *kp [[buffer(1)]], device const float *vp [[buffer(2)]], \
    device const float *qn [[buffer(3)]], device const float *kn [[buffer(4)]], device half *Q [[buffer(5)]], \
    device int8_t *K8 [[buffer(6)]], device int8_t *V8 [[buffer(7)]], device half *KS [[buffer(8)]], \
    device half *VS [[buffer(9)]], constant uint4 &st [[buffer(10)]], constant uint2 &st2 [[buffer(11)]], \
    constant float &eps [[buffer(12)]], device const float *cs [[buffer(13)]], \
    device int8_t *Q8 [[buffer(14)]], device float *QS [[buffer(15)]], \
    device uchar *Vf [[buffer(16)]], constant uint2 &vl [[buffer(17)]], constant float &vsc [[buffer(18)]], \
    uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]], \
    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
kernel void attn_prep8qv_f8c(PREP_F8_ARGS) {
    threadgroup float part[8], buf[256];
    prep8qv_f8<true>(qg, kp, vp, qn, kn, Q, K8, V8, KS, VS, st, st2, eps, cs, Q8, QS, Vf, vl, vsc, g, d, sl, sg, part, buf);
}
kernel void attn_prep8qv_f8v(PREP_F8_ARGS) {
    threadgroup float part[8], buf[256];
    prep8qv_f8<false>(qg, kp, vp, qn, kn, Q, K8, V8, KS, VS, st, st2, eps, cs, Q8, QS, Vf, vl, vsc, g, d, sl, sg, part, buf);
}

// FP8 V copy rows from an ANE INT8 V segment (as kv8_v_to_v16); prm = (Lmax, s0, vh head stride, vr row stride)
kernel void kv8_v_to_f8c(device const int8_t *V8 [[buffer(0)]], device const half *VS [[buffer(1)]], device uchar *Vf [[buffer(2)]],
                         constant uint4 &prm [[buffer(3)]], constant uint &sh [[buffer(4)]], constant float &vsc [[buffer(5)]],
                         uint3 p [[thread_position_in_grid]]) {
    uint d = p.x, r = p.y, h = p.z;
    Vf[f8idx(h, d, prm.x, prm.y + r)] = f2e4m3(float(V8[h * prm.z + r * prm.w + d]));
}
kernel void kv8_v_to_f8v(device const int8_t *V8 [[buffer(0)]], device const half *VS [[buffer(1)]], device uchar *Vf [[buffer(2)]],
                         constant uint4 &prm [[buffer(3)]], constant uint &sh [[buffer(4)]], constant float &vsc [[buffer(5)]],
                         uint3 p [[thread_position_in_grid]]) {
    uint d = p.x, r = p.y, h = p.z;
    Vf[f8idx(h, d, prm.x, prm.y + r)] = f2e4m3(float(V8[h * prm.z + r * prm.w + d]) * float(VS[h * sh + r]) * vsc);
}

// softmax over a tile's scaled scores (as attn_softmax8n) writing e4m3 probabilities P8[r][n] (row stride n).
// F8=v: P = e4m3(256 e), stats (m, 256 vsc sum e). F8=c: P' = e4m3(e x vs_j x c), c = 256 / max_j vs_j of the row's
// KV head over the tile (VS bound at the tile's first key, head stride ssh halves), stats (m, sum e), CB[kv head] = c.
template <bool CODES>
inline void softmax_f8(device half *S, device float *st, uint TK, uint a, uint n, uint p0, device uchar *P8, float vsc,
                       device float *CB, device const half *VS, uint ssh, uint r, uint tid, uint sl, uint sg, threadgroup float *part) {
    device half *row = S + r * TK;
    int lim = int(p0 + (r % T) + 1) - int(a);
    uint nv = uint(clamp(lim, 0, int(n))), gh = (r / T) / 6;
    float m = -INFINITY;
    for (uint j = tid; j < nv; j += 256) m = max(m, float(row[j]));
    m = simd_max(m);
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float c = 256.0f;
    device const half *vs = VS + gh * ssh;
    if (CODES) {
        float vm = 0;
        for (uint j = tid; j < n; j += 256) vm = max(vm, float(vs[j]));
        vm = simd_max(vm);
        if (sl == 0) part[sg] = vm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        vm = part[0];
        for (uint i = 1; i < 8; i++) vm = max(vm, part[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c = 256.0f / max(vm, 1e-12f);
    }
    device uchar *prow = P8 + r * n;
    float s = 0;
    for (uint j = tid; j < n; j += 256) {
        float e = (j < nv && m != -INFINITY) ? exp(float(row[j]) - m) : 0.0f;
        prow[j] = f2e4m3(CODES ? e * float(vs[j]) * c : e * 256.0f);
        s += e;
    }
    s = simd_sum(s);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tot = 0; for (uint i = 0; i < 8; i++) tot += part[i];
        st[2 * r] = m; st[2 * r + 1] = CODES ? tot : tot * 256.0f * vsc;
        if (CODES && (r % (6 * T)) == 0) CB[gh] = c;
    }
}
#define SM_F8_ARGS device half *S [[buffer(0)]], device float *st [[buffer(1)]], constant uint &TK [[buffer(2)]], \
    constant uint &a [[buffer(3)]], constant uint &n [[buffer(4)]], constant uint &p0 [[buffer(5)]], \
    device uchar *P8 [[buffer(6)]], constant float &vsc [[buffer(7)]], device float *CB [[buffer(8)]], \
    device const half *VS [[buffer(9)]], constant uint &ssh [[buffer(10)]], \
    uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
kernel void attn_softmax8n_f8c(SM_F8_ARGS) { threadgroup float part[8]; softmax_f8<true>(S, st, TK, a, n, p0, P8, vsc, CB, VS, ssh, r, tid, sl, sg, part); }
kernel void attn_softmax8n_f8v(SM_F8_ARGS) { threadgroup float part[8]; softmax_f8<false>(S, st, TK, a, n, p0, P8, vsc, CB, VS, ssh, r, tid, sl, sg, part); }

// O tile (64 rows x 128 dims, fp32) = P8 (64 rows x n keys, e4m3) x V (n keys x 128 dims, e4m3); F8=c divides by the
// tile's c of the KV head. Grid (2, 6T / 64, 4), 128 threads.
template <bool CODES>
inline void pv_f8(device uchar *P8, device uchar *Vf, device float *O, device const float *CB, uint Lmax, uint a, uint n, uint3 tg) {
    uint dt = tg.x, rt = tg.y, gh = tg.z;
    using TF = tensor<device metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline>;
    TF tP((typename TF::data_handle_type)(P8 + (gh * 6 * T + rt * 64) * n), dextents<int32_t, 2>(int(n), 64));
    TF tV((typename TF::data_handle_type)(Vf + ((gh * 2 + dt) * Lmax + a) * 128), dextents<int32_t, 2>(128, int(n)));
    constexpr auto d = matmul2d_descriptor(64, 128, static_cast<int>(dynamic_extent), false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    auto cO = op.template get_destination_cooperative_tensor<TF, TF, float>();
    op.run(tP, tV, cO);
    float sc = CODES ? 1.0f / CB[gh] : 1.0f;
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        O[(gh * 6 * T + rt * 64 + ix[1]) * 256 + dt * 128 + ix[0]] = cO[i] * sc;
    }
}
#define PV_F8_ARGS device uchar *P8 [[buffer(0)]], device uchar *Vf [[buffer(1)]], device float *O [[buffer(2)]], \
    constant uint &Lmax [[buffer(3)]], constant uint &a [[buffer(4)]], constant uint &n [[buffer(5)]], \
    device const float *CB [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]]
kernel void attn_pv_f8c(PV_F8_ARGS) { pv_f8<true>(P8, Vf, O, CB, Lmax, a, n, tg); }
kernel void attn_pv_f8v(PV_F8_ARGS) { pv_f8<false>(P8, Vf, O, CB, Lmax, a, n, tg); }

// F8=w: F8=v with the softmax vectorized (half4 loads, uchar4 stores) and the e4m3 conversion through half precision:
// x in [2^-6, 448] -> the half's exponent and top 3 mantissa bits (round to nearest even), re-biased; below 2^-6 the
// subnormal codes m x 2^-9. Inputs are probabilities x 256 (>= 0).
inline uint e4m3_pos(float x) {
    half hx = half(min(x, 448.0f));
    ushort h = as_type<ushort>(hx);
    if (h < 0x2400) return uint(rint(float(hx) * 512.0f));
    uint r = (uint(h) + 0x3Fu + ((uint(h) >> 7) & 1u)) >> 7;
    return min(r - 64u, 0x7Eu);
}
kernel void attn_softmax8n_f8w(SM_F8_ARGS) {
    threadgroup float part[8];
    device half4 *row = (device half4 *)(S + r * TK);
    int lim = int(p0 + (r % T) + 1) - int(a);
    uint nv = uint(clamp(lim, 0, int(n))), n4 = n / 4;
    float m = -INFINITY;
    for (uint j4 = tid; j4 < n4; j4 += 256) {
        float4 x = float4(row[j4]); uint j = 4 * j4;
        m = max(m, max(max(j < nv ? x.x : -INFINITY, j + 1 < nv ? x.y : -INFINITY), max(j + 2 < nv ? x.z : -INFINITY, j + 3 < nv ? x.w : -INFINITY)));
    }
    m = simd_max(m);
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uchar4 *prow = (device uchar4 *)(P8 + r * n);
    float s = 0;
    for (uint j4 = tid; j4 < n4; j4 += 256) {
        float4 x = float4(row[j4]) - m; uint j = 4 * j4;
        float4 e = (m == -INFINITY) ? float4(0.0f) : float4(j < nv ? exp(x.x) : 0.0f, j + 1 < nv ? exp(x.y) : 0.0f, j + 2 < nv ? exp(x.z) : 0.0f, j + 3 < nv ? exp(x.w) : 0.0f);
        prow[j4] = uchar4(e4m3_pos(e.x * 256.0f), e4m3_pos(e.y * 256.0f), e4m3_pos(e.z * 256.0f), e4m3_pos(e.w * 256.0f));
        s += e.x + e.y + e.z + e.w;
    }
    s = simd_sum(s);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float tot = 0; for (uint i = 0; i < 8; i++) tot += part[i]; st[2 * r] = m; st[2 * r + 1] = tot * 256.0f * vsc; }
}

// F8=x: F8=w with exp in half precision (e4m3 keeps 3 mantissa bits) and the causal mask tested only on the tile's
// boundary chunk; sums in fp32.
inline uint e4m3_h(half hx) {   // hx in [0, 448]
    ushort h = as_type<ushort>(hx);
    if (h < 0x2400) return uint(rint(float(hx) * 512.0f));
    uint r = (uint(h) + 0x3Fu + ((uint(h) >> 7) & 1u)) >> 7;
    return min(r - 64u, 0x7Eu);
}
kernel void attn_softmax8n_f8x(SM_F8_ARGS) {
    threadgroup float part[8];
    device half4 *row = (device half4 *)(S + r * TK);
    int lim = int(p0 + (r % T) + 1) - int(a);
    uint nv = uint(clamp(lim, 0, int(n))), n4 = n / 4;
    half4 mh = half4(-INFINITY);
    for (uint j4 = tid; j4 < n4; j4 += 256) {
        half4 x = row[j4]; uint j = 4 * j4;
        if (j + 3 >= nv) x = half4(j < nv ? x.x : -INFINITY, j + 1 < nv ? x.y : -INFINITY, j + 2 < nv ? x.z : -INFINITY, j + 3 < nv ? x.w : -INFINITY);
        mh = max(mh, x);
    }
    float m = simd_max(float(max(max(mh.x, mh.y), max(mh.z, mh.w))));
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uchar4 *prow = (device uchar4 *)(P8 + r * n);
    float s = 0;
    half m16 = half(m);
    for (uint j4 = tid; j4 < n4; j4 += 256) {
        half4 e = exp(row[j4] - m16); uint j = 4 * j4;
        if (j + 3 >= nv || m == -INFINITY) e = half4(j < nv ? e.x : 0.0h, j + 1 < nv ? e.y : 0.0h, j + 2 < nv ? e.z : 0.0h, j + 3 < nv ? e.w : 0.0h);
        if (m == -INFINITY) e = half4(0.0h);
        half4 e2 = e * 256.0h;
        prow[j4] = uchar4(e4m3_h(e2.x), e4m3_h(e2.y), e4m3_h(e2.z), e4m3_h(e2.w));
        s += float(e.x) + float(e.y) + float(e.z) + float(e.w);
    }
    s = simd_sum(s);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float tot = 0; for (uint i = 0; i < 8; i++) tot += part[i]; st[2 * r] = m; st[2 * r + 1] = tot * 256.0f * vsc; }
}
