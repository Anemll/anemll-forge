"""QK8 (15 Nov 2026): int8 x int8 attention scores on the GPU. attn_prep16q8 = attn_prep16 that also writes Q8 / QS
(int8 [24][T][256], per (head, token) scale) and the layer's K8c / KSc (int8 [4][Lmax][256], per (kv head, key) scale);
attn_qk8i = attn_qk_h on the int8 operands (int32 accumulate, scores = acc x qs x ks, half). Engine env QK8=1."""
from pathlib import Path
E = Path(__file__).resolve().parent
k = E / "kernels.metal"; s = k.read_text()
if "attn_qk8i" not in s:
    a = "kernel void attn_prep16(device const float *qg [[buffer(0)]],"
    kern = r'''// QK8: attn_prep16 + int8 copies of the new q rows (Q8 / QS) and k rows (K8c / KSc, per kv head and key)
kernel void attn_prep16q8(device const float *qg [[buffer(0)]], device const float *kp [[buffer(1)]], device const float *vp [[buffer(2)]],
                          device const float *qn [[buffer(3)]], device const float *kn [[buffer(4)]], device half *Q [[buffer(5)]],
                          device half *Kc [[buffer(6)]], device half *Vc [[buffer(7)]], constant uint &p0 [[buffer(8)]],
                          constant uint &Lmax [[buffer(9)]], constant float &eps [[buffer(10)]], device const float *cs [[buffer(11)]],
                          device int8_t *Q8 [[buffer(12)]], device float *QS [[buffer(13)]], device int8_t *K8 [[buffer(14)]],
                          device float *KS [[buffer(15)]],
                          uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                          uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8], buf[256], mx[8];
    uint h = g.x, t = g.y;
    float v;
    if (h < 24) v = qg[(h * 512 + d) * T + t];
    else if (h < 28) v = kp[((h - 24) * 256 + d) * T + t];
    else { Vc[((h - 28) * Lmax + p0 + t) * 256 + d] = half(vp[((h - 28) * 256 + d) * T + t]); return; }
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
    half hv = half(v);
    if (h < 24) Q[(h * T + t) * 256 + d] = hv;
    else Kc[((h - 24) * Lmax + p0 + t) * 256 + d] = hv;
    float m = simd_max(abs(float(hv)));
    if (sl == 0) mx[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mm = 0;
    for (uint i = 0; i < 8; i++) mm = max(mm, mx[i]);
    float sc = mm > 0 ? mm / 127.0f : 1.0f;
    int8_t q8 = int8_t(rint(float(hv) / sc));
    if (h < 24) { Q8[(h * T + t) * 256 + d] = q8; if (d == 0) QS[h * T + t] = sc; }
    else { K8[((h - 24) * Lmax + p0 + t) * 256 + d] = q8; if (d == 0) KS[(h - 24) * Lmax + p0 + t] = sc; }
}
// QK8: S tile = (Q8 K8[a ..]^T) x qs x ks (int32 accumulate), as attn_qk_h
kernel void attn_qk8i(device int8_t *Q8 [[buffer(0)]], device int8_t *K8 [[buffer(1)]], device half *S [[buffer(2)]],
                      constant uint &Lmax [[buffer(3)]], constant uint &a [[buffer(4)]], constant uint &TK [[buffer(5)]],
                      device const float *QS [[buffer(6)]], device const float *KS [[buffer(7)]],
                      uint3 tg [[threadgroup_position_in_grid]]) {
    uint kt = tg.x, rt = tg.y, gh = tg.z, r0 = gh * 6 * T + rt * 64, k0 = gh * Lmax + a + kt * 64;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tQ(Q8 + r0 * 256, dextents<int32_t, 2>(256, 64));
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK(K8 + k0 * 256, dextents<int32_t, 2>(256, 64));
    constexpr auto d = matmul2d_descriptor(64, 64, 256, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    auto cI = op.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
    op.run(tQ, tK, cI);
    for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
        if (!cI.is_valid_element(i)) continue;
        auto ix = cI.get_multidimensional_index(i);
        uint key = ix[0], row = ix[1];
        S[(r0 + row) * TK + kt * 64 + key] = half(float(cI[i]) * QS[r0 + row] * KS[k0 + key]);
    }
}
'''
    assert s.count(a) == 1
    s = s.replace(a, kern + a)
    k.write_text(s); print("kernels patched")
w = E / "gpu_v8.swift"; s = w.read_text()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:90]); s = s.replace(a, b)
if "QK8" not in s:
    rep('''let LRM = ProcessInfo.processInfo.environment["LRM"] == "1"''',
        '''let LRM = ProcessInfo.processInfo.environment["LRM"] == "1"
let QK8 = ProcessInfo.processInfo.environment["QK8"] == "1"   // int8 x int8 attention scores (attn_prep16q8 + attn_qk8i)
let Q8B = buf(24 * T * 256), QSB = buf(24 * T * 4)''')
    rep('''    var S: MTLBuffer? = nil, CS: MTLBuffer? = nil, Kc: MTLBuffer? = nil, Vc: MTLBuffer? = nil''',
        '''    var S: MTLBuffer? = nil, CS: MTLBuffer? = nil, Kc: MTLBuffer? = nil, Vc: MTLBuffer? = nil
    var K8c: MTLBuffer? = nil, KSc: MTLBuffer? = nil   // QK8: int8 keys + per (kv head, key) scales''')
    rep('''else if !KV8 { Kc = buf(4 * Lmax * 256 * 2); Vc = buf(4 * Lmax * 256 * 2) }''',
        '''else if !KV8 { Kc = buf(4 * Lmax * 256 * 2); Vc = buf(4 * Lmax * 256 * 2); if QK8 { K8c = buf(4 * Lmax * 256); KSc = buf(4 * Lmax * 4) } }''')
    rep('''        e.run(pso("attn_prep16"), [BIG, KP, VP, l.bufs["qn"], l.bufs["kn"], QA, l.Kc, l.Vc, nil, nil, nil, csAll],''',
        '''        if QK8 { e.run(pso("attn_prep16q8"), [BIG, KP, VP, l.bufs["qn"], l.bufs["kn"], QA, l.Kc, l.Vc, nil, nil, nil, csAll, Q8B, QSB, l.K8c, l.KSc],
              [(UInt32(p0), 8), (UInt32(Lmax), 9), (eps, 10)], grid: sz(32, T), tg: sz(256), offsets: [11: p0 * 64 * 2 * 4]) } else {
        e.run(pso("attn_prep16"), [BIG, KP, VP, l.bufs["qn"], l.bufs["kn"], QA, l.Kc, l.Vc, nil, nil, nil, csAll],''')
    # close the else after the original prep16 call (its argument line ends with offsets: [11: ...])
    i0 = s.index('e.run(pso("attn_prep16"), [BIG, KP, VP')
    i1 = s.index("\n", s.index("offsets: [11: p0 * 64 * 2 * 4])", i0))
    s = s[:i1] + " }" + s[i1:]
    rep('''            if QKT.isEmpty { e.run(pso("attn_qk_h"), [QA, l.Kc, P16], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128)) }''',
        '''            if QK8 { e.run(pso("attn_qk8i"), [Q8B, l.K8c, P16, nil, nil, nil, QSB, l.KSc], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128)) }
            else if QKT.isEmpty { e.run(pso("attn_qk_h"), [QA, l.Kc, P16], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128)) }''')
    w.write_text(s); print("swift patched")
