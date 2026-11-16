"""KV8 attention at the fp16 path's speed (16 Nov 2026), engine env KV8D=1 (with KV8=1 QK8=1): the KV8 core was 12%
slower per call at 64K (profile: attn_pv8 half x int8 +40% over the fp16 attn_pv_w, attn_softmax8 +22% for the per-key
scales). attn_qk8isk = attn_qk8is with the key scales folded in (S = acc x QS x KS[key]); attn_softmax8n = the softmax
without per-key scales (P = plain probabilities, no x4096 needed); attn_pv8d = PV with V dequantized on chip: per
64-key chunk the INT8 V rows x their value scales -> fp16 in threadgroup memory, then half x half multiply-accumulate
(V still crosses DRAM as 1 byte). Backups: *.pre_kv8b.*"""
from pathlib import Path
E = Path(__file__).resolve().parent
k = E / "kernels.metal"; s = k.read_text()
if "attn_pv8d" not in s:
    s += r'''
// ---- KV8 attention, fp16-rate path (patch_kv8b.py, KV8D=1) ----
// S tile = (Q8 K8^T) x QS x KS: INT8 scores on an ANE key segment with the key scales folded in (KS bound at the
// tile's first key, head stride ssh halves). Grid (n / 64, 6T / 64, 4).
kernel void attn_qk8isk(device int8_t *Q8 [[buffer(0)]], device int8_t *K8 [[buffer(1)]], device half *S [[buffer(2)]],
                        constant uint2 &kst [[buffer(3)]], constant uint &TK [[buffer(4)]], device const float *QS [[buffer(5)]],
                        device const half *KS [[buffer(6)]], constant uint &ssh [[buffer(7)]],
                        uint3 tg [[threadgroup_position_in_grid]]) {
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
        S[(r0 + row) * TK + kt * 64 + key] = half(float(cI[i]) * QS[r0 + row] * float(ks[key]));
    }
}
// softmax over a tile's (already scaled) scores; P = exp(s - m); stats (m, sum) as attn_softmax8 (no per-key scales)
kernel void attn_softmax8n(device half *S [[buffer(0)]], device float *st [[buffer(1)]], constant uint &TK [[buffer(2)]],
                           constant uint &a [[buffer(3)]], constant uint &n [[buffer(4)]], constant uint &p0 [[buffer(5)]],
                           uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    device half *row = S + r * TK;
    int lim = int(p0 + (r % T) + 1) - int(a);
    uint nv = uint(clamp(lim, 0, int(n)));
    float m = -INFINITY;
    for (uint j = tid; j < nv; j += 256) m = max(m, float(row[j]));
    m = simd_max(m);
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0;
    for (uint j = tid; j < n; j += 256) {
        float e = j < nv ? exp(float(row[j]) - m) : 0.0f;
        row[j] = half(e); s += e;
    }
    s = simd_sum(s);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float tot = 0; for (uint i = 0; i < 8; i++) tot += part[i]; st[2 * r] = m; st[2 * r + 1] = tot; }
}
// O tile (64 rows x 128 dims) = P (64 rows x n keys) x V, V dequantized on chip per 64-key chunk (INT8 code x the
// key's value scale -> fp16 threadgroup tile). V8 / VS bound at the tile's first key (V8: head stride vst.x, row
// stride vst.y bytes; VS: head stride ssh halves). Grid (2, 6T / 64, 4), 128 threads.
kernel void attn_pv8d(device half *P [[buffer(0)]], device int8_t *V8 [[buffer(1)]], device float *O [[buffer(2)]],
                      constant uint2 &vst [[buffer(3)]], constant uint &TK [[buffer(4)]], constant uint &n [[buffer(5)]],
                      device const half *VS [[buffer(6)]], constant uint &ssh [[buffer(7)]],
                      uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half Vt[64 * 128];
    uint dt = tg.x, rt = tg.y, gh = tg.z, r0 = gh * 6 * T + rt * 64;
    device const int8_t *vb = V8 + gh * vst.x + dt * 128;
    device const half *vsb = VS + gh * ssh;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tV(Vt, dextents<int32_t, 2>(128, 64));
    constexpr auto d = matmul2d_descriptor(64, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    using TPt = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    auto cO = op.template get_destination_cooperative_tensor<TPt, decltype(tV), float>();
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0;
    for (uint k0 = 0; k0 < n; k0 += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 64 * 128 / 4; i += 128) {
            uint key = (i * 4) / 128, c = (i * 4) % 128;
            char4 v4 = *(device const char4 *)(vb + (k0 + key) * vst.y + c);
            float sc = float(vsb[k0 + key]);
            Vt[key * 128 + c] = half(float(v4.x) * sc); Vt[key * 128 + c + 1] = half(float(v4.y) * sc);
            Vt[key * 128 + c + 2] = half(float(v4.z) * sc); Vt[key * 128 + c + 3] = half(float(v4.w) * sc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        TPt tP(P + r0 * TK + k0, dextents<int32_t, 2>(64, 64), array<int32_t, 2>{1, int(TK)});
        op.run(tP, tV, cO);
    }
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        O[(r0 + ix[1]) * 256 + dt * 128 + ix[0]] = cO[i];
    }
}
'''
    k.write_text(s); print("kernels patched")
w = E / "gpu_v8.swift"; s = w.read_text()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:90]); s = s.replace(a, b)
if "KV8D" not in s:
    rep('''let QK8 = ProcessInfo.processInfo.environment["QK8"] == "1"''',
        '''let QK8 = ProcessInfo.processInfo.environment["QK8"] == "1"
let KV8D = ProcessInfo.processInfo.environment["KV8D"] == "1"   // KV8 core: key scales in QK, V dequantized on chip (patch_kv8b.py)''')
    rep('''                if QK8 { e.run(pso("attn_qk8is"), [Q8B, g.k, P16, nil, nil, QSB], [(SIMD2<UInt32>(g.kh, g.kd), 3), (UInt32(TK), 4)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128), offsets: [1: off]) }''',
        '''                if QK8 && KV8D { e.run(pso("attn_qk8isk"), [Q8B, g.k, P16, nil, nil, QSB, g.ks], [(SIMD2<UInt32>(g.kh, g.kd), 3), (UInt32(TK), 4), (g.sh, 7)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128), offsets: [1: off, 6: off * 2]) }
                else if QK8 { e.run(pso("attn_qk8is"), [Q8B, g.k, P16, nil, nil, QSB], [(SIMD2<UInt32>(g.kh, g.kd), 3), (UInt32(TK), 4)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128), offsets: [1: off]) }''')
    rep('''                e.run(pso("attn_softmax8"), [P16, STT, nil, nil, nil, nil, g.ks, g.vs], [(UInt32(TK), 2), (UInt32(a0), 3), (UInt32(n), 4), (UInt32(p0), 5), (g.sh, 8)],
                      grid: sz(24 * T), tg: sz(256), offsets: [6: off * 2, 7: off * 2])
                e.run(pso("attn_pv8"), [P16, g.v, OA], [(SIMD2<UInt32>(g.vh, g.vr), 3), (UInt32(TK), 4), (UInt32(n), 5)], grid: sz(2, 6 * T / 64, 4), tg: sz(128),
                      offsets: [1: off * Int(g.vr)])''', '''                if QK8 && KV8D {
                    e.run(pso("attn_softmax8n"), [P16, STT], [(UInt32(TK), 2), (UInt32(a0), 3), (UInt32(n), 4), (UInt32(p0), 5)], grid: sz(24 * T), tg: sz(256))
                    e.run(pso("attn_pv8d"), [P16, g.v, OA, nil, nil, nil, g.vs], [(SIMD2<UInt32>(g.vh, g.vr), 3), (UInt32(TK), 4), (UInt32(n), 5), (g.sh, 7)],
                          grid: sz(2, 6 * T / 64, 4), tg: sz(128), offsets: [1: off * Int(g.vr), 6: off * 2])
                } else {
                e.run(pso("attn_softmax8"), [P16, STT, nil, nil, nil, nil, g.ks, g.vs], [(UInt32(TK), 2), (UInt32(a0), 3), (UInt32(n), 4), (UInt32(p0), 5), (g.sh, 8)],
                      grid: sz(24 * T), tg: sz(256), offsets: [6: off * 2, 7: off * 2])
                e.run(pso("attn_pv8"), [P16, g.v, OA], [(SIMD2<UInt32>(g.vh, g.vr), 3), (UInt32(TK), 4), (UInt32(n), 5)], grid: sz(2, 6 * T / 64, 4), tg: sz(128),
                      offsets: [1: off * Int(g.vr)])
                }''')
    w.write_text(s); print("swift patched")
