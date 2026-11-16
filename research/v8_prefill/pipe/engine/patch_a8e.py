"""A8 step e (16 Nov 2026): KV8 (the GPU's attention on the ANE's own INT8 KV segments) at the QK8 rate and without
wasted work. attn_prep8q = attn_prep8 that also writes Q8 (int8 per (head, token), scale / 16 folded into QS);
attn_qk8is = INT8 x INT8 scores straight from an ANE key segment (dim-major, the attn_qk8 view), S = acc x QS (the
softmax then applies the key scales as before). The KV8 branch of encodeMixer now honours core (A8 attention core) and
A8NOOUT, and uses the INT8 o projection (I8 / I8A) when the full mixer runs. Engine env QK8=1 selects the INT8 scores.
Backups: *.pre_a8e.*"""
from pathlib import Path
E = Path(__file__).resolve().parent
k = E / "kernels.metal"; s = k.read_text()
if "attn_qk8is" not in s:
    a = "kernel void attn_prep8(device const float *qg [[buffer(0)]],"
    assert s.count(a) == 1
    i0 = s.index(a); i1 = s.index("\n}\n", i0) + 3
    body = s[i0:i1]
    q8 = body.replace("kernel void attn_prep8(", "kernel void attn_prep8q(")
    q8 = q8.replace('''                       uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],''',
                    '''                       device int8_t *Q8 [[buffer(14)]], device float *QS [[buffer(15)]],
                       uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],''')
    old = '''        if (h < 24) { Q[(h * T + t) * 256 + d] = half(v * 0.0625f); return; }   // Q / 16 (scores vs INT8 codes stay in fp16; = the 1/sqrt(256) scale)'''
    assert q8.count(old) == 1
    q8 = q8.replace(old, '''        if (h < 24) {   // Q / 16 as before, and INT8 Q: code = rint(q / sq), QS = sq / 16 (score = acc x QS)
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
        }''')
    assert "attn_prep8q" in q8 and "Q8[" in q8
    q8 += r'''// QK8 on an ANE key segment: S tile = (Q8 K8^T) x QS, int32 accumulate; K8 as attn_qk8 (dim-major, bound at the
// tile's first key). Grid (n / 64, 6T / 64, 4).
kernel void attn_qk8is(device int8_t *Q8 [[buffer(0)]], device int8_t *K8 [[buffer(1)]], device half *S [[buffer(2)]],
                       constant uint2 &kst [[buffer(3)]], constant uint &TK [[buffer(4)]], device const float *QS [[buffer(5)]],
                       uint3 tg [[threadgroup_position_in_grid]]) {
    uint kt = tg.x, rt = tg.y, gh = tg.z, r0 = gh * 6 * T + rt * 64;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tQ(Q8 + r0 * 256, dextents<int32_t, 2>(256, 64));
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK(K8 + gh * kst.x + kt * 64, dextents<int32_t, 2>(64, 256),
                                                                  array<int32_t, 2>{1, int(kst.y)});
    constexpr auto d = matmul2d_descriptor(64, 64, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    auto cI = op.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
    op.run(tQ, tK, cI);
    for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
        if (!cI.is_valid_element(i)) continue;
        auto ix = cI.get_multidimensional_index(i);
        uint key = ix[0], row = ix[1];
        S[(r0 + row) * TK + kt * 64 + key] = half(float(cI[i]) * QS[r0 + row]);
    }
}
'''
    s = s[:i1] + q8 + s[i1:]
    k.write_text(s); print("kernels patched")
w = E / "gpu_v8.swift"; s = w.read_text()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:90]); s = s.replace(a, b)
if "attn_qk8is" not in s:
    rep('''            e.run(pso("attn_prep8"), [BIG, KP, VP, l.bufs["qn"], l.bufs["kn"], QA, g0.k, g0.v, g0.ks, g0.vs, nil, nil, nil, csAll],
                  [(SIMD4<UInt32>(g0.kh, g0.kd, g0.vh, g0.vr), 10), (SIMD2<UInt32>(g0.sh, UInt32(p0 - g0.start)), 11), (eps, 12)],
                  grid: sz(32, T), tg: sz(256), offsets: [13: p0 * 64 * 2 * 4])''',
        '''            e.run(pso(QK8 ? "attn_prep8q" : "attn_prep8"), [BIG, KP, VP, l.bufs["qn"], l.bufs["kn"], QA, g0.k, g0.v, g0.ks, g0.vs, nil, nil, nil, csAll, Q8B, QSB],
                  [(SIMD4<UInt32>(g0.kh, g0.kd, g0.vh, g0.vr), 10), (SIMD2<UInt32>(g0.sh, UInt32(p0 - g0.start)), 11), (eps, 12)],
                  grid: sz(32, T), tg: sz(256), offsets: [13: p0 * 64 * 2 * 4])''')
    rep('''                e.run(pso("attn_qk8"), [QA, g.k, P16], [(SIMD2<UInt32>(g.kh, g.kd), 3), (UInt32(TK), 4)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128), offsets: [1: off])''',
        '''                if QK8 { e.run(pso("attn_qk8is"), [Q8B, g.k, P16, nil, nil, QSB], [(SIMD2<UInt32>(g.kh, g.kd), 3), (UInt32(TK), 4)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128), offsets: [1: off]) }
                else { e.run(pso("attn_qk8"), [QA, g.k, P16], [(SIMD2<UInt32>(g.kh, g.kd), 3), (UInt32(TK), 4)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128), offsets: [1: off]) }''')
    rep('''            e.run(pso("attn_finish"), [OR, MD, BIG, ON], grid: sz(256, T, 24), tg: sz(64, 1, 1), groups: false)
            rotate(e, l, "mix_out", ON, ON16, 6144)
            lut(e, l, "o", ON16, EPI ? X : Y, acc: EPI)
            if !EPI && !deferAdd { e.run(pso("add_into"), [X, Y], grid: sz(HID * T), tg: sz(256), groups: false) }
            return''', '''            e.run(pso("attn_finish"), [OR, MD, BIG, ON], grid: sz(256, T, 24), tg: sz(64, 1, 1), groups: false)
            if core { return }   // A8 attention core: AO (ANE) does the rest
            rotate(e, l, "mix_out", ON, ON16, 6144)
            if A8NOOUT { return }
            if I8 && I8A { e.run(pso("quant8_blk"), [ON16, X8B, XSB], grid: sz(6144 / 1024, T / 32), tg: sz(256)); lutI8(e, l, "o", X8B, XSB, ON16, EPI ? X : Y, acc: EPI) }
            else { lut(e, l, "o", ON16, EPI ? X : Y, acc: EPI) }
            if !EPI && !deferAdd { e.run(pso("add_into"), [X, Y], grid: sz(HID * T), tg: sz(256), groups: false) }
            return''')
    w.write_text(s); print("swift patched")
