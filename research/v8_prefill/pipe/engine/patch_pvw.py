"""PV with the full 256 dims per threadgroup (16 Nov 2026): the probability matrix is read once instead of twice
(attn_pv_w / attn_pv8 split the dims in two 128-wide halves). attn_pv_ww (fp16 V), attn_pv8w (INT8 V, P' = p x value
scale as attn_softmax8). 8 simdgroups, grid (1, 6T / 64, 4). Engine env PVWW=1. Backups: *.pre_pvw.*"""
from pathlib import Path
E = Path(__file__).resolve().parent
k = E / "kernels.metal"; s = k.read_text()
if "attn_pv_ww" not in s:
    s += r'''
// ---- PV over all 256 dims per threadgroup (patch_pvw.py, PVWW=1): P read once ----
kernel void attn_pv_ww(device half *P [[buffer(0)]], device half *Vc [[buffer(1)]], device float *O [[buffer(2)]],
                       constant uint &Lmax [[buffer(3)]], constant uint &a [[buffer(4)]], constant uint &TK [[buffer(5)]],
                       constant uint &n [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]]) {
    uint rt = tg.y, gh = tg.z;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tP(P + (gh * 6 * T + rt * 64) * TK, dextents<int32_t, 2>(int(n), 64),
                                                                array<int32_t, 2>{1, int(TK)});
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tV(Vc + (gh * Lmax + a) * 256, dextents<int32_t, 2>(256, int(n)),
                                                                array<int32_t, 2>{1, 256});
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tO(O + (gh * 6 * T + rt * 64) * 256, dextents<int32_t, 2>(256, 64),
                                                                 array<int32_t, 2>{1, 256});
    constexpr auto d = matmul2d_descriptor(64, 256, static_cast<int>(dynamic_extent), false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<8>> op;
    op.run(tP, tV, tO);
}
kernel void attn_pv8w(device half *P [[buffer(0)]], device int8_t *V8 [[buffer(1)]], device float *O [[buffer(2)]],
                      constant uint2 &vst [[buffer(3)]], constant uint &TK [[buffer(4)]], constant uint &n [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]]) {
    uint rt = tg.y, gh = tg.z;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tP(P + (gh * 6 * T + rt * 64) * TK, dextents<int32_t, 2>(int(n), 64),
                                                                array<int32_t, 2>{1, int(TK)});
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tV(V8 + gh * vst.x, dextents<int32_t, 2>(256, int(n)),
                                                                  array<int32_t, 2>{1, int(vst.y)});
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tO(O + (gh * 6 * T + rt * 64) * 256, dextents<int32_t, 2>(256, 64),
                                                                 array<int32_t, 2>{1, 256});
    constexpr auto d = matmul2d_descriptor(64, 256, static_cast<int>(dynamic_extent), false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<8>> op;
    op.run(tP, tV, tO);
}
'''
    k.write_text(s); print("kernels patched")
w = E / "gpu_v8.swift"; s = w.read_text()
def rep(a, b, count=1):
    global s
    assert s.count(a) == count, (s.count(a), a[:90]); s = s.replace(a, b)
if "PVWW" not in s:
    rep('''let KV8D = ProcessInfo.processInfo.environment["KV8D"] == "1"''', '''let KV8D = ProcessInfo.processInfo.environment["KV8D"] == "1"
let PVWW = ProcessInfo.processInfo.environment["PVWW"] == "1"   // PV over 256 dims per threadgroup (patch_pvw.py)''')
    rep('''                e.run(pso("attn_pv_w"), [P16, l.Vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(2, 6 * T / 64, 4), tg: sz(128))''',
        '''                if PVWW { e.run(pso("attn_pv_ww"), [P16, l.Vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(1, 6 * T / 64, 4), tg: sz(256)) }
                else { e.run(pso("attn_pv_w"), [P16, l.Vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(2, 6 * T / 64, 4), tg: sz(128)) }''', count=3)
    rep('''                e.run(pso("attn_pv8"), [P16, g.v, OA], [(SIMD2<UInt32>(g.vh, g.vr), 3), (UInt32(TK), 4), (UInt32(n), 5)], grid: sz(2, 6 * T / 64, 4), tg: sz(128),
                      offsets: [1: off * Int(g.vr)])''', '''                if PVWW { e.run(pso("attn_pv8w"), [P16, g.v, OA], [(SIMD2<UInt32>(g.vh, g.vr), 3), (UInt32(TK), 4), (UInt32(n), 5)], grid: sz(1, 6 * T / 64, 4), tg: sz(256),
                      offsets: [1: off * Int(g.vr)]) }
                else { e.run(pso("attn_pv8"), [P16, g.v, OA], [(SIMD2<UInt32>(g.vh, g.vr), 3), (UInt32(TK), 4), (UInt32(n), 5)], grid: sz(2, 6 * T / 64, 4), tg: sz(128),
                      offsets: [1: off * Int(g.vr)]) }''')
    w.write_text(s); print("swift patched")
