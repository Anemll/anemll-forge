"""KV8 hybrid K8V16 (16 Nov 2026), engine env K8V16=1 (with KV8=1 QK8=1): the keys only in the ANE's INT8 cache
(INT8 x INT8 scores straight from the segments, key scales folded in: attn_qk8isk), the values ALSO kept as fp16 on
the GPU (Vc, +32 KB / token) so PV runs the fp16 kernel (attn_pv_w; the half x INT8 PV was +40%) after the plain
softmax (attn_softmax8n). The ANE cache still gets both INT8 K and V (decode). attn_prep8qv = attn_prep8q + the fp16
V rows into Vc. v8_kv8_v16_fill(upto): Vc rows [0, upto) of every attention layer from the ANE's INT8 V (a warm start
from a bridge state). Backups *.pre_k8v16.*"""
from pathlib import Path
E = Path(__file__).resolve().parent
k = E / "kernels.metal"; s = k.read_text()
if "attn_prep8qv" not in s:
    a = "kernel void attn_prep8q(device const float *qg [[buffer(0)]],"
    i0 = s.index(a); i1 = s.index("\n}\n", i0) + 3
    body = s[i0:i1].replace("kernel void attn_prep8q(", "kernel void attn_prep8qv(")
    old = "                       device int8_t *Q8 [[buffer(14)]], device float *QS [[buffer(15)]],"
    assert body.count(old) == 1
    body = body.replace(old, old + "\n                       device half *Vc [[buffer(16)]], constant uint2 &vl [[buffer(17)]],")
    old = "    else { uint vh = h - 28; V8[vh * st.z + r * st.w + d] = code; if (d == 0) VS[vh * st2.x + r] = sc; }"
    assert body.count(old) == 1
    body = body.replace(old, "    else { uint vh = h - 28; V8[vh * st.z + r * st.w + d] = code; if (d == 0) VS[vh * st2.x + r] = sc;\n"
                             "           Vc[(vh * vl.x + vl.y + t) * 256 + d] = half(hv); }   // + the fp16 row (vl = Lmax, p0)")
    s = s[:i1] + "// K8V16 (patch_k8v16.py): attn_prep8q + the fp16 V rows into the GPU's Vc\n" + body + r'''// Vc rows [r0, r0 + n) of one attention layer from an ANE INT8 V segment (row r of the segment = position s0 + r);
// prm = (Lmax, s0, vh head stride bytes, vr row stride bytes), sh = scale head stride (halves). Grid (256, n, 4).
kernel void kv8_v_to_v16(device const int8_t *V8 [[buffer(0)]], device const half *VS [[buffer(1)]], device half *Vc [[buffer(2)]],
                         constant uint4 &prm [[buffer(3)]], constant uint &sh [[buffer(4)]], uint3 p [[thread_position_in_grid]]) {
    uint d = p.x, r = p.y, h = p.z;
    Vc[(h * prm.x + prm.y + r) * 256 + d] = half(float(V8[h * prm.z + r * prm.w + d]) * float(VS[h * sh + r]));
}
''' + s[i1:]
    k.write_text(s); print("kernels patched")
w = E / "gpu_v8.swift"; s = w.read_text()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:90]); s = s.replace(a, b)
if "K8V16" not in s:
    rep('''let KV8D = ProcessInfo.processInfo.environment["KV8D"] == "1"''', '''let KV8D = ProcessInfo.processInfo.environment["KV8D"] == "1"
let K8V16 = KV8 && QK8 && ProcessInfo.processInfo.environment["K8V16"] == "1"   // INT8 K (ANE) + fp16 V copy (patch_k8v16.py)''')
    rep('''        if m.kind == "gdn" { S = buf(48 * 128 * 128 * 4); CS = buf(3 * 10240 * 4) } else if !KV8 {''',
        '''        if m.kind == "gdn" { S = buf(48 * 128 * 128 * 4); CS = buf(3 * 10240 * 4) } else if KV8 && K8V16 { Vc = buf(4 * Lmax * 256 * 2) } else if !KV8 {''')
    rep('''            e.run(pso(QK8 ? "attn_prep8q" : "attn_prep8"), [BIG, KP, VP, l.bufs["qn"], l.bufs["kn"], QA, g0.k, g0.v, g0.ks, g0.vs, nil, nil, nil, csAll, Q8B, QSB],
                  [(SIMD4<UInt32>(g0.kh, g0.kd, g0.vh, g0.vr), 10), (SIMD2<UInt32>(g0.sh, UInt32(p0 - g0.start)), 11), (eps, 12)],
                  grid: sz(32, T), tg: sz(256), offsets: [13: p0 * 64 * 2 * 4])''',
        '''            e.run(pso(K8V16 ? "attn_prep8qv" : QK8 ? "attn_prep8q" : "attn_prep8"), [BIG, KP, VP, l.bufs["qn"], l.bufs["kn"], QA, g0.k, g0.v, g0.ks, g0.vs, nil, nil, nil, csAll, Q8B, QSB, l.Vc],
                  [(SIMD4<UInt32>(g0.kh, g0.kd, g0.vh, g0.vr), 10), (SIMD2<UInt32>(g0.sh, UInt32(p0 - g0.start)), 11), (eps, 12),
                   (SIMD2<UInt32>(UInt32(Lmax), UInt32(p0)), 17)],
                  grid: sz(32, T), tg: sz(256), offsets: [13: p0 * 64 * 2 * 4])''')
    rep('''                if QK8 && KV8D { e.run(pso("attn_qk8isk"),''', '''                if QK8 && (KV8D || K8V16) { e.run(pso("attn_qk8isk"),''')
    rep('''                if QK8 && KV8D {
                    e.run(pso("attn_softmax8n"), [P16, STT], [(UInt32(TK), 2), (UInt32(a0), 3), (UInt32(n), 4), (UInt32(p0), 5)], grid: sz(24 * T), tg: sz(256))''',
        '''                if K8V16 {   // plain softmax, fp16-V PV over the GPU's copy (absolute key rows)
                    e.run(pso("attn_softmax8n"), [P16, STT], [(UInt32(TK), 2), (UInt32(a0), 3), (UInt32(n), 4), (UInt32(p0), 5)], grid: sz(24 * T), tg: sz(256))
                    e.run(pso("attn_pv_w"), [P16, l.Vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(2, 6 * T / 64, 4), tg: sz(128))
                } else if QK8 && KV8D {
                    e.run(pso("attn_softmax8n"), [P16, STT], [(UInt32(TK), 2), (UInt32(a0), 3), (UInt32(n), 4), (UInt32(p0), 5)], grid: sz(24 * T), tg: sz(256))''')
    s += r'''
// ---- K8V16 (patch_k8v16.py): the GPU's fp16 V copy from the ANE's INT8 V (warm start) ----
/// Every attention layer's Vc rows [0, upto) from its registered ANE segments, one command buffer. GPU ms.
@_cdecl("v8_kv8_v16_fill")
public func v8_kv8_v16_fill(_ upto: Int32) -> Double {
    guard K8V16 else { return -1 }
    let cb = q.makeCommandBuffer()!, ce = cb.makeComputeCommandEncoder()!
    for l in layers where l.m.kind != "gdn" {
        guard let vc = l.Vc else { return -2 }
        for g in l.segs where g.start < Int(upto) {
            let n = min(g.rows, Int(upto) - g.start)
            var prm = SIMD4<UInt32>(UInt32(Lmax), UInt32(g.start), g.vh, g.vr), sh = g.sh
            ce.setComputePipelineState(pso("kv8_v_to_v16"))
            ce.setBuffer(g.v, offset: 0, index: 0); ce.setBuffer(g.vs, offset: 0, index: 1); ce.setBuffer(vc, offset: 0, index: 2)
            ce.setBytes(&prm, length: 16, index: 3); ce.setBytes(&sh, length: 4, index: 4)
            ce.dispatchThreads(sz(256, n, 4), threadsPerThreadgroup: sz(256))
        }
    }
    ce.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return cb.error != nil ? -3 : (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}
'''
    w.write_text(s); print("swift patched")
