"""A8 step d (16 Nov 2026): a layer's mixer OUTPUT projection on the ANE (AO program, DeltaNet out_proj or attention
o_proj, 6144 -> 5120). encodeMixer stops before the output projection while A8NOOUT is set (the rotated mixer output
is left in ON16, X stays the residual). C API: v8_a8_out_setup (int8 palettes [64][64][2], sorts the out / o tables),
v8_a8_out_expand (wo [6][1024][5120], q2), v8_submit_a8mixo (pair call: residual + previous FFN, the mixer without its
output projection, ON16 -> INT8 rows for AO); v8_a8_att_expand skips o when wo < 0. Backups: *.pre_a8d.*"""
from pathlib import Path
E = Path(__file__).resolve().parent
w = E / "gpu_v8.swift"; s = w.read_text()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80]); s = s.replace(a, b)
if "v8_submit_a8mixo" not in s:
    rep('''        if I8 { e.run(pso("quant8_blk"), [ON16, X8B, XSB], grid: sz(6144 / 1024, T / 32), tg: sz(256)); lutI8(e, l, "out", X8B, XSB, ON16, EPI ? X : Y, acc: EPI) }''',
        '''        if A8NOOUT { return }   // A8: the output projection runs on the ANE (ON16 -> AO)
        if I8 { e.run(pso("quant8_blk"), [ON16, X8B, XSB], grid: sz(6144 / 1024, T / 32), tg: sz(256)); lutI8(e, l, "out", X8B, XSB, ON16, EPI ? X : Y, acc: EPI) }''')
    rep('''        rotate(e, l, "mix_out", ON, ON16, 6144)
        if I8 && I8A {''', '''        rotate(e, l, "mix_out", ON, ON16, 6144)
        if A8NOOUT { return }   // A8: the output projection runs on the ANE (ON16 -> AO)
        if I8 && I8A {''')
    rep('''    for (mi, (key, dst, mode)) in [("q", wq, 0), ("o", wo, 1)].enumerated() {''',
        '''    for (mi, (key, dst, mode)) in [("q", wq, 0), ("o", wo, 1)].enumerated() where dst >= 0 {''')
    s += r'''
// ---- A8 output projections on the ANE (patch_a8d.py) ----
var A8NOOUT = false
var A8OPAL: MTLBuffer? = nil
/// pal: int8 [64][64][2] (prep_a8out.py out_pal8.bin). Sorts every layer's out (DeltaNet) / o (attention) table by row.
@_cdecl("v8_a8_out_setup")
public func v8_a8_out_setup(_ p: UnsafeRawPointer, _ n: Int) -> Int32 {
    A8OPAL = dev.makeBuffer(bytes: p, length: n, options: .storageModeShared)
    for l in layers {
        let key = l.m.kind == "gdn" ? "out" : "o"
        guard let mt = l.m.mats[key], let tb = l.bufs["tab_" + key], mt.G * mt.cd == 32, mt.bits == 4, mt.cin == 6144, a8Sort(tb) else { return -2 }
    }
    return A8OPAL == nil ? -1 : 0
}
/// Layer `layer`'s output projection (LUT part) as INT8 codes into AO's wo [6][1024][5120], on q2. Ticket.
@_cdecl("v8_a8_out_expand")
public func v8_a8_out_expand(_ layer: Int32, _ wo: Int32) -> Int32 {
    guard let k = layerIndex[Int(layer)], let pal = A8OPAL else { return -1 }
    let l = layers[k], key = l.m.kind == "gdn" ? "out" : "o", mt = l.m.mats[key]!, tb = l.bufs["tab_" + key]!
    let cb = q2.makeCommandBuffer()!, ce = cb.makeComputeCommandEncoder()!
    var prm = SIMD4<UInt32>(UInt32(mt.cin), UInt32(mt.G), 1, 0)
    var fm = SIMD4<UInt32>(UInt32(mt.bits), UInt32(mt.cd), UInt32(mt.G * mt.bits / 8), 0)
    ce.setComputePipelineState(pso("a8_expand_g"))
    ce.setBuffer(kern[mt.hwx], offset: 0, index: 0); ce.setBuffer(tb, offset: 0, index: 1)
    ce.setBuffer(pal, offset: Int(layer) * 128, index: 2); ce.setBuffer(wrapped[Int(wo)], offset: 0, index: 3)
    ce.setBytes(&prm, length: 16, index: 4); ce.setBytes(&fm, length: 16, index: 5)
    ce.dispatchThreadgroups(sz(mt.cin / 32, tb.length / 12 / 32), threadsPerThreadgroup: sz(1024))
    ce.endEncoding()
    inflightLock.lock(); ticket += 1; let t = ticket; inflight[t] = cb; inflightLock.unlock()
    cb.commit()
    return t
}
/// Pair call: X = q + y rows [yrow, yrow + 128) (the previous FFN; y < 0: none) -> qA / qB, the layer's mixer without
/// its output projection, the rotated mixer output (ON16) -> INT8 rows [orow, orow + 128) of o8 (os bytes per row) +
/// per (token, 1024-block) scales x 128 in to (ts halves per row). Ticket.
@_cdecl("v8_submit_a8mixo")
public func v8_submit_a8mixo(_ layer: Int32, _ qA: Int32, _ qB: Int32, _ p0: Int32, _ y: Int32, _ yrow: Int32, _ o8: Int32,
                             _ to: Int32, _ orow: Int32, _ os: Int32, _ ts: Int32) -> Int32 {
    guard let k = layerIndex[Int(layer)], T == 128 else { return -1 }
    let l = layers[k], cb = q.makeCommandBuffer()!, e = Enc(mixEncoder(cb))
    e.run(pso("from_half_pair"), [wrapped[Int(qA)], wrapped[Int(qB)], X], grid: sz(HID * T), tg: sz(256), groups: false)
    if y >= 0 { e.run(pso("a8_add_tm"), [X, wrapped[Int(y)]], [(SIMD4<UInt32>(UInt32(yrow), A8YS, 0, 0), 2)], grid: sz(HID / 32, T / 32), tg: sz(256)) }
    A8NOOUT = true
    encodeMixer(e, l, p0: Int(p0), deferAdd: false)
    A8NOOUT = false
    e.run(pso("to_half_pair"), [X, wrapped[Int(qA)], wrapped[Int(qB)]], grid: sz(HID * T), tg: sz(256), groups: false)
    e.run(pso("a8_quant_tm"), [ON16, wrapped[Int(o8)], wrapped[Int(to)]], [(SIMD4<UInt32>(UInt32(orow), UInt32(os), UInt32(ts), 1024), 3)],
          grid: sz(6, T), tg: sz(256))
    e.e.endEncoding()
    inflightLock.lock(); ticket += 1; let t = ticket; inflight[t] = cb; inflightLock.unlock()
    cb.commit()
    return t
}
'''
    w.write_text(s); print("swift patched")
