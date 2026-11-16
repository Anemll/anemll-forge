"""A8 step c (16 Nov 2026): attention projections on the ANE (v8_work/w8a8/a8_att_build.py AQKV / AO). GPU side:
a8_expand_g (tiled expander for any 32-row subgroup: 6-bit cd 2 G 16 FFN, 4-bit cd 2 G 16 and 4-bit cd 1 G 32 q / o),
a8_load_tm (the ANE's token-major [blocks][rows][1024] output -> fp32 channel-major scratch), a8_quant_tm with a part
width (prm.w; 0 = 2560). C API: v8_a8_att_setup (q / o int8 palettes, sorts the q / o tables), v8_a8_att_expand
(wq [12][5120][1024], wo [6][1024][5120]), v8_submit_a8 flag 4 (the FFN-input path with ln1 / mix_in: the attention
input), v8_submit_a8core (q / k / v from the ANE -> attention core -> mix_out rotation -> INT8 rows for AO).
Backups: *.pre_a8c.*"""
from pathlib import Path
E = Path(__file__).resolve().parent
k = E / "kernels.metal"; s = k.read_text()
if "a8_expand_g" not in s:
    a = '''    float v[10], m = 0;
    for (uint j = 0; j < 10; j++) { v[j] = float(h[(hf * 2560 + j * 256 + tid) * T + t]); m = max(m, abs(v[j])); }'''
    b = '''    const uint PW = prm.w ? prm.w : 2560, NJ = PW / 256;   // part width (channels per scale)
    float v[10], m = 0;
    for (uint j = 0; j < NJ; j++) { v[j] = float(h[(hf * PW + j * 256 + tid) * T + t]); m = max(m, abs(v[j])); }'''
    assert s.count(a) == 1; s = s.replace(a, b)
    a = '''    device int8_t *row = x8 + (prm.x + t) * prm.y + hf * 2560;
    for (uint j = 0; j < 10; j++) row[j * 256 + tid] = int8_t(clamp(rint(v[j] * inv), -127.0f, 127.0f));'''
    b = '''    device int8_t *row = x8 + (prm.x + t) * prm.y + hf * PW;
    for (uint j = 0; j < NJ; j++) row[j * 256 + tid] = int8_t(clamp(rint(v[j] * inv), -127.0f, 127.0f));'''
    assert s.count(a) == 1; s = s.replace(a, b)
    s += r'''
// A8 general tiled expander (patch_a8c.py): any subgroup of 32 output rows (G x cd = 32), BITS 4 / 6, cd 1 / 2; tile =
// 32 row-consecutive subgroups x 32 columns, 1024 threads. prm = (cin, G, mode, xoff), fm = (bits, cd, bytes per column).
kernel void a8_expand_g(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device const char2 *pal [[buffer(2)]],
                        device int8_t *dst [[buffer(3)]], constant uint4 &prm [[buffer(4)]], constant uint4 &fm [[buffer(5)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uchar ti[32 * 32 * 16];
    threadgroup char2 tp[64];
    const uint BITS = fm.x, CD = fm.y, BPC = fm.z, NI = 32 / CD, MASK = (1u << BITS) - 1;
    uint c0 = tg.x * 32, s0 = tg.y * 32;
    if (tid < 64) tp[tid] = pal[tid];
    {
        uint s = tid / 32, c = tid % 32;
        device const uchar *src = k + tab[3 * (s0 + s) + 1] + 64 + (c0 + c) * BPC;
        threadgroup uchar *d = ti + (s * 32 + c) * 16;
        for (uint q = 0; q < BPC; q++) d[q] = src[q];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint c = tid / 32, s = tid % 32;
    threadgroup const uchar *b = ti + (s * 32 + c) * 16;
    uint o0 = CD * tab[3 * (s0 + s) + 2], x = prm.w + c0 + c, cc = c0 + c;
    device int8_t *d = prm.z == 0 ? dst + (o0 >> 10) * (5120u * 1024u) + cc * 1024u + (o0 & 1023u)
                                  : dst + (x >> 10) * (1024u * 5120u) + (x & 1023u) * 5120u + o0;
    uchar ob[32];
    for (uint i = 0; i < NI; i++) {
        uint bit = i * BITS, by = bit >> 3, sh = bit & 7;
        uint v = (uint(b[by]) | (by + 1 < BPC ? uint(b[by + 1]) << 8 : 0u)) >> sh;
        char2 p = tp[v & MASK];
        if (CD == 2) { ob[2 * i] = uchar(p.x); ob[2 * i + 1] = uchar(p.y); } else { ob[i] = uchar(p.x); }
    }
    for (uint h = 0; h < 2; h++) {
        uint4 w;
        for (uint q = 0; q < 4; q++) w[q] = uint(ob[16 * h + 4 * q]) | (uint(ob[16 * h + 4 * q + 1]) << 8) | (uint(ob[16 * h + 4 * q + 2]) << 16) | (uint(ob[16 * h + 4 * q + 3]) << 24);
        *(device uint4 *)(d + 16 * h) = w;
    }
}
// x (fp32 channel-major [C][T]) = src (the ANE's token-major output [C / 1024 blocks][Tb rows][1024] fp16) rows r0 ..;
// prm = (r0, Tb). Grid (C / 32, T / 32) threadgroups of 256.
kernel void a8_load_tm(device float *x [[buffer(0)]], device const half *y [[buffer(1)]], constant uint4 &prm [[buffer(2)]],
                       uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tile[32][33];
    uint c0 = g.x * 32, t0 = g.y * 32;
    device const half *yb = y + ulong(c0 >> 10) * prm.y * 1024u;
    for (uint i = tid; i < 1024; i += 256) { uint tt = i / 32, cc = i % 32; tile[tt][cc] = float(yb[(prm.x + t0 + tt) * 1024u + ((c0 + cc) & 1023u)]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 1024; i += 256) { uint cc = i / 32, tt = i % 32; x[(c0 + cc) * T + t0 + tt] = tile[tt][cc]; }
}
'''
    k.write_text(s); print("kernels patched")
w = E / "gpu_v8.swift"; s = w.read_text()
if "v8_submit_a8core" not in s:
    # flag 4: the attention input (ln1 + mix_in rotation) instead of the FFN input (ln2 + mlp_in)
    a = '''        e.run(pso("rms_fwht4"), [X, l.bufs["ln2"], l.bufs["rot_mlp_in"], HQ16, H16, RP], [(UInt32(HID), 6), (man.eps, 7)],
              grid: sz(HID / 1024, T / 4), tg: sz(512))'''
    b = '''        let att = flags & 4 != 0
        e.run(pso("rms_fwht4"), [X, l.bufs[att ? "ln1" : "ln2"], l.bufs[att ? "rot_mix_in" : "rot_mlp_in"], HQ16, H16, RP], [(UInt32(HID), 6), (man.eps, 7)],
              grid: sz(HID / 1024, T / 4), tg: sz(512))'''
    assert s.count(a) == 1; s = s.replace(a, b)
    s += r'''
// ---- A8 attention projections on the ANE (patch_a8c.py) ----
var A8APAL: MTLBuffer? = nil
func a8Sort(_ tb: MTLBuffer) -> Bool {   // in place, by row; true when the rows are 32-row consecutive tiles
    let n = tb.length / 12, p = tb.contents().bindMemory(to: UInt32.self, capacity: 3 * n)
    var rows = (0..<n).map { (p[3 * $0], p[3 * $0 + 1], p[3 * $0 + 2]) }
    rows.sort { $0.2 < $1.2 }
    for (i, r) in rows.enumerated() { p[3 * i] = r.0; p[3 * i + 1] = r.1; p[3 * i + 2] = r.2 }
    return n % 32 == 0
}
/// pal: int8 [64][2 (q, o)][64][2] (prep_a8att.py att_pal8.bin). Sorts every attention layer's q / o tables by row.
@_cdecl("v8_a8_att_setup")
public func v8_a8_att_setup(_ p: UnsafeRawPointer, _ n: Int) -> Int32 {
    A8APAL = dev.makeBuffer(bytes: p, length: n, options: .storageModeShared)
    for l in layers where l.m.kind != "gdn" {
        for key in ["q", "o"] {
            guard let mt = l.m.mats[key], let tb = l.bufs["tab_" + key], mt.G * mt.cd == 32, mt.bits == 4, a8Sort(tb) else { return -2 }
        }
    }
    return A8APAL == nil ? -1 : 0
}
/// Attention layer `layer`'s q / o as INT8 codes into AQKV's wq [12][5120][1024] and AO's wo [6][1024][5120], on q2.
@_cdecl("v8_a8_att_expand")
public func v8_a8_att_expand(_ layer: Int32, _ wq: Int32, _ wo: Int32) -> Int32 {
    guard let k = layerIndex[Int(layer)], layers[k].m.kind != "gdn", let pal = A8APAL else { return -1 }
    let l = layers[k]
    let cb = q2.makeCommandBuffer()!, ce = cb.makeComputeCommandEncoder()!
    for (mi, (key, dst, mode)) in [("q", wq, 0), ("o", wo, 1)].enumerated() {
        let mt = l.m.mats[key]!, tb = l.bufs["tab_" + key]!
        var prm = SIMD4<UInt32>(UInt32(mt.cin), UInt32(mt.G), UInt32(mode), 0)
        var fm = SIMD4<UInt32>(UInt32(mt.bits), UInt32(mt.cd), UInt32(mt.G * mt.bits / 8), 0)
        ce.setComputePipelineState(pso("a8_expand_g"))
        ce.setBuffer(kern[mt.hwx], offset: 0, index: 0); ce.setBuffer(tb, offset: 0, index: 1)
        ce.setBuffer(pal, offset: (Int(layer) * 2 + mi) * 128, index: 2); ce.setBuffer(wrapped[Int(dst)], offset: 0, index: 3)
        ce.setBytes(&prm, length: 16, index: 4); ce.setBytes(&fm, length: 16, index: 5)
        ce.dispatchThreadgroups(sz(mt.cin / 32, tb.length / 12 / 32), threadsPerThreadgroup: sz(1024))
    }
    ce.endEncoding()
    inflightLock.lock(); ticket += 1; let t = ticket; inflight[t] = cb; inflightLock.unlock()
    cb.commit()
    return t
}
/// Pair call, attention layer: q / k / v rows [qrow, qrow + 128) of the ANE's AQKV output (fp16 [14][qTb][1024]) ->
/// the attention core at p0 (KV written as in v8_submit_pair) -> mix_out rotation -> INT8 rows [orow, orow + 128) of
/// o8 (6144 per row, os bytes) + per (token, 1024-block) scales x 128 in to (ts halves per row). Ticket.
@_cdecl("v8_submit_a8core")
public func v8_submit_a8core(_ layer: Int32, _ p0: Int32, _ qkv: Int32, _ qrow: Int32, _ qTb: Int32, _ o8: Int32, _ to: Int32,
                             _ orow: Int32, _ os: Int32, _ ts: Int32) -> Int32 {
    guard let k = layerIndex[Int(layer)], layers[k].m.kind != "gdn", T == 128 else { return -1 }
    let l = layers[k], cb = q.makeCommandBuffer()!, e = Enc(mixEncoder(cb))
    let prm = SIMD4<UInt32>(UInt32(qrow), UInt32(qTb), 0, 0), blk = Int(qTb) * 1024 * 2
    e.run(pso("a8_load_tm"), [BIG, wrapped[Int(qkv)]], [(prm, 2)], grid: sz(12288 / 32, T / 32), tg: sz(256), bar: false)
    e.run(pso("a8_load_tm"), [KP, wrapped[Int(qkv)]], [(prm, 2)], grid: sz(1024 / 32, T / 32), tg: sz(256), offsets: [1: 12 * blk], bar: false)
    e.run(pso("a8_load_tm"), [VP, wrapped[Int(qkv)]], [(prm, 2)], grid: sz(1024 / 32, T / 32), tg: sz(256), offsets: [1: 13 * blk])
    encodeMixer(e, l, p0: Int(p0), core: true)
    rotate(e, l, "mix_out", ON, ON16, 6144)
    e.run(pso("a8_quant_tm"), [ON16, wrapped[Int(o8)], wrapped[Int(to)]], [(SIMD4<UInt32>(UInt32(orow), UInt32(os), UInt32(ts), 1024), 3)],
          grid: sz(6, T), tg: sz(256))
    e.e.endEncoding()
    inflightLock.lock(); ticket += 1; let t = ticket; inflight[t] = cb; inflightLock.unlock()
    cb.commit()
    return t
}
'''
    w.write_text(s); print("swift patched")
