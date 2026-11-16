"""A8 step b (16 Nov 2026): tiled expander. a8_expand_t: one threadgroup (1024 threads) per tile of 32 row-consecutive
subgroups (G 16: 1024 output rows) x 32 columns; the tile's indices are loaded coalesced (lanes over columns) into
threadgroup memory, then decoded with lanes over subgroups, so each simdgroup writes 1 KB contiguous. Needs the tables
sorted by row (v8_a8_setup sorts them in place; the LUT GEMM kernels do not depend on table order). Env A8EXP=t selects
it (default: a8_expand). Backups: *.pre_a8b.*"""
from pathlib import Path
E = Path(__file__).resolve().parent
k = E / "kernels.metal"; s = k.read_text()
if "a8_expand_t" not in s:
    s += r'''
// A8 tiled expander (patch_a8b.py): tile = 32 row-consecutive subgroups (G 16) x 32 columns, 1024 threads.
// prm = (cin, G, mode, xoff) as a8_expand; grid (cin / 32, nsub / 32).
kernel void a8_expand_t(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device const char2 *pal [[buffer(2)]],
                        device int8_t *dst [[buffer(3)]], constant uint4 &prm [[buffer(4)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uchar ti[32 * 32 * 12];   // [subgroup][column][12 bytes = 16 indices]
    threadgroup char2 tp[64];
    uint c0 = tg.x * 32, s0 = tg.y * 32;
    if (tid < 64) tp[tid] = pal[tid];
    {   // load: lanes over columns
        uint s = tid / 32, c = tid % 32;
        device const uchar *src = k + tab[3 * (s0 + s) + 1] + 64 + (c0 + c) * 12;
        threadgroup uchar *d = ti + (s * 32 + c) * 12;
        for (uint q = 0; q < 12; q++) d[q] = src[q];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint c = tid / 32, s = tid % 32;      // decode + write: lanes over subgroups
    threadgroup const uchar *b = ti + (s * 32 + c) * 12;
    uint o0 = 2 * tab[3 * (s0 + s) + 2], x = prm.w + c0 + c, cc = c0 + c;
    device int8_t *d = prm.z == 0 ? dst + (o0 >> 10) * (5120u * 1024u) + cc * 1024u + (o0 & 1023u)
                                  : dst + (x >> 10) * (1024u * 5120u) + (x & 1023u) * 5120u + o0;
    for (uint h = 0; h < 2; h++) {
        ulong v = 0;
        for (uint q = 0; q < 6; q++) v |= ulong(b[6 * h + q]) << (8 * q);
        uint4 w = 0;
        for (uint r = 0; r < 8; r++) {
            char2 p = tp[uint(v >> (6 * r)) & 63u];
            w[r / 2] |= (uint(uchar(p.x)) | (uint(uchar(p.y)) << 8)) << (16 * (r % 2));
        }
        *(device uint4 *)(d + 16 * h) = w;
    }
}
'''
    k.write_text(s); print("kernels patched")
w = E / "gpu_v8.swift"; s = w.read_text()
if "A8EXP" not in s:
    a = '''    A8XS = UInt32(xs); A8TS = UInt32(ts); A8YS = UInt32(ys)
    return A8PAL == nil ? -1 : 0'''
    b = '''    A8XS = UInt32(xs); A8TS = UInt32(ts); A8YS = UInt32(ys)
    if A8EXP == "t" {   // tiled expander: tables sorted by row, 32 consecutive G-16 subgroups per 1024-row tile
        for l in layers { for m in ["gate", "up", "down"] { for j in 0..<8 {
            guard let mt = l.m.mats["\\(m)_c\\(j)"], let tb = l.bufs["tab_\\(m)_c\\(j)"] else { continue }
            let n = tb.length / 12, p = tb.contents().bindMemory(to: UInt32.self, capacity: 3 * n)
            var rows = (0..<n).map { (p[3 * $0], p[3 * $0 + 1], p[3 * $0 + 2]) }
            rows.sort { $0.2 < $1.2 }
            for (i, r) in rows.enumerated() { p[3 * i] = r.0; p[3 * i + 1] = r.1; p[3 * i + 2] = r.2 }
            if mt.G != 16 || n % 32 != 0 || mt.cin % 32 != 0 || (0..<n).contains(where: { p[3 * $0 + 2] != p[2] + UInt32(16 * $0) }) { return -3 }
            if m != "down" && (2 * p[2]) % 1024 != 0 { return -4 }
        } } }
    }
    return A8PAL == nil ? -1 : 0'''
    assert s.count(a) == 1; s = s.replace(a, b)
    a = '''var A8PAL: MTLBuffer? = nil, A8XS: UInt32 = 5120, A8TS: UInt32 = 2, A8YS: UInt32 = 5120'''
    assert s.count(a) == 1
    s = s.replace(a, a + '''
let A8EXP = ProcessInfo.processInfo.environment["A8EXP"] ?? ""   // "t": tiled expander (a8_expand_t)''')
    a = '''            ce.setComputePipelineState(pso("a8_expand"))
            ce.setBuffer(kern[mt.hwx], offset: 0, index: 0); ce.setBuffer(tb, offset: 0, index: 1)
            ce.setBuffer(pal, offset: (Int(layer) * 3 + mi) * 128, index: 2); ce.setBuffer(wrapped[Int(dst)], offset: 0, index: 3)
            ce.setBytes(&prm, length: 16, index: 4)
            ce.dispatchThreads(sz(mt.cin, tb.length / 12), threadsPerThreadgroup: sz(64, 4))'''
    b = '''            ce.setComputePipelineState(pso(A8EXP == "t" ? "a8_expand_t" : "a8_expand"))
            ce.setBuffer(kern[mt.hwx], offset: 0, index: 0); ce.setBuffer(tb, offset: 0, index: 1)
            ce.setBuffer(pal, offset: (Int(layer) * 3 + mi) * 128, index: 2); ce.setBuffer(wrapped[Int(dst)], offset: 0, index: 3)
            ce.setBytes(&prm, length: 16, index: 4)
            if A8EXP == "t" { ce.dispatchThreadgroups(sz(mt.cin / 32, tb.length / 12 / 32), threadsPerThreadgroup: sz(1024)) }
            else { ce.dispatchThreads(sz(mt.cin, tb.length / 12), threadsPerThreadgroup: sz(64, 4)) }'''
    assert s.count(a) == 1; s = s.replace(a, b)
    w.write_text(s); print("swift patched")
