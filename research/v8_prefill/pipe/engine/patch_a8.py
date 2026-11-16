"""A8 (16 Nov 2026): GPU side of the ANE's runtime-INT8 FFN (v8_work/w8a8/a8_ffn_real_probe.py program; driver
v8_work/pipe/v8_a8.py). Kernels: a8_expand (LUT -> INT8 codes of one chunk matrix, from the HWX), a8_quant_tm (the FFN
input -> token-major INT8 rows + per (token, 2560-half) scales), a8_add_tm (residual += the ANE's token-major FFN
output). C API: v8_a8_setup (int8 palettes, ANE row strides), v8_a8_expand (one layer's bg / bu / bd, second queue),
v8_submit_a8 (pair call: residual add of the previous FFN, the mixer, the FFN input). Backups: *.pre_a8.*"""
from pathlib import Path
E = Path(__file__).resolve().parent
k = E / "kernels.metal"; s = k.read_text()
if "a8_expand" not in s:
    s += r'''
// ---- A8 FFN (patch_a8.py): the GPU side of the ANE's runtime-INT8 FFN ----
// LUT -> INT8 codes of one chunk matrix (prep_a8.py tables), one thread per (column c, subgroup s): the subgroup's G
// index rows x 2 (cd 2) = 2G consecutive output rows of column c as 2G codes (pal8 lookup). prm = (cin, G, mode, xoff).
// mode 0 (gate / up): dst [17][5120][1024], (o, c) -> (o >> 10) * 5120 * 1024 + c * 1024 + (o & 1023);
// mode 1 (down): dst [17][1024][5120], (o, x = xoff + c) -> (x >> 10) * 1024 * 5120 + (x & 1023) * 5120 + o.
kernel void a8_expand(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device const char2 *pal [[buffer(2)]],
                      device int8_t *dst [[buffer(3)]], constant uint4 &prm [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x, s = gid.y, cin = prm.x, G = prm.y;
    if (c >= cin) return;
    device const uchar *ix = k + tab[3 * s + 1] + 4 * G;   // indices follow the subgroup's 2G fp16 row scales
    uint o0 = 2 * tab[3 * s + 2], x = prm.w + c;
    device int8_t *d = prm.z == 0 ? dst + (o0 >> 10) * (5120u * 1024u) + c * 1024u + (o0 & 1023u)
                                  : dst + (x >> 10) * (1024u * 5120u) + (x & 1023u) * 5120u + o0;
    for (uint g0 = 0; g0 < G; g0 += 8) {   // 8 indices = 6 bytes at byte (c * G + g0) * 6 / 8 (LSB-first)
        uint b = (c * G + g0) * 6 / 8;
        ulong v = 0;
        for (uint q = 0; q < 6; q++) v |= ulong(ix[b + q]) << (8 * q);
        uint4 w = 0;
        for (uint r = 0; r < 8; r++) {
            char2 p = pal[uint(v >> (6 * r)) & 63u];
            uint u = uint(uchar(p.x)) | (uint(uchar(p.y)) << 8);
            w[r / 2] |= u << (16 * (r % 2));
        }
        *(device uint4 *)(d + 2 * g0) = w;
    }
}
// the FFN input of one call (h: fp16 channel-major [5120][T], rms_fwht4's output) -> INT8 rows x8[(r0 + t) * xs + ch]
// per (token, 2560-channel half), absmax scale; tx[(r0 + t) * ts + half] = scale x 128. prm = (r0, xs bytes, ts halves).
// Grid (2, T) threadgroups of 256.
kernel void a8_quant_tm(device const half *h [[buffer(0)]], device int8_t *x8 [[buffer(1)]], device half *tx [[buffer(2)]],
                        constant uint4 &prm [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float mx[8];
    uint hf = g.x, t = g.y;
    float v[10], m = 0;
    for (uint j = 0; j < 10; j++) { v[j] = float(h[(hf * 2560 + j * 256 + tid) * T + t]); m = max(m, abs(v[j])); }
    m = simd_max(m);
    if (sl == 0) mx[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mm = 0;
    for (uint i = 0; i < 8; i++) mm = max(mm, mx[i]);
    float sc = mm > 0 ? mm / 127.0f : 1.0f, inv = 1.0f / sc;
    device int8_t *row = x8 + (prm.x + t) * prm.y + hf * 2560;
    for (uint j = 0; j < 10; j++) row[j * 256 + tid] = int8_t(clamp(rint(v[j] * inv), -127.0f, 127.0f));
    if (tid == 0) tx[(prm.x + t) * prm.z + hf] = half(sc * 128.0f);
}
// x (fp32 channel-major [5120][T]) += y rows r0 .. r0 + T - 1 (token-major fp16, ys halves per row), 32 x 32 tiles.
// prm = (r0, ys). Grid (5120 / 32, T / 32) threadgroups of 256.
kernel void a8_add_tm(device float *x [[buffer(0)]], device const half *y [[buffer(1)]], constant uint4 &prm [[buffer(2)]],
                      uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tile[32][33];
    uint c0 = g.x * 32, t0 = g.y * 32;
    for (uint i = tid; i < 1024; i += 256) { uint tt = i / 32, cc = i % 32; tile[tt][cc] = float(y[(prm.x + t0 + tt) * prm.y + c0 + cc]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 1024; i += 256) { uint cc = i / 32, tt = i % 32; x[(c0 + cc) * T + t0 + tt] += tile[tt][cc]; }
}
'''
    k.write_text(s); print("kernels patched")
w = E / "gpu_v8.swift"; s = w.read_text()
if "v8_submit_a8" not in s:
    s += r'''
// ---- A8 FFN (patch_a8.py) ----
var A8PAL: MTLBuffer? = nil, A8XS: UInt32 = 5120, A8TS: UInt32 = 2, A8YS: UInt32 = 5120
let q2 = dev.makeCommandQueue()!   // weight expansion runs beside the mixer queue
/// pal8: int8 [64][3][64][2] (prep_a8.py pal8.bin); xs / ts / ys: the ANE A8 program's row strides (x bytes, tx and y halves).
@_cdecl("v8_a8_setup")
public func v8_a8_setup(_ p: UnsafeRawPointer, _ n: Int, _ xs: Int32, _ ts: Int32, _ ys: Int32) -> Int32 {
    A8PAL = dev.makeBuffer(bytes: p, length: n, options: .storageModeShared)
    A8XS = UInt32(xs); A8TS = UInt32(ts); A8YS = UInt32(ys)
    return A8PAL == nil ? -1 : 0
}
/// Layer `layer`'s FFN as INT8 codes into the ANE program's weight inputs bg / bu / bd (wrapped handles), on q2. Ticket.
@_cdecl("v8_a8_expand")
public func v8_a8_expand(_ layer: Int32, _ bg: Int32, _ bu: Int32, _ bd: Int32) -> Int32 {
    guard let k = layerIndex[Int(layer)], let pal = A8PAL else { return -1 }
    let l = layers[k]
    let cb = q2.makeCommandBuffer()!, ce = cb.makeComputeCommandEncoder()!
    var n = 0
    for (mi, (m, dst, mode)) in [("gate", bg, 0), ("up", bu, 0), ("down", bd, 1)].enumerated() {
        for j in 0..<8 {
            guard let mt = l.m.mats["\(m)_c\(j)"], let tb = l.bufs["tab_\(m)_c\(j)"] else { continue }
            precondition(mt.cd == 2 && mt.bits == 6 && mt.G % 8 == 0, "a8_expand: 6-bit cd2 tables")
            var prm = SIMD4<UInt32>(UInt32(mt.cin), UInt32(mt.G), UInt32(mode), UInt32(mt.xoff ?? 0))
            ce.setComputePipelineState(pso("a8_expand"))
            ce.setBuffer(kern[mt.hwx], offset: 0, index: 0); ce.setBuffer(tb, offset: 0, index: 1)
            ce.setBuffer(pal, offset: (Int(layer) * 3 + mi) * 128, index: 2); ce.setBuffer(wrapped[Int(dst)], offset: 0, index: 3)
            ce.setBytes(&prm, length: 16, index: 4)
            ce.dispatchThreads(sz(mt.cin, tb.length / 12), threadsPerThreadgroup: sz(64, 4))
            n += 1
        }
    }
    ce.endEncoding()
    if n != 6 { return -2 }
    inflightLock.lock(); ticket += 1; let t = ticket; inflight[t] = cb; inflightLock.unlock()
    cb.commit()
    return t
}
/// Pair call (NT = 128) of the A8 schedule on blocks qA / qB (fp16 [5120][64] each, the residual after the previous
/// mixer): X = q + y rows [yrow, yrow + 128) (the previous layer's ANE FFN output; y < 0: none), then (flags & 1) the
/// layer's mixer, X -> qA / qB, then (flags & 2) the FFN input -> x8 / tx rows [xrow, xrow + 128). Ticket.
@_cdecl("v8_submit_a8")
public func v8_submit_a8(_ layer: Int32, _ qA: Int32, _ qB: Int32, _ p0: Int32, _ y: Int32, _ yrow: Int32, _ x8: Int32,
                         _ tx: Int32, _ xrow: Int32, _ flags: Int32) -> Int32 {
    guard let k = layerIndex[Int(layer)], T == 128 else { return -1 }
    let l = layers[k], cb = q.makeCommandBuffer()!, e = Enc(mixEncoder(cb))
    e.run(pso("from_half_pair"), [wrapped[Int(qA)], wrapped[Int(qB)], X], grid: sz(HID * T), tg: sz(256), groups: false)
    if y >= 0 { e.run(pso("a8_add_tm"), [X, wrapped[Int(y)]], [(SIMD4<UInt32>(UInt32(yrow), A8YS, 0, 0), 2)], grid: sz(HID / 32, T / 32), tg: sz(256)) }
    if flags & 1 != 0 { encodeMixer(e, l, p0: Int(p0), deferAdd: false) }
    e.run(pso("to_half_pair"), [X, wrapped[Int(qA)], wrapped[Int(qB)]], grid: sz(HID * T), tg: sz(256), groups: false)
    if flags & 2 != 0 {
        e.run(pso("rms_partial"), [X, RP], grid: sz(HID / 320), tg: sz(256))
        e.run(pso("rms_fwht4"), [X, l.bufs["ln2"], l.bufs["rot_mlp_in"], HQ16, H16, RP], [(UInt32(HID), 6), (man.eps, 7)],
              grid: sz(HID / 1024, T / 4), tg: sz(512))
        e.run(pso("a8_quant_tm"), [HQ16, wrapped[Int(x8)], wrapped[Int(tx)]], [(SIMD4<UInt32>(UInt32(xrow), A8XS, A8TS, 0), 3)],
              grid: sz(2, T), tg: sz(256))
    }
    e.e.endEncoding()
    inflightLock.lock(); ticket += 1; let t = ticket; inflight[t] = cb; inflightLock.unlock()
    cb.commit()
    return t
}
'''
    w.write_text(s); print("swift patched")
