// The GPU side of a decode verify's attention layer, localized test 2 (16 Nov 2026): the engine's attn_prep8q compiled
// with T = 8 (q / k RMS norm, RoPE on the first 64 dims, INT8 Q + scale, INT8 K / V rows + scales appended into the
// ANE cache segment at p0 .. p0 + 7), decode_attn_seg.swift's attention over the segmented cache (causal among the 8
// rows), then a combine that also applies the output gate (o = attn x sigmoid(gate), channel-major [6144][8], the
// input of the output projection). Synthetic projections, norm weights and history; checks: the prep's codes and
// scales against a CPU implementation of the same math, and the gated output against a CPU attention over the
// GPU-written INT8 Q / K / V.
//   swiftc -O -o decode_step_gpu decode_step_gpu.swift -framework Metal && V8_KERNELS=../engine/kernels.metal ./decode_step_gpu
import Foundation
import Metal

let env = ProcessInfo.processInfo.environment
let T = 8
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
let ksrc = try! String(contentsOfFile: env["V8_KERNELS"]!, encoding: .utf8).replacingOccurrences(of: "#define T 64", with: "#define T \(T)")
let klib: MTLLibrary
do { klib = try dev.makeLibrary(source: ksrc, options: opts) } catch { print("kernels.metal with T = 8: \(error)".prefix(2000)); exit(1) }
let pPrep = try! dev.makeComputePipelineState(function: klib.makeFunction(name: "attn_prep8q")!)
// the decode attention + a gated combine (decode_attn_seg.swift's kernel, unchanged)
var dsrc = try! String(contentsOfFile: "decode_attn_seg.swift", encoding: .utf8)
dsrc = String(dsrc[dsrc.range(of: "#include <metal_stdlib>")!.lowerBound..<dsrc.range(of: "// O [4 kv heads][48][256]")!.lowerBound])
dsrc += """
// o[(h * 256 + d) * 8 + t] = attention x sigmoid(gate) for query head h = 6 gh + j / 8, row t = j % 8 (gate from qg)
kernel void dec_combine_gate(device const float *OP [[buffer(0)]], device const float *MP [[buffer(1)]], device const float *LP [[buffer(2)]],
                             device float *o [[buffer(3)]], constant DPrm &p [[buffer(4)]], device const float *qg [[buffer(5)]],
                             uint3 g [[thread_position_in_grid]]) {
    uint d = g.x, j = g.y, gh = g.z, qi = j / M, r = j % M, h = gh * 6 + j / 8, t = j % 8;
    float Mx = -INFINITY;
    for (uint c = 0; c < p.nc; ++c) Mx = max(Mx, MP[((gh * p.nc + c) * QSP + qi) * M + r]);
    float num = 0.0f, den = 0.0f;
    for (uint c = 0; c < p.nc; ++c) {
        uint s = (gh * p.nc + c) * QSP + qi;
        float mc = MP[s * M + r];
        if (mc == -INFINITY) continue;
        float w = exp(mc - Mx);
        num += w * OP[(s * M + r) * 256 + d]; den += w * LP[s * M + r];
    }
    float gt = qg[(h * 512 + 256 + d) * 8 + t];
    o[(h * 256 + d) * 8 + t] = num / den / (1.0f + exp(-gt));
}
"""
let dlib = try! dev.makeLibrary(source: dsrc, options: opts)
let pA = try! dev.makeComputePipelineState(function: dlib.makeFunction(name: "dec_attn")!)
let pCG = try! dev.makeComputePipelineState(function: dlib.makeFunction(name: "dec_combine_gate")!)
func buf(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: max(n, 16), options: .storageModeShared)! }
func F(_ b: MTLBuffer) -> UnsafeMutablePointer<Float> { b.contents().bindMemory(to: Float.self, capacity: b.length / 4) }
func H(_ b: MTLBuffer) -> UnsafeMutablePointer<Float16> { b.contents().bindMemory(to: Float16.self, capacity: b.length / 2) }
func I8(_ b: MTLBuffer) -> UnsafeMutablePointer<Int8> { b.contents().bindMemory(to: Int8.self, capacity: b.length) }
var rng: UInt64 = 0x243F6A8885A308D3
func rf() -> Float { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return Float(rng >> 40) / Float(1 << 24) * 2 - 1 }
struct DPrm { var n, p0, ck, nc: UInt32; var start: (UInt32, UInt32, UInt32, UInt32, UInt32); var rows: (UInt32, UInt32, UInt32, UInt32, UInt32) }

// one attention layer: segments 2048, 2048, 4096; history rows [0, p0) random INT8 + scales; the step writes p0 .. p0 + 7
let segRows = [2048, 2048, 4096], p0 = Int(env["P0"] ?? "5000")!, n = p0 + 8, eps: Float = 1e-6
struct Seg { let rows: Int, k, ks, v, vs: MTLBuffer }
let segs = segRows.map { r -> Seg in
    let s = Seg(rows: r, k: buf(4 * 256 * r), ks: buf(4 * r * 2), v: buf(4 * r * 256), vs: buf(4 * r * 2))
    arc4random_buf(s.k.contents(), s.k.length); arc4random_buf(s.v.contents(), s.v.length)
    for i in 0..<(4 * r) { H(s.ks)[i] = Float16(0.02 + 0.01 * abs(rf())); H(s.vs)[i] = Float16(0.005 + 0.01 * abs(rf())) }
    return s
}
var segStart = [Int](), acc = 0; for r in segRows { segStart.append(acc); acc += r }
let gi = segStart.lastIndex { $0 <= p0 }!, g0 = segs[gi], r0 = p0 - segStart[gi]
precondition(r0 + T <= g0.rows, "the 8 rows cross a segment (not handled by this test)")
// the ANE part A's outputs (channel-major [C][8]): q + gate per head (512 per head), k, v; norm weights; RoPE table
let qg = buf(12288 * T * 4), kp = buf(1024 * T * 4), vp = buf(1024 * T * 4), qn = buf(256 * 4), kn = buf(256 * 4)
for i in 0..<(12288 * T) { F(qg)[i] = rf() * 2 }; for i in 0..<(1024 * T) { F(kp)[i] = rf() * 2; F(vp)[i] = rf() * 3 }
for d in 0..<256 { F(qn)[d] = 0.5 + abs(rf()); F(kn)[d] = 0.5 + abs(rf()) }
let ropeInv: [Double] = (0..<32).map { 1.0 / pow(10_000_000.0, Double(2 * $0) / 64.0) }
let cs = buf((p0 + T) * 64 * 2 * 4)
for tt in 0..<(p0 + T) { for d in 0..<64 { let a = Double(tt) * ropeInv[d % 32]; F(cs)[(tt * 64 + d) * 2] = Float(cos(a)); F(cs)[(tt * 64 + d) * 2 + 1] = Float(sin(a)) } }
let Qh = buf(24 * T * 256 * 2), Q8 = buf(24 * T * 256), QS = buf(24 * T * 4)
let NCMAX = 64, OP = buf(4 * NCMAX * 3 * 16 * 256 * 4), MP = buf(4 * NCMAX * 3 * 16 * 4), LP = buf(4 * NCMAX * 3 * 16 * 4), ON = buf(6144 * T * 4)

// ---- run: prep -> attention -> gated combine (one command buffer) ----
var st = SIMD4<UInt32>(UInt32(256 * g0.rows), UInt32(g0.rows), UInt32(g0.rows * 256), 256), st2 = SIMD2<UInt32>(UInt32(g0.rows), UInt32(r0)), ep = eps
var st5 = [UInt32](repeating: 0, count: 5), rw5 = [UInt32](repeating: 0, count: 5)
for i in 0..<segs.count { st5[i] = UInt32(segStart[i]); rw5[i] = UInt32(segRows[i]) }
let CK = 1024
var prm = DPrm(n: UInt32(n), p0: UInt32(p0), ck: UInt32(CK), nc: UInt32((n + CK - 1) / CK), start: (st5[0], st5[1], st5[2], st5[3], st5[4]), rows: (rw5[0], rw5[1], rw5[2], rw5[3], rw5[4]))
let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
e.setComputePipelineState(pPrep)
for (i, b) in [qg, kp, vp, qn, kn, Qh, g0.k, g0.v, g0.ks, g0.vs].enumerated() { e.setBuffer(b, offset: 0, index: i) }
e.setBytes(&st, length: 16, index: 10); e.setBytes(&st2, length: 8, index: 11); e.setBytes(&ep, length: 4, index: 12)
e.setBuffer(cs, offset: p0 * 64 * 2 * 4, index: 13); e.setBuffer(Q8, offset: 0, index: 14); e.setBuffer(QS, offset: 0, index: 15)
e.dispatchThreadgroups(MTLSize(width: 32, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
e.memoryBarrier(scope: .buffers)
e.setComputePipelineState(pA)
e.setBuffer(Q8, offset: 0, index: 0); e.setBuffer(QS, offset: 0, index: 1)
for i in 0..<5 { let s = segs[min(i, segs.count - 1)]; e.setBuffer(s.k, offset: 0, index: 2 + i); e.setBuffer(s.ks, offset: 0, index: 7 + i); e.setBuffer(s.v, offset: 0, index: 12 + i); e.setBuffer(s.vs, offset: 0, index: 17 + i) }
e.setBuffer(OP, offset: 0, index: 22); e.setBuffer(MP, offset: 0, index: 23); e.setBuffer(LP, offset: 0, index: 24)
e.setBytes(&prm, length: MemoryLayout<DPrm>.size, index: 25)
e.dispatchThreadgroups(MTLSize(width: Int(prm.nc), height: 3, depth: 4), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
e.memoryBarrier(scope: .buffers)
e.setComputePipelineState(pCG)
for (i, b) in [OP, MP, LP, ON].enumerated() { e.setBuffer(b, offset: 0, index: i) }
e.setBytes(&prm, length: MemoryLayout<DPrm>.size, index: 4); e.setBuffer(qg, offset: 0, index: 5)
e.dispatchThreads(MTLSize(width: 256, height: 48, depth: 4), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
precondition(cb.error == nil, "\(cb.error!)")
print(String(format: "GPU step (prep + attention + gated combine), 1 layer, N %d: %.3f ms", n, (cb.gpuEndTime - cb.gpuStartTime) * 1e3))

// ---- check 1: the prep against the same math on the CPU ----
var qd = 0, kd = 0, vd = 0, qsr: Float = 0, ksr: Float = 0
for h in 0..<32 { for t in 0..<T {
    var v = (0..<256).map { d -> Float in h < 24 ? F(qg)[(h * 512 + d) * T + t] : h < 28 ? F(kp)[((h - 24) * 256 + d) * T + t] : F(vp)[((h - 28) * 256 + d) * T + t] }
    if h < 28 {
        let ss = v.reduce(0) { $0 + $1 * $1 }, rs = 1 / sqrt(ss / 256 + eps)
        v = (0..<256).map { v[$0] * rs * (h < 24 ? F(qn)[$0] : F(kn)[$0]) }
        let b = v
        for d in 0..<64 { let c = F(cs)[((p0 + t) * 64 + d) * 2], s = F(cs)[((p0 + t) * 64 + d) * 2 + 1]; v[d] = b[d] * c + (d < 32 ? -b[d + 32] : b[d - 32]) * s }
    }
    if h < 24 {
        let mq = v.map { abs($0) }.max()!, sq = mq > 0 ? mq / 127 : 1
        for d in 0..<256 { let c = Int8(max(-127, min(127, (v[d] / sq).rounded()))); qd = max(qd, abs(Int(c) - Int(I8(Q8)[(h * T + t) * 256 + d]))) }
        qsr = max(qsr, abs(sq / 16 - F(QS)[h * T + t]) / (sq / 16))
    } else {
        let hv = v.map { Float(Float16($0)) }, m = hv.map { abs($0) }.max()!, sc = Float(Float16(max(m / 127, 1e-6)))
        let kh = h < 28 ? h - 24 : h - 28
        for d in 0..<256 {
            let c = Int8(max(-127, min(127, (hv[d] / sc).rounded())))
            if h < 28 { kd = max(kd, abs(Int(c) - Int(I8(g0.k)[kh * 256 * g0.rows + d * g0.rows + r0 + t]))) }
            else { vd = max(vd, abs(Int(c) - Int(I8(g0.v)[(kh * g0.rows + r0 + t) * 256 + d]))) }
        }
        let gs = Float(h < 28 ? H(g0.ks)[kh * g0.rows + r0 + t] : H(g0.vs)[kh * g0.rows + r0 + t])
        ksr = max(ksr, abs(gs - sc) / sc)
    }
} }
let prepOK = qd <= 1 && kd <= 1 && vd <= 1 && qsr < 1e-3 && ksr < 2e-3
print("check prep: max code difference Q \(qd) K \(kd) V \(vd) (<= 1 allowed), scale error Q \(String(format: "%.1e", qsr)) K/V \(String(format: "%.1e", ksr))  \(prepOK ? "OK" : "WRONG")")

// ---- check 2: the gated output against a CPU attention over the GPU-written INT8 Q / K / V ----
func kv(_ key: Int) -> (Seg, Int) { var a = 0; for s in segs { if key < a + s.rows { return (s, key - a) }; a += s.rows }; fatalError() }
var me: Float = 0, mr: Float = 0, bad = 0
for h in 0..<24 { for t in 0..<T {
    let gh = h / 6, last = p0 + t
    var s = [Float](repeating: 0, count: last + 1)
    for k in 0...last {
        let (sg, r) = kv(k); var a: Int32 = 0
        for d in 0..<256 { a += Int32(I8(Q8)[(h * T + t) * 256 + d]) * Int32(I8(sg.k)[gh * 256 * sg.rows + d * sg.rows + r]) }
        s[k] = Float(a) * F(QS)[h * T + t] * Float(H(sg.ks)[gh * sg.rows + r])
    }
    let mx = s.max()!; var den: Float = 0; var o = [Float](repeating: 0, count: 256)
    for k in 0...last { let (sg, r) = kv(k); let w = exp(s[k] - mx); den += w; let wv = w * Float(H(sg.vs)[gh * sg.rows + r])
        for d in 0..<256 { o[d] += wv * Float(I8(sg.v)[(gh * sg.rows + r) * 256 + d]) } }
    for d in 0..<256 {
        let gt = F(qg)[(h * 512 + 256 + d) * T + t], ref = o[d] / den / (1 + exp(-gt)), got = F(ON)[(h * 256 + d) * T + t]
        if !got.isFinite { bad += 1 }
        mr = max(mr, abs(ref)); me = max(me, abs(ref - got))
    }
} }
let outOK = bad == 0 && me / mr < 0.01
print(String(format: "check gated output: max error %.3f%% of %.4f, non-finite %d  %@", 100 * me / mr, mr, bad, outOK ? "OK" : "WRONG"))
exit(prepOK && outOK ? 0 : 1)
