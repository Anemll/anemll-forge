// Decode attention on the GPU, standalone (16 Nov 2026): the V8 engine's own attention-core kernels (kernels.metal,
// compiled with T rows per call) over synthetic KV of the M6 target's shapes (16 attention layers, 24 query heads on 4
// KV heads, head dim 256, the ANE cache layout: INT8 keys dim-major [kvh][256][S] with a per-key scale, INT8 values
// [kvh][S][256] with a per-row scale, the K8V16 fp16 value copy [kvh][Lmax][256]), for the actual key count N.
// Variants: k8v16 (attn_qk8isk, attn_softmax8n, attn_pv_w over fp16 V: the server's path) and kv8 (attn_qk8is,
// attn_softmax8, attn_pv8 over INT8 V: half the value bytes). The kernels tile 64 rows per KV head, so T must keep 6T a
// multiple of 64 (T = 32, 64, 128); the decode verify's 8 rows (48 per KV head) fill one tile: the T = 32 / 64 / 128
// times are fitted against the tile count and taken at one tile. Also the command-buffer round trip (a handoff).
//   swiftc -O -o decode_attn_bench decode_attn_bench.swift -framework Metal
//   V8_KERNELS=../engine/kernels.metal NT=64 ./decode_attn_bench [N ...]
import Foundation
import Metal

let env = ProcessInfo.processInfo.environment
let T = Int(env["NT"] ?? "64")!, TK = Int(env["TKV"] ?? "2048")!, Lmax = 98304
let NLB = Int(env["NLAYERS"] ?? "4")!            // distinct layer buffers, cycled over the 16 layer calls
let RT = Int(env["ROWTILES"] ?? "0")!            // >0: only this many 64-row tiles per KV head (the decode verify: 1)
func rt() -> Int { RT > 0 ? RT : 6 * T / 64 }
let SKIP = Set((env["SKIP"] ?? "").split(separator: ",").map(String.init))   // timing ablation: kernels not dispatched
let REPS = Int(env["REPS"] ?? "7")!
let NS: [Int] = CommandLine.arguments.count > 1 ? CommandLine.arguments.dropFirst().map { Int($0)! }
    : [8192, 16384, 24576, 32768, 49152, 65536, 81920, 98304]
precondition((6 * T) % 64 == 0, "6T must be a multiple of 64")
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let src = try! String(contentsOfFile: env["V8_KERNELS"]!, encoding: .utf8)
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
let lib = try! dev.makeLibrary(source: src.replacingOccurrences(of: "#define T 64", with: "#define T \(T)"), options: opts)
var cache: [String: MTLComputePipelineState] = [:]
func pso(_ n: String) -> MTLComputePipelineState {
    if let p = cache[n] { return p }
    let p = try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!); cache[n] = p; return p
}
func buf(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: max(n, 16), options: .storageModeShared)! }
func fillI8(_ b: MTLBuffer) { arc4random_buf(b.contents(), b.length) }
func fillH(_ b: MTLBuffer, _ lo: Float, _ hi: Float) {
    let n = b.length / 2, p = b.contents().bindMemory(to: Float16.self, capacity: n)
    var x: UInt32 = 12345
    for i in 0..<n { x = x &* 1664525 &+ 1013904223; p[i] = Float16(lo + (hi - lo) * Float(x >> 8) / 16777216) }
}
func fillF(_ b: MTLBuffer, _ v: Float) {
    let n = b.length / 4, p = b.contents().bindMemory(to: Float.self, capacity: n); for i in 0..<n { p[i] = v }
}
// per-call buffers (the engine's sizes)
let Q8B = buf(24 * T * 256), QSB = buf(24 * T * 4), P16 = buf(24 * T * TK * 2), OA = buf(24 * T * 256 * 4)
let STT = buf(2 * 24 * T * 4), OR = buf(24 * T * 256 * 4), MD = buf(2 * 24 * T * 4), BIG = buf(12288 * T * 4), ON = buf(6144 * T * 4)
fillI8(Q8B); fillF(QSB, 0.02); fillF(BIG, 0.0)
// per-layer KV (one segment of Lmax rows)
struct Lay { let k, ks, v, vs, vc: MTLBuffer }
let lays: [Lay] = (0..<NLB).map { _ in
    let l = Lay(k: buf(4 * 256 * Lmax), ks: buf(4 * Lmax * 2), v: buf(4 * Lmax * 256), vs: buf(4 * Lmax * 2), vc: buf(4 * Lmax * 256 * 2))
    fillI8(l.k); fillI8(l.v); fillH(l.ks, 0.005, 0.02); fillH(l.vs, 0.005, 0.02); fillH(l.vc, -1, 1); return l
}
let kh = UInt32(256 * Lmax), kd = UInt32(Lmax), sh = UInt32(Lmax), vh = UInt32(Lmax * 256), vr = UInt32(256)

func run(_ e: MTLComputeCommandEncoder, _ name: String, _ bs: [MTLBuffer?], _ consts: [(Any, Int)], grid: MTLSize, tg: MTLSize,
         groups: Bool = true, offsets: [Int: Int] = [:]) {
    if SKIP.contains(name) { return }
    e.setComputePipelineState(pso(name))
    for (i, b) in bs.enumerated() where b != nil { e.setBuffer(b, offset: offsets[i] ?? 0, index: i) }
    for (v, idx) in consts {
        if var u = v as? UInt32 { e.setBytes(&u, length: 4, index: idx) } else if var u2 = v as? SIMD2<UInt32> { e.setBytes(&u2, length: 8, index: idx) }
    }
    if groups { e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg) } else { e.dispatchThreads(grid, threadsPerThreadgroup: tg) }
    e.memoryBarrier(scope: .buffers)
}
func sz(_ x: Int, _ y: Int = 1, _ z: Int = 1) -> MTLSize { MTLSize(width: x, height: y, depth: z) }

/// One attention layer's core for T rows at p0 = N - T (keys 0 ..< N), the engine's call sequence.
func layer(_ e: MTLComputeCommandEncoder, _ l: Lay, _ n_: Int, _ kv8: Bool) {
    let p0 = n_ - T, L_ = n_
    var a0 = 0
    while a0 < L_ {
        let n = min(TK, L_ - a0), off = a0
        if kv8 {
            run(e, "attn_qk8is", [Q8B, l.k, P16, nil, nil, QSB], [(SIMD2<UInt32>(kh, kd), 3), (UInt32(TK), 4)], grid: sz(n / 64, rt(), 4), tg: sz(128), offsets: [1: off])
            run(e, "attn_softmax8", [P16, STT, nil, nil, nil, nil, l.ks, l.vs], [(UInt32(TK), 2), (UInt32(a0), 3), (UInt32(n), 4), (UInt32(p0), 5), (sh, 8)],
                grid: sz(24 * T), tg: sz(256), offsets: [6: off * 2, 7: off * 2])
            run(e, "attn_pv8", [P16, l.v, OA], [(SIMD2<UInt32>(vh, vr), 3), (UInt32(TK), 4), (UInt32(n), 5)], grid: sz(2, rt(), 4), tg: sz(128), offsets: [1: off * Int(vr)])
        } else {
            run(e, "attn_qk8isk", [Q8B, l.k, P16, nil, nil, QSB, l.ks], [(SIMD2<UInt32>(kh, kd), 3), (UInt32(TK), 4), (sh, 7)], grid: sz(n / 64, rt(), 4), tg: sz(128), offsets: [1: off, 6: off * 2])
            run(e, "attn_softmax8n", [P16, STT], [(UInt32(TK), 2), (UInt32(a0), 3), (UInt32(n), 4), (UInt32(p0), 5)], grid: sz(24 * T), tg: sz(256))
            run(e, "attn_pv_w", [P16, l.vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(2, rt(), 4), tg: sz(128))
        }
        run(e, "attn_combine", [OA, STT, OR, MD], [(UInt32(a0 == 0 ? 1 : 0), 4)], grid: sz(256, 24 * T), tg: sz(256, 1, 1), groups: false)
        a0 += n
    }
    run(e, "attn_finish", [OR, MD, BIG, ON], [], grid: sz(256, T, 24), tg: sz(64, 1, 1), groups: false)
}
func verifyMs(_ n: Int, _ kv8: Bool) -> Double {   // 16 attention layers in one command buffer: GPU ms (median)
    var ts: [Double] = []
    for _ in 0..<REPS {
        let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        for li in 0..<16 { layer(e, lays[li % NLB], n, kv8) }
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        precondition(cb.error == nil, "\(cb.error!)")
        ts.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
    }
    return ts.sorted()[ts.count / 2]
}
// handoff: an empty command buffer and a one-dispatch command buffer, CPU wall commit -> completed
func rtt(_ dispatch: Bool) -> Double {
    var ts: [Double] = []
    for _ in 0..<60 {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let cb = q.makeCommandBuffer()!
        if dispatch { let e = cb.makeComputeCommandEncoder()!; run(e, "attn_finish", [OR, MD, BIG, ON], [], grid: sz(256, 1, 1), tg: sz(64, 1, 1), groups: false); e.endEncoding() }
        cb.commit(); cb.waitUntilCompleted()
        ts.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
    }
    return ts.sorted()[ts.count / 2]
}
_ = verifyMs(4096, false); _ = verifyMs(4096, true)   // warm up (pipelines, pages)
var out: [[String: Any]] = []
for n in NS {
    let a = verifyMs(n, false), b = verifyMs(n, true)
    out.append(["T": T, "N": n, "k8v16_ms": (a * 100).rounded() / 100, "kv8_ms": (b * 100).rounded() / 100])
}
let r0 = rtt(false), r1 = rtt(true)
let res: [String: Any] = ["T": T, "TK": TK, "row_tiles": rt(), "layers": 16, "rtt_empty_ms": (r0 * 1000).rounded() / 1000, "rtt_one_dispatch_ms": (r1 * 1000).rounded() / 1000, "runs": out]
print(String(data: try! JSONSerialization.data(withJSONObject: res, options: [.sortedKeys]), encoding: .utf8)!)
