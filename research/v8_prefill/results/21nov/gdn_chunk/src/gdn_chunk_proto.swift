// 21 Nov 2026 (M6 ANE bot, item 4): chunked (WY) fp16 gated-delta-rule recurrence vs the served gdn_rec, standalone
// (no forge, no ANE). Kernels: the served engine kernels.metal (T 64 -> 128, as gdn_rec_scale.swift) + gdn_chunk.metal
// (GC = chunk size). One DeltaNet layer call = 128 tokens: q, k [16][128][128], v / o [48][128][128], g / beta [48][128],
// S [48][128][128] fp32, grid gdn_rec (48, 4) x 128 / gdn_chunk 48 x 128.
// Synthetic inputs, model-like: k = normalize(sqrt(rho) m_head + sqrt(1 - rho) n) (rho = key correlation within a head),
// q the same with its own direction then x 128^-1/2; v ~ N(0, 1); per head a decay rate r_h = e^U(ln 0.001, ln 0.3),
// log g = -r_h U(0.5, 1.5) (g 0.64 .. 0.9995); beta = sigmoid(N(0, 1.5)). Seeded.
//   gdn_chunk_proto parity NCALLS RHO [GC]   chain of NCALLS calls from S = 0: per call max |o| errors of gdn_rec (fp32)
//                                             and gdn_chunk vs a CPU double reference (6 heads; all 48 if NCALLS <= 16)
//                                             and chunk vs rec (all heads), final state errors. JSON lines.
//   gdn_chunk_proto time GC                  per-call GPU ms, gdn_rec vs gdn_chunk, chains of 16 / 64 / 256 calls in one
//                                             command buffer (GPU busy), 15 reps after 3 warm-up; median / p90.
//   swiftc -O -o gdn_chunk_proto gdn_chunk_proto.swift -framework Metal
import Foundation
import Metal

let A = CommandLine.arguments
let ENG = "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/engine"
let HERE = "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power"
let mode = A[1]
let GC = Int(mode == "parity" ? (A.count > 4 ? A[4] : "64") : (A.count > 2 ? A[2] : "64"))!
let T = 128, D = 128, HV = 48, HK = 16
let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!
let ksrc = try! String(contentsOfFile: ENG + "/kernels.metal", encoding: .utf8)
let csrc = try! String(contentsOfFile: HERE + "/gdn_chunk.metal", encoding: .utf8)
func sha(_ s: String) -> String {
    let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/bin/shasum"); p.arguments = ["-a", "256"]
    let i = Pipe(), o = Pipe(); p.standardInput = i; p.standardOutput = o; try! p.run()
    i.fileHandleForWriting.write(s.data(using: .utf8)!); i.fileHandleForWriting.closeFile(); p.waitUntilExit()
    return String(String(data: o.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)!.prefix(16))
}
let VAR = ProcessInfo.processInfo.environment["CHUNK_VAR"] ?? ""   // e.g. "S0H,XH,PH" or "NOSOLVE" (diagnostic)
let DEFS = VAR.split(separator: ",").map { v -> String in let kv = v.split(separator: "="); return kv.count == 2 ? "#define \(kv[0]) \(kv[1])\n" : "#define \(v) 1\n" }.joined()
let NPV = VAR.split(separator: ",").compactMap { v -> Int? in let kv = v.split(separator: "="); return kv.count == 2 && kv[0] == "NP" ? Int(kv[1]) : nil }.first ?? 0   // > 0: gdn_chunk2
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
let lib = try! dev.makeLibrary(source: ksrc.replacingOccurrences(of: "#define T 64", with: "#define T 128") + "\n#define GC \(GC)\n" + DEFS + csrc, options: opts)
let pRec = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "gdn_rec")!)
let KNAME = VAR.contains("STACK") ? "gdn_chunk3" : (NPV > 0 ? "gdn_chunk2" : "gdn_chunk")
let pChk = try! dev.makeComputePipelineState(function: lib.makeFunction(name: KNAME)!)
print("BANNER gdn_chunk_proto \(mode) | kernels.metal sha \(sha(ksrc)) (served gdn_rec, T=128) | gdn_chunk.metal sha \(sha(csrc)) GC=\(GC) variant [\(VAR)] | "
      + "q,k [16][128][128] v,o [48][128][128] g,beta [48][128] S [48][128][128] fp32; chunk: fp16 operands (S0 / P / X fp32), fp32 accumulate | "
      + "kernel \(KNAME) | device \(dev.name) | \(ProcessInfo.processInfo.environment["PI_STATE"] ?? "Pi/forge state not given")")

func fbuf(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: n * 4, options: .storageModeShared)! }
func ptr(_ b: MTLBuffer) -> UnsafeMutablePointer<Float> { b.contents().bindMemory(to: Float.self, capacity: b.length / 4) }
var rs: UInt64 = 0x9E3779B97F4A7C15
func rnd() -> Double { rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27; return Double((rs &* 2685821657736338717) >> 11) / 9007199254740992.0 }
func nrm() -> Double { let u = max(rnd(), 1e-300), w = rnd(); return sqrt(-2 * log(u)) * cos(2 * .pi * w) }

struct Call { let q, k, v, g, b: MTLBuffer }
var RHO = 0.3
var headDir: [[Double]] = (0..<(2 * HK)).map { _ in (0..<D).map { _ in nrm() } }
var rate: [Double] = (0..<HV).map { _ in exp(log(0.001) + rnd() * (log(0.3) - log(0.001))) }
func unitv(_ m: [Double]) -> [Double] {
    var x = (0..<D).map { j in sqrt(RHO) * m[j] / sqrt(Double(D)) + sqrt(1 - RHO) * nrm() / sqrt(Double(D)) }
    let n = sqrt(x.reduce(0) { $0 + $1 * $1 }); for j in 0..<D { x[j] /= n }; return x
}
func makeCall() -> Call {
    let c = Call(q: fbuf(HK * T * D), k: fbuf(HK * T * D), v: fbuf(HV * T * D), g: fbuf(HV * T), b: fbuf(HV * T))
    let q = ptr(c.q), k = ptr(c.k), v = ptr(c.v), g = ptr(c.g), b = ptr(c.b)
    for kh in 0..<HK { for t in 0..<T {
        let kv = unitv(headDir[kh]), qv = unitv(headDir[HK + kh])
        for j in 0..<D { k[(kh * T + t) * D + j] = Float(kv[j]); q[(kh * T + t) * D + j] = Float(qv[j] / sqrt(Double(D))) }
    } }
    for i in 0..<(HV * T * D) { v[i] = Float(nrm()) }
    for h in 0..<HV { for t in 0..<T {
        g[h * T + t] = Float(exp(-rate[h] * (0.5 + rnd())))
        b[h * T + t] = Float(1 / (1 + exp(-1.5 * nrm())))
    } }
    return c
}
let shB = dev.makeBuffer(length: HV * max(1, NPV) * 7 * GC * D * 2, options: .storageModePrivate)!
let sfB = dev.makeBuffer(length: HV * max(1, NPV) * (GC * GC + GC * D) * 4, options: .storageModePrivate)!
let s16B = dev.makeBuffer(length: HV * D * D * 2, options: .storageModePrivate)!
func encode(_ cb: MTLCommandBuffer, chunk: Bool, _ c: Call, _ S: MTLBuffer, _ O: MTLBuffer) {
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(chunk ? pChk : pRec)
    for (i, b) in [c.q, c.k, c.v, c.g, c.b, S, O].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    if chunk { e.setBuffer(shB, offset: 0, index: 7); e.setBuffer(sfB, offset: 0, index: 8); e.setBuffer(s16B, offset: 0, index: 9)
        e.dispatchThreadgroups(MTLSize(width: HV, height: max(1, NPV), depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)) }
    else { e.dispatchThreadgroups(MTLSize(width: HV, height: 4, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)) }
    e.endEncoding()
}
func pct(_ v: [Double], _ p: Double) -> Double { let s = v.sorted(); return s[min(s.count - 1, Int(p * Double(s.count)))] }
func jline(_ d: [String: Any]) { print(String(data: try! JSONSerialization.data(withJSONObject: d, options: [.sortedKeys]), encoding: .utf8)!) }

if mode == "parity" {
    let N = Int(A[2])!; RHO = Double(A[3])!
    let refHeads = N <= 16 ? Array(0..<HV) : [0, 9, 18, 27, 36, 45]
    let Sr = fbuf(HV * D * D), Sc = fbuf(HV * D * D), Or = fbuf(HV * T * D), Oc = fbuf(HV * T * D)
    memset(Sr.contents(), 0, Sr.length); memset(Sc.contents(), 0, Sc.length)
    var Sref = [[Double]](repeating: [Double](repeating: 0, count: D * D), count: HV)
    var worst: [String: Double] = [:]
    func upd(_ k: String, _ x: Double) { worst[k] = max(worst[k] ?? 0, x) }
    for n in 0..<N {
        let c = makeCall()
        let cb = queue.makeCommandBuffer()!
        encode(cb, chunk: false, c, Sr, Or); encode(cb, chunk: true, c, Sc, Oc)
        cb.commit(); cb.waitUntilCompleted()
        precondition(cb.status == .completed, "command buffer failed: \(String(describing: cb.error))")
        let q = ptr(c.q), k = ptr(c.k), v = ptr(c.v), g = ptr(c.g), b = ptr(c.b), orr = ptr(Or), oc = ptr(Oc)
        // chunk vs rec, all heads
        var dmax = 0.0, rmax = 0.0, d2 = 0.0, r2 = 0.0
        for i in 0..<(HV * T * D) { let d = Double(oc[i] - orr[i]); dmax = max(dmax, abs(d)); rmax = max(rmax, abs(Double(orr[i]))); d2 += d * d; r2 += Double(orr[i] * orr[i]) }
        var rec: [String: Any] = ["call": n, "chunk_vs_rec_max_abs": dmax, "chunk_vs_rec_max_rel": dmax / rmax, "chunk_vs_rec_rms_rel": sqrt(d2 / r2), "o_max": rmax]
        // CPU double reference on refHeads
        var eR = 0.0, eC = 0.0, mx = 0.0, eR2 = 0.0, eC2 = 0.0, x2 = 0.0
        for h in refHeads {
            let kh = h / 3
            Sref[h].withUnsafeMutableBufferPointer { S in
                for t in 0..<T {
                    let gt = Double(g[h * T + t]), bt = Double(b[h * T + t])
                    for i in 0..<D {
                        var dot = 0.0
                        let row = i * D
                        for j in 0..<D { S[row + j] *= gt; dot += S[row + j] * Double(k[(kh * T + t) * D + j]) }
                        let u = bt * (Double(v[(h * T + t) * D + i]) - dot)
                        var out = 0.0
                        for j in 0..<D { S[row + j] += u * Double(k[(kh * T + t) * D + j]); out += S[row + j] * Double(q[(kh * T + t) * D + j]) }
                        let oi = (h * T + t) * D + i
                        let a = abs(Double(orr[oi]) - out), cc = abs(Double(oc[oi]) - out)
                        eR = max(eR, a); eC = max(eC, cc); mx = max(mx, abs(out)); eR2 += a * a; eC2 += cc * cc; x2 += out * out
                    }
                }
            }
        }
        rec["ref_heads"] = refHeads.count
        rec["rec_vs_ref_max_rel"] = eR / mx; rec["chunk_vs_ref_max_rel"] = eC / mx
        rec["rec_vs_ref_rms_rel"] = sqrt(eR2 / x2); rec["chunk_vs_ref_rms_rel"] = sqrt(eC2 / x2)
        upd("chunk_vs_rec_max_rel", dmax / rmax); upd("chunk_vs_ref_max_rel", eC / mx); upd("rec_vs_ref_max_rel", eR / mx)
        upd("chunk_vs_ref_rms_rel", sqrt(eC2 / x2)); upd("rec_vs_ref_rms_rel", sqrt(eR2 / x2)); upd("chunk_vs_rec_rms_rel", sqrt(d2 / r2))
        if n < 4 || n % 16 == 15 || n == N - 1 { jline(rec) }
    }
    // final state
    let sr = ptr(Sr), sc = ptr(Sc)
    var dmax = 0.0, smax = 0.0, d2 = 0.0, s2 = 0.0, eR = 0.0, eC = 0.0, mx = 0.0
    for i in 0..<(HV * D * D) { let d = Double(sc[i] - sr[i]); dmax = max(dmax, abs(d)); smax = max(smax, abs(Double(sr[i]))); d2 += d * d; s2 += Double(sr[i] * sr[i]) }
    for h in refHeads { for i in 0..<(D * D) { let r = Sref[h][i]; eR = max(eR, abs(Double(sr[h * D * D + i]) - r)); eC = max(eC, abs(Double(sc[h * D * D + i]) - r)); mx = max(mx, abs(r)) } }
    jline(["final_state": true, "calls": N, "tokens": N * T, "rho": RHO, "GC": GC, "state_chunk_vs_rec_max_rel": dmax / smax, "state_chunk_vs_rec_rms_rel": sqrt(d2 / s2),
           "state_rec_vs_ref_max_rel": eR / mx, "state_chunk_vs_ref_max_rel": eC / mx, "state_max": smax, "worst_over_calls": worst])
} else {
    let sets = (0..<4).map { _ in makeCall() }
    let Sr = fbuf(HV * D * D), Sc = fbuf(HV * D * D), O = fbuf(HV * T * D)
    var res: [String: Any] = ["GC": GC]
    for N in [16, 64, 256] {
        for chunk in [false, true] {
            var per: [Double] = []
            for rep in 0..<18 {
                memset((chunk ? Sc : Sr).contents(), 0, Sr.length)
                let cb = queue.makeCommandBuffer()!
                for n in 0..<N { encode(cb, chunk: chunk, sets[n % 4], chunk ? Sc : Sr, O) }
                cb.commit(); cb.waitUntilCompleted()
                precondition(cb.status == .completed)
                if rep >= 3 { per.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3 / Double(N)) }
            }
            let key = "\(chunk ? "chunk" : "rec")_\(N * T / 1024)K"
            res[key] = ["median_ms_per_call": pct(per, 0.5), "p90_ms_per_call": pct(per, 0.9), "min": per.min()!, "reps": per.count]
            print(String(format: "%@ calls %3d (%2dK tokens): %.4f ms per call (p90 %.4f)", chunk ? "gdn_chunk" : "gdn_rec  ", N, N * T / 1024, pct(per, 0.5), pct(per, 0.9)))
        }
    }
    jline(res)
}
