// DeltaNet projection GEMM prototypes (queue 9, 19 Nov 2026, M6): the served GDN projections of layer 0 (qkv 5120 ->
// 10240, z 5120 -> 6144, out 6144 -> 5120; lut_gemm32_i8t_k256 on the build's HWX weights, mapped zero-copy as the engine
// does, plus the LRM low-rank correction: dense_splitk + splitk_reduce for t = B x, dense_gemm for y += A t) against
//   fold    the low-rank A t folded into the LUT GEMM's epilogue (lut_gemm32_i8t_k256lr, kernels_gdnp.metal)
//   t256    256 tokens per weight pass (the served kernels compiled with T = 256; one pass = two 128-token calls)
// GPU ms per 128 tokens (median), outputs checked against served. Inputs random (INT8 activations + per-1024 scales,
// fp16 activations for the low-rank B x), seeded.   gdn_gemm_proto [CALLS]
import Foundation
import Metal

let CALLS = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1])! : 200
let P = "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe", E = P + "/engine"
let man = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: P + "/w_dec8/manifest_best.json"))) as! [String: Any]
let dir = man["dir"] as! String
let L0 = (man["layers"] as! [[String: Any]]).first { ($0["i"] as! Int) == 0 }!
let mats = L0["mats"] as! [String: [String: Any]]
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let hw = (man["hwx"] as! [[String: Any]])[mats["qkv"]!["hwx"] as! Int]
let kern: MTLBuffer = {
    let page = Int(getpagesize()), size = ((hw["size"] as! Int) + page - 1) / page * page
    let fd = open(hw["path"] as! String, O_RDONLY); precondition(fd >= 0, "open hwx")
    let p = mmap(nil, size, PROT_READ, MAP_SHARED, fd, off_t(hw["off"] as! Int)); precondition(p != MAP_FAILED)
    return dev.makeBuffer(bytesNoCopy: p!, length: size, options: .storageModeShared, deallocator: nil)!
}()
func fileBuf(_ f: String) -> MTLBuffer {
    let d = try! Data(contentsOf: URL(fileURLWithPath: f.hasPrefix("/") ? f : dir + "/" + f))
    return d.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: d.count, options: .storageModeShared)! }
}
func buf(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: max(n, 16), options: .storageModeShared)! }
let src0 = try! String(contentsOfFile: E + "/kernels.metal", encoding: .utf8), srcP = try! String(contentsOfFile: E + "/kernels_gdnp.metal", encoding: .utf8)
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
func lib(_ T: Int, proto: Bool) -> MTLLibrary {
    do { return try dev.makeLibrary(source: (src0 + (proto ? srcP : "")).replacingOccurrences(of: "#define T 64", with: "#define T \(T)"), options: opts) }
    catch { print("COMPILE FAILED T \(T): \(error)".prefix(3000)); exit(1) }
}
let lib128 = lib(128, proto: true), lib256 = lib(256, proto: true)
let ONLY = ProcessInfo.processInfo.environment["ONLY"].map { Set($0.split(separator: ",").map(String.init)) }
func pso(_ l: MTLLibrary, _ name: String, bits: Int = 0, g: Int = 0, cin: Int = 0, acc: Bool = false, cd: Int = 2) -> MTLComputePipelineState {
    let fc = MTLFunctionConstantValues()
    var b = UInt32(bits), gg = UInt32(g), c = UInt32(cin), a = acc, cdv = UInt32(cd)
    if name.hasPrefix("lut_gemm") { fc.setConstantValue(&a, type: .bool, index: 3); fc.setConstantValue(&b, type: .uint, index: 0); fc.setConstantValue(&gg, type: .uint, index: 1); fc.setConstantValue(&c, type: .uint, index: 2); fc.setConstantValue(&cdv, type: .uint, index: 4) }
    if name == "dense_gemm" { fc.setConstantValue(&a, type: .bool, index: 3) }
    return try! dev.makeComputePipelineState(function: try! l.makeFunction(name: name, constantValues: fc))
}
struct Mat { let key: String, tab: MTLBuffer, lra: MTLBuffer, lrb: MTLBuffer, bits: Int, G: Int, cin: Int, cout: Int, cd: Int, fmul: Float }
let M: [Mat] = ["qkv", "z", "out"].map { k in
    let m = mats[k]!
    return Mat(key: k, tab: fileBuf(m["tab"] as! String), lra: fileBuf(m["lr_a"] as! String), lrb: fileBuf(m["lr_b"] as! String), bits: m["bits"] as! Int,
               G: m["G"] as! Int, cin: m["cin"] as! Int, cout: m["cout"] as! Int, cd: m["cd"] as! Int, fmul: Float(m["fmul"] as? Double ?? 1))
}
let lrbQKVZ: MTLBuffer = { let b = buf(M[0].lrb.length + M[1].lrb.length); memcpy(b.contents(), M[0].lrb.contents(), M[0].lrb.length); memcpy(b.contents() + M[0].lrb.length, M[1].lrb.contents(), M[1].lrb.length); return b }()
// inputs for T tokens: value(c, t) depends only on (c, t), so the first 128 tokens of the T = 256 inputs equal the T = 128 ones
func hashv(_ c: Int, _ t: Int, _ s: Int) -> Int { var h = UInt64(c &* 2654435761 ^ t &* 40503 ^ s &* 97); h ^= h >> 13; h = h &* 0x5bd1e995; h ^= h >> 15; return Int(h % 1_000_003) }
struct IO { let T: Int, x8: MTLBuffer, xs: MTLBuffer, x16: MTLBuffer, x8o: MTLBuffer, xso: MTLBuffer, x16o: MTLBuffer, y: [MTLBuffer], part: MTLBuffer, tr2: MTLBuffer, tr2h: MTLBuffer, tr: MTLBuffer, trh: MTLBuffer }
func io(_ T: Int) -> IO {
    func mk(_ cin: Int, _ s: Int) -> (MTLBuffer, MTLBuffer, MTLBuffer) {
        let x8 = buf(cin * T), xs = buf(cin / 1024 * T * 4), x16 = buf(cin * T * 2)
        let p8 = x8.contents().bindMemory(to: Int8.self, capacity: cin * T), ps = xs.contents().bindMemory(to: Float.self, capacity: cin / 1024 * T)
        let ph = x16.contents().bindMemory(to: Float16.self, capacity: cin * T)
        for c in 0..<cin { for t in 0..<T { let h = hashv(c, t, s); p8[c * T + t] = Int8(h % 255 - 127); ph[c * T + t] = Float16(Float(h % 2001 - 1000) / 1000) } }
        for g in 0..<(cin / 1024) { for t in 0..<T { ps[g * T + t] = 0.001 + Float(hashv(g, t, s + 7) % 1000) * 1e-6 } }
        return (x8, xs, x16)
    }
    let a = mk(5120, 1), o = mk(6144, 2)
    return IO(T: T, x8: a.0, xs: a.1, x16: a.2, x8o: o.0, xso: o.1, x16o: o.2, y: M.map { buf($0.cout * T * 4) }, part: buf(32 * 128 * T * 4),
              tr2: buf(128 * T * 4), tr2h: buf(128 * T * 2), tr: buf(64 * T * 4), trh: buf(64 * T * 2))
}
let io128 = io(128), io256 = io(256)
func splits(_ Mr: Int, _ K: Int) -> Int { var s = max(1, min(32, 64 / max(1, Mr / 64))); while s > 1 && (K % (s * 64) != 0 || K / s < 128) { s -= 1 }; return s }
struct Enc { let e: MTLComputeCommandEncoder
    func run(_ p: MTLComputePipelineState, _ b: [MTLBuffer?], _ c: [(Any, Int)] = [], tgs: MTLSize? = nil, threads: MTLSize? = nil, tg: Int, offsets: [Int: Int] = [:]) {
        e.setComputePipelineState(p)
        for (i, x) in b.enumerated() { if let x { e.setBuffer(x, offset: offsets[i] ?? 0, index: i) } }
        for (v, i) in c { if var u = v as? UInt32 { e.setBytes(&u, length: 4, index: i) } else if var f = v as? Float { e.setBytes(&f, length: 4, index: i) } }
        if let tgs { e.dispatchThreadgroups(tgs, threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)) }
        else { e.dispatchThreads(threads!, threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)) }
    }
}
func sz(_ a: Int, _ b: Int = 1) -> MTLSize { MTLSize(width: a, height: b, depth: 1) }
// one variant = one command buffer covering T tokens; returns GPU ms
func encode(_ e: Enc, _ variant: String, _ x: IO) {
    let l = x.T == 128 ? lib128 : lib256, T = x.T
    let spk = pso(l, "dense_splitk"), red = pso(l, "splitk_reduce"), dg = pso(l, "dense_gemm", acc: true)
    func splitk(_ w: MTLBuffer, _ x16: MTLBuffer, _ y: MTLBuffer, _ yh: MTLBuffer, Mr: Int, K: Int) {
        let s = splits(Mr, K)
        e.run(spk, [w, x16, x.part], [(UInt32(Mr), 3), (UInt32(K), 4), (UInt32(K / s), 5)], tgs: sz(Mr / 64, s), tg: 128)
        e.run(red, [x.part, y, nil, nil, nil, yh], [(UInt32(Mr * T), 2), (UInt32(s), 3), (UInt32(2), 4)], threads: sz(Mr * T), tg: 256)
    }
    func lut(_ m: Mat, _ name: String, _ y: MTLBuffer, x8: MTLBuffer, xs: MTLBuffer, A: MTLBuffer? = nil, tt: MTLBuffer? = nil, ttOff: Int = 0) {
        let p = pso(l, name, bits: m.bits, g: m.G, cin: m.cin, acc: false, cd: m.cd), nsub = m.tab.length / 12
        e.run(p, [kern, m.tab, x8, y, nil, xs, A, tt], [(m.fmul, 4)], tgs: sz(nsub / (32 / (m.cd * m.G))), tg: name.hasSuffix("ws") ? 256 : 128, offsets: [7: ttOff])
    }
    if variant == "fold" {
        splitk(lrbQKVZ, x.x16, x.tr2, x.tr2h, Mr: 128, K: 5120)
        lut(M[0], "lut_gemm32_i8t_k256lr", x.y[0], x8: x.x8, xs: x.xs, A: M[0].lra, tt: x.tr2h)
        lut(M[1], "lut_gemm32_i8t_k256lr", x.y[1], x8: x.x8, xs: x.xs, A: M[1].lra, tt: x.tr2h, ttOff: 64 * T * 2)
        splitk(M[2].lrb, x.x16o, x.tr, x.trh, Mr: 64, K: 6144)
        lut(M[2], "lut_gemm32_i8t_k256lr", x.y[2], x8: x.x8o, xs: x.xso, A: M[2].lra, tt: x.trh)
        return
    }
    // served order (encodeMixer + lutI8 + lowrank): qkv LUT, B_qkvz x, A_qkv t; z LUT, A_z t; out LUT, B_out x, A_out t
    // bounds (timing only, outputs incomplete): nodg = no dense_gemm (the most a fold can save), nolr = no low-rank at
    // all, k128 / k128nodec = the k128 LUT form and the same without its decode (the most hidden decode can save)
    let ln = variant == "k128" ? "lut_gemm32_i8t_k128" : variant == "k128nodec" ? "lut_gemm32_i8t_k128_nodec" : variant.hasPrefix("ws") ? "lut_gemm32_i8t_k256ws" : "lut_gemm32_i8t_k256"
    let wsT = variant.hasPrefix("ws") ? 256 : 128
    let dgOn = variant != "nodg" && variant != "nolr" && variant != "wsnodg", lrOn = variant != "nolr"
    lut(M[0], ln, x.y[0], x8: x.x8, xs: x.xs)
    if lrOn { splitk(lrbQKVZ, x.x16, x.tr2, x.tr2h, Mr: 128, K: 5120) }
    if dgOn { e.run(dg, [M[0].lra, x.tr2h, x.y[0]], [(UInt32(10240), 3), (UInt32(64), 4)], tgs: sz(10240 / 64), tg: 128) }
    lut(M[1], ln, x.y[1], x8: x.x8, xs: x.xs)
    if dgOn { e.run(dg, [M[1].lra, x.tr2h, x.y[1]], [(UInt32(6144), 3), (UInt32(64), 4)], tgs: sz(6144 / 64), tg: 128, offsets: [1: 64 * T * 2]) }
    if variant == "lut_only" { return }
    lut(M[2], ln, x.y[2], x8: x.x8o, xs: x.xso)
    if lrOn { splitk(M[2].lrb, x.x16o, x.tr, x.trh, Mr: 64, K: 6144) }
    if dgOn { e.run(dg, [M[2].lra, x.trh, x.y[2]], [(UInt32(5120), 3), (UInt32(64), 4)], tgs: sz(5120 / 64), tg: 128) }
}
func call(_ variant: String, _ x: IO) -> Double {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!   // serial encoder: dispatches in order
    encode(Enc(e: e), variant, x); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if let err = cb.error { print("GPU error \(err)"); exit(1) }
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}
func snap(_ x: IO) -> [[Float]] { (0..<3).map { i in let n = M[i].cout * x.T; let p = x.y[i].contents().bindMemory(to: Float.self, capacity: n); return Array(UnsafeBufferPointer(start: p, count: n)) } }
// correctness: fold vs served (T = 128); t256's first 128 tokens vs served
_ = call("served", io128); let ref = snap(io128)
_ = call("fold", io128); let fo = snap(io128)
_ = call("served", io256); let t2 = snap(io256)
_ = call("ws", io128); let wsr = snap(io128)
for i in 0..<3 { var md: Float = 0; for j in 0..<wsr[i].count { md = max(md, abs(wsr[i][j] - ref[i][j])) }; print(String(format: "check %-3@: ws max|diff| vs served %.3e", M[i].key, md)) }
for i in 0..<3 {
    var md: Float = 0, mr: Float = 0, mt: Float = 0, exact = true
    let T2 = 256
    for o in 0..<M[i].cout { for t in 0..<128 {
        let r = ref[i][o * 128 + t]; mr = max(mr, abs(r)); md = max(md, abs(fo[i][o * 128 + t] - r))
        let d2 = abs(t2[i][o * T2 + t] - r); mt = max(mt, d2); if d2 != 0 { exact = false } } }
    print(String(format: "check %-3@: fold max|diff| %.3e of max|y| %.3e | t256 first 128 tokens max|diff| %.3e (bit-exact %@)", M[i].key, md, mr, mt, exact ? "yes" : "no"))
}
let warm = Date(); while Date().timeIntervalSince(warm) < 2 { _ = call("served", io128) }
var res: [[String: Any]] = []
for round in 0..<2 {
    for (v, x, div) in [("served", io128, 1.0), ("ws", io128, 1.0), ("fold", io128, 1.0), ("served", io256, 2.0), ("nodg", io128, 1.0), ("nolr", io128, 1.0),
                        ("k128", io128, 1.0), ("k128nodec", io128, 1.0), ("nodg", io256, 2.0), ("ws", io256, 2.0), ("wsnodg", io128, 1.0), ("wsnodg", io256, 2.0)] where ONLY == nil || ONLY!.contains(v) {
        var ms: [Double] = []; for _ in 0..<CALLS { ms.append(call(v, x)) }
        ms.sort(); let med = ms[ms.count / 2] / div
        let name = x.T == 256 ? (v == "served" ? "t256" : "t256_" + v) : v
        print(String(format: "round %d  %-9@ %.4f ms per 128 tokens (p10 %.4f, p90 %.4f)%@", round, name, med, ms[ms.count / 10] / div, ms[ms.count * 9 / 10] / div,
                     name == "lut_only" ? "  [qkv + z LUT and their low-rank only]" : ""))
        res.append(["round": round, "variant": name, "ms_per_128": med])
    }
}
if let p = ProcessInfo.processInfo.environment["OUT"] { try! JSONSerialization.data(withJSONObject: res, options: .prettyPrinted).write(to: URL(fileURLWithPath: p)) }
