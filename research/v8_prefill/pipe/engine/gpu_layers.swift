// GPU layer engine: a contiguous range of Qwen3.8-27B release layers on the M6 GPU, weights read zero-copy from the
// ANE's compiled programs (model.hwx __KERN, palettized matrices) plus small dense tensors from files (manifest.json,
// written by prep_engine.py). One command buffer per 64-token block; DeltaNet and KV state persist across blocks.
//   gpu_layers <manifest.json> file <in.f16> <out.f16> <nblocks>     blocks of [5120][64] channel-major fp16
//   gpu_layers <manifest.json> serve <slots.f16>                       stdin: "run <slot> <p0>" | "reset" | "quit"
import Foundation
import Metal

struct Hwx: Codable { let path: String; let off: Int; let size: Int }
struct Mat: Codable { let hwx: Int; let tab: String; let bits: Int; let G: Int; let cd: Int; let cin: Int; let cout: Int; let fmul: Float
                      let lr_a: String?; let lr_b: String? }
struct Dense: Codable { let w: String; let cout: Int; let cin: Int; let lr_a: String?; let lr_b: String? }
struct Layer: Codable { let i: Int; let kind: String; let ln1: String; let ln2: String; let rot: [String: String]
                        let mats: [String: Mat]; let dense: [String: Dense]
                        let conv_w: String?; let neg_a: String?; let dt: String?; let normw: String?; let qn: String?; let kn: String? }
struct Manifest: Codable { let dir: String; let Lmax: Int; let eps: Float; let rope_inv: [Float]; let hwx: [Hwx]; let layers: [Layer] }

let args = CommandLine.arguments
let man = try! JSONDecoder().decode(Manifest.self, from: Data(contentsOf: URL(fileURLWithPath: args[1])))
let dev = MTLCreateSystemDefaultDevice()!
let T = Int(ProcessInfo.processInfo.environment["NT"] ?? "64")!, HID = 5120, INTER = 17408
let Lmax = Int(ProcessInfo.processInfo.environment["LMAX"] ?? "\(man.Lmax)")!
let TK = min(Int(ProcessInfo.processInfo.environment["TKV"] ?? "4096")!, Lmax)
func path(_ f: String) -> String { f.hasPrefix("/") ? f : man.dir + "/" + f }
func fileBuf(_ f: String) -> MTLBuffer {
    let d = try! Data(contentsOf: URL(fileURLWithPath: path(f)))
    return d.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: d.count, options: .storageModeShared)! }
}
func buf(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: max(n, 16), options: .storageModeShared)! }
// HWX weight segments, zero-copy
var kern: [MTLBuffer] = []
for h in man.hwx {
    let page = Int(getpagesize()), size = (h.size + page - 1) / page * page
    let fd = open(h.path, O_RDONLY); precondition(fd >= 0, "open \(h.path)")
    let p = mmap(nil, size, PROT_READ, MAP_SHARED, fd, off_t(h.off)); precondition(p != MAP_FAILED)
    kern.append(dev.makeBuffer(bytesNoCopy: p!, length: size, options: .storageModeShared, deallocator: nil)!)
}
// pipelines
let src = try! String(contentsOfFile: (args[0] as NSString).deletingLastPathComponent + "/kernels.metal", encoding: .utf8)
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
let lib = try! dev.makeLibrary(source: src.replacingOccurrences(of: "#define T 64", with: "#define T \(T)"), options: opts)
var psoCache: [String: MTLComputePipelineState] = [:]
var psoName: [ObjectIdentifier: String] = [:]
let PROF = ProcessInfo.processInfo.environment["PROFILE"] != nil
var profT: [String: Double] = [:], profN: [String: Int] = [:]
let q = dev.makeCommandQueue()!
func pso(_ name: String, bits: Int = 0, g: Int = 0, cin: Int = 0, acc: Bool? = nil, cd: Int = 2) -> MTLComputePipelineState {
    let key = "\(name)/\(bits)/\(g)/\(cin)/\(String(describing: acc))/\(cd)"
    if let p = psoCache[key] { return p }
    let fc = MTLFunctionConstantValues()
    var b = UInt32(bits), gg = UInt32(g), c = UInt32(cin), a = acc ?? false, cdv = UInt32(cd)
    if name.hasPrefix("lut_gemm") { fc.setConstantValue(&b, type: .uint, index: 0); fc.setConstantValue(&gg, type: .uint, index: 1); fc.setConstantValue(&c, type: .uint, index: 2); fc.setConstantValue(&cdv, type: .uint, index: 4) }
    if name == "dense_gemm" { fc.setConstantValue(&a, type: .bool, index: 3) }
    let f = try! lib.makeFunction(name: name, constantValues: fc)
    let p = try! dev.makeComputePipelineState(function: f)
    psoCache[key] = p
    psoName[ObjectIdentifier(p)] = name + (name.hasPrefix("lut_gemm") ? "_\(bits)b_cd\(cd)" : "")
    return p
}
// per-layer weights and state
final class L {
    let m: Layer
    var bufs: [String: MTLBuffer] = [:]
    var S: MTLBuffer? = nil, CS: MTLBuffer? = nil, Kc: MTLBuffer? = nil, Vc: MTLBuffer? = nil
    init(_ m: Layer) {
        self.m = m
        for (k, f) in [("ln1", m.ln1), ("ln2", m.ln2)] { bufs[k] = fileBuf(f) }
        for (k, f) in m.rot { bufs["rot_" + k] = fileBuf(f) }
        for (k, mt) in m.mats { bufs["tab_" + k] = fileBuf(mt.tab); if let a = mt.lr_a { bufs["lra_" + k] = fileBuf(a); bufs["lrb_" + k] = fileBuf(mt.lr_b!) } }
        for (k, d) in m.dense { bufs["w_" + k] = fileBuf(d.w); if let a = d.lr_a { bufs["lra_" + k] = fileBuf(a); bufs["lrb_" + k] = fileBuf(d.lr_b!) } }
        for (k, f) in [("conv_w", m.conv_w), ("neg_a", m.neg_a), ("dt", m.dt), ("normw", m.normw), ("qn", m.qn), ("kn", m.kn)] { if let f { bufs[k] = fileBuf(f) } }
        if m.kind == "gdn" { S = buf(48 * 128 * 128 * 4); CS = buf(3 * 10240 * 4) } else { Kc = buf(4 * Lmax * 256 * 2); Vc = buf(4 * Lmax * 256 * 2) }
        reset()
    }
    func reset() { for b in [S, CS, Kc, Vc].compactMap({ $0 }) { memset(b.contents(), 0, b.length) } }
}
let layers = man.layers.map { L($0) }
// scratch (channel-major [C][64])
let X = buf(HID * T * 4), H32 = buf(HID * T * 4), H16 = buf(HID * T * 2), HQ16 = buf(HID * T * 2), Y = buf(HID * T * 4)
let BIG = buf(12288 * T * 4), Z = buf(6144 * T * 4), AB = buf(128 * T * 4), CONV = buf(10240 * T * 4)
let QH = buf(16 * T * 128 * 4), KH = buf(16 * T * 128 * 4), VH = buf(48 * T * 128 * 4), BETA = buf(48 * T * 4), DEC = buf(48 * T * 4)
let O = buf(48 * T * 128 * 4), ON = buf(6144 * T * 4), ON16 = buf(6144 * T * 2), KP = buf(1024 * T * 4), VP = buf(1024 * T * 4)
let QA = buf(24 * T * 256 * 2), SC = buf(24 * T * TK * 4), P16 = buf(24 * T * TK * 2), OA = buf(24 * T * 256 * 4)
let OR = buf(24 * T * 256 * 4), STT = buf(2 * 24 * T * 4), MD = buf(2 * 24 * T * 4)
let OPART = buf(16 * 24 * T * 256 * 4), STPART = buf(16 * 2 * 24 * T * 4)
let FLASH = ProcessInfo.processInfo.environment["FLASH"] == "1"
let PVW = ProcessInfo.processInfo.environment["PVW"] != "0"
let G = buf(INTER * T * 4), U = buf(INTER * T * 4), A16 = buf(INTER * T * 2), TR = buf(64 * T * 4), TR16 = buf(64 * T * 2)
let IN16 = buf(HID * T * 2)
let PART = buf(4 * 1024 * 1024)
let RP = buf(16 * T * 4)
let invBuf = man.rope_inv.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
let ropeTheta = Double(ProcessInfo.processInfo.environment["ROPE_THETA"] ?? "10000000")!
let ropeInv = (0..<32).map { 1.0 / pow(ropeTheta, Double(2 * $0) / 64.0) }  // inv freq in double (the manifest's are float32)
let csBuf = buf(T * 64 * 2 * 4)   // RoPE cos / sin per (token, rotary dim): angle (p0 + t) * inv[d % 32] in double, as the ANE host does
func setRope(_ p0: Int) {
    let p = csBuf.contents().bindMemory(to: Float.self, capacity: T * 128)
    for t in 0..<T { for d in 0..<64 { let a = Double(p0 + t) * ropeInv[d % 32]; p[(t * 64 + d) * 2] = Float(cos(a)); p[(t * 64 + d) * 2 + 1] = Float(sin(a)) } }
}

final class Enc {
    let e: MTLComputeCommandEncoder
    init(_ e: MTLComputeCommandEncoder) { self.e = e }
    func run(_ p: MTLComputePipelineState, _ bs: [MTLBuffer?], _ consts: [(Any, Int)] = [], grid: MTLSize, tg: MTLSize, groups: Bool = true) {
        let cb1 = PROF ? q.makeCommandBuffer()! : nil
        let e = PROF ? cb1!.makeComputeCommandEncoder()! : self.e
        e.setComputePipelineState(p)
        for (i, b) in bs.enumerated() where b != nil { e.setBuffer(b, offset: 0, index: i) }
        for (v, idx) in consts {
            if var u = v as? UInt32 { e.setBytes(&u, length: 4, index: idx) } else if var f = v as? Float { e.setBytes(&f, length: 4, index: idx) }
        }
        if groups { e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg) } else { e.dispatchThreads(grid, threadsPerThreadgroup: tg) }
        if PROF {
            e.endEncoding(); cb1!.commit(); cb1!.waitUntilCompleted()
            let n = psoName[ObjectIdentifier(p)] ?? "?"
            profT[n, default: 0] += cb1!.gpuEndTime - cb1!.gpuStartTime; profN[n, default: 0] += 1
        } else { e.memoryBarrier(scope: .buffers) }
    }
}
func sz(_ x: Int, _ y: Int = 1, _ z: Int = 1) -> MTLSize { MTLSize(width: x, height: y, depth: z) }
func rotate(_ e: Enc, _ l: L, _ key: String, _ x: MTLBuffer, _ out: MTLBuffer, _ n: Int) {
    if let s = l.bufs["rot_" + key] { e.run(pso("fwht4"), [x, s, out], grid: sz(n / 1024, T / 4), tg: sz(512)) }
    else { e.run(pso("to_half"), [x, out], grid: sz(n * T), tg: sz(256), groups: false) }
}
// y (= or +=) W x for small M: split K over threadgroups, then reduce (mode 0: assign, 1: add, 2: assign + half copy)
func denseSplit(_ e: Enc, _ w: MTLBuffer, _ x16: MTLBuffer, _ y: MTLBuffer, M: Int, K: Int, mode: Int, y16: MTLBuffer? = nil) {
    var splits = max(1, min(Int(ProcessInfo.processInfo.environment["SPLITK_MAX"] ?? "32")!, 64 / max(1, M / 64)))
    while splits > 1 && (K % (splits * 64) != 0 || K / splits < 128) { splits -= 1 }
    e.run(pso("dense_splitk"), [w, x16, PART], [(UInt32(M), 3), (UInt32(K), 4), (UInt32(K / splits), 5)], grid: sz(M / 64, splits), tg: sz(128))
    e.run(pso("splitk_reduce"), [PART, y, nil, nil, nil, y16 ?? y], [(UInt32(M * T), 2), (UInt32(splits), 3), (UInt32(mode), 4)],
          grid: sz(M * T), tg: sz(256), groups: false)
}
func lowrank(_ e: Enc, _ l: L, _ key: String, _ x16: MTLBuffer, _ y: MTLBuffer, cout: Int, cin: Int) {
    guard let a = l.bufs["lra_" + key], let b = l.bufs["lrb_" + key] else { return }
    denseSplit(e, b, x16, TR, M: 64, K: cin, mode: 2, y16: TR16)
    e.run(pso("dense_gemm", acc: true), [a, TR16, y], [(UInt32(cout), 3), (UInt32(64), 4)], grid: sz(cout / 64), tg: sz(128))
}
func lut(_ e: Enc, _ l: L, _ key: String, _ x16: MTLBuffer, _ y: MTLBuffer) {
    let m = l.m.mats[key]!
    let nsub = l.bufs["tab_" + key]!.length / 12
    let tm = T >= 128 ? 32 : 64
    e.run(pso(tm == 32 ? "lut_gemm32" : "lut_gemm", bits: m.bits, g: m.G, cin: m.cin, cd: m.cd), [kern[m.hwx], l.bufs["tab_" + key]!, x16, y], [(m.fmul, 4)],
          grid: sz(nsub / (tm / (m.cd * m.G))), tg: sz(128))
    lowrank(e, l, key, x16, y, cout: m.cout, cin: m.cin)
}
func dense(_ e: Enc, _ l: L, _ key: String, _ x16: MTLBuffer, _ y: MTLBuffer) {
    let d = l.m.dense[key]!
    denseSplit(e, l.bufs["w_" + key]!, x16, y, M: d.cout, K: d.cin, mode: 0)
    lowrank(e, l, key, x16, y, cout: d.cout, cin: d.cin)
}
func encodeBlock(_ cb: MTLCommandBuffer, p0: Int) {
    let e = Enc(cb.makeComputeCommandEncoder()!)
    let eps = man.eps
    for l in layers {
        e.run(pso("rms_partial"), [X, RP], grid: sz(HID / 320), tg: sz(256))
        e.run(pso("rms_apply"), [X, l.bufs["ln1"], H32, H16, RP], [(UInt32(HID), 5), (eps, 6)], grid: sz(HID * T), tg: sz(256), groups: false)
        rotate(e, l, "mix_in", H32, HQ16, HID)
        if l.m.kind == "gdn" {
            lut(e, l, "qkv", HQ16, BIG); lut(e, l, "z", HQ16, Z)
            dense(e, l, "ab", H16, AB)
            e.run(pso("gdn_conv"), [BIG, l.CS, l.bufs["conv_w"], CONV], grid: sz(10240), tg: sz(256), groups: false)
            e.run(pso("gdn_qkv"), [CONV, QH, KH, VH], grid: sz(80, T), tg: sz(128))
            e.run(pso("gdn_gates"), [AB, l.bufs["dt"], l.bufs["neg_a"], BETA, DEC], grid: sz(48, T), tg: sz(48, 1), groups: false)
            e.run(pso("gdn_rec"), [QH, KH, VH, DEC, BETA, l.S, O], grid: sz(48, 4), tg: sz(128))
            e.run(pso("gdn_norm"), [O, Z, l.bufs["normw"], ON], [(eps, 4)], grid: sz(48, T), tg: sz(128))
            rotate(e, l, "mix_out", ON, ON16, 6144)
            lut(e, l, "out", ON16, Y)
        } else {
            let L_ = UInt32(p0 + T)
            lut(e, l, "q", HQ16, BIG); dense(e, l, "k", HQ16, KP); dense(e, l, "v", HQ16, VP)
            e.run(pso("attn_prep16"), [BIG, KP, VP, l.bufs["qn"], l.bufs["kn"], QA, l.Kc, l.Vc, nil, nil, nil, csBuf],
                  [(UInt32(p0), 8), (UInt32(Lmax), 9), (eps, 10)], grid: sz(32, T), tg: sz(256))
            let ntile = (p0 + T + 63) / 64
            let ns = max(1, min(16, ntile / 16))   // key splits: at least 16 key tiles each
            if FLASH {
                e.run(pso("attn_flash"), [QA, l.Kc, l.Vc, OPART, STPART], [(UInt32(Lmax), 5), (UInt32(p0), 6), (UInt32(ns), 7)],
                      grid: sz(6 * T / 64, 4, ns), tg: sz(256))
                e.run(pso("attn_combine_splits"), [OPART, STPART, OR, MD], [(UInt32(ns), 4)], grid: sz(256, 24 * T), tg: sz(256, 1, 1), groups: false)
            } else {
                var a0 = 0
                while a0 < Int(L_) {
                    let n = min(TK, Int(L_) - a0)
                    e.run(pso("attn_qk_h"), [QA, l.Kc, P16], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128))
                    e.run(pso("attn_softmax_h"), [P16, STT], [(UInt32(TK), 2), (UInt32(a0), 3), (UInt32(n), 4), (UInt32(p0), 5)], grid: sz(24 * T), tg: sz(256))
                    if PVW { e.run(pso("attn_pv_w"), [P16, l.Vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(2, 6 * T / 64, 4), tg: sz(128)) }
                    else { e.run(pso("attn_pv_t"), [P16, l.Vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(4, 6 * T / 64, 4), tg: sz(128)) }
                    e.run(pso("attn_combine"), [OA, STT, OR, MD], [(UInt32(a0 == 0 ? 1 : 0), 4)], grid: sz(256, 24 * T), tg: sz(256, 1, 1), groups: false)
                    a0 += n
                }
            }
            e.run(pso("attn_finish"), [OR, MD, BIG, ON], grid: sz(256, T, 24), tg: sz(64, 1, 1), groups: false)
            rotate(e, l, "mix_out", ON, ON16, 6144)
            lut(e, l, "o", ON16, Y)
        }
        e.run(pso("rms_partial_add"), [X, RP, Y], grid: sz(HID / 320), tg: sz(256))
        e.run(pso("rms_apply"), [X, l.bufs["ln2"], H32, H16, RP], [(UInt32(HID), 5), (eps, 6)], grid: sz(HID * T), tg: sz(256), groups: false)
        rotate(e, l, "mlp_in", H32, HQ16, HID)
        lut(e, l, "gate", HQ16, G); lut(e, l, "up", HQ16, U)
        if let s = l.bufs["rot_mlp_mid"] { e.run(pso("fwht4_swiglu"), [G, s, A16, U], grid: sz(INTER / 1024, T / 4), tg: sz(512)) }
        else { e.run(pso("swiglu"), [G, U, G], grid: sz(INTER * T), tg: sz(256), groups: false); rotate(e, l, "mlp_mid", G, A16, INTER) }
        lut(e, l, "down", A16, Y)
        e.run(pso("add_into"), [X, Y], grid: sz(HID * T), tg: sz(256), groups: false)
    }
    e.e.endEncoding()
}
func setInput(_ p: UnsafePointer<Float16>) { let x = X.contents().bindMemory(to: Float.self, capacity: HID * T); for i in 0..<(HID * T) { x[i] = Float(p[i]) } }
func getOutput(_ p: UnsafeMutablePointer<Float16>) { let x = X.contents().bindMemory(to: Float.self, capacity: HID * T); for i in 0..<(HID * T) { p[i] = Float16(x[i]) } }
var slotBuf: MTLBuffer? = nil
let nFA = man.layers.filter { $0.kind != "gdn" }.count
let slotHalves = 2 * 5120 * (Int(ProcessInfo.processInfo.environment["NT"] ?? "64")!) + nFA * 2 * 4 * (Int(ProcessInfo.processInfo.environment["NT"] ?? "64")!) * 256  // serve mode: the slots file, zero-copy; conversions run on the GPU in the block's command buffer
func runBlock(p0: Int, slot: Int = -1) -> Double {
    setRope(p0)
    let cb = q.makeCommandBuffer()!
    if slot >= 0 {
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(pso("from_half")); e.setBuffer(slotBuf, offset: slot * slotHalves * 2, index: 0); e.setBuffer(X, offset: 0, index: 1)
        e.dispatchThreads(sz(HID * T), threadsPerThreadgroup: sz(256)); e.endEncoding()
    }
    encodeBlock(cb, p0: p0)
    if slot >= 0 {
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(pso("to_half")); e.setBuffer(X, offset: 0, index: 0); e.setBuffer(slotBuf, offset: (slot * slotHalves + HID * T) * 2, index: 1)
        e.dispatchThreads(sz(HID * T), threadsPerThreadgroup: sz(256)); e.endEncoding()
    }
    cb.commit(); cb.waitUntilCompleted()
    if let err = cb.error { print("command buffer error: \(err)"); exit(1) }
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}
setvbuf(stdout, nil, _IOLBF, 0)
let P0OFF = Int(ProcessInfo.processInfo.environment["P0_OFFSET"] ?? "0")!  // file mode: start position (timing at long context)
if args[2] == "file" {
    let inp = try! Data(contentsOf: URL(fileURLWithPath: args[3])), nb = Int(args[5])!
    var out = Data(count: nb * HID * T * 2)
    var ts: [Double] = []
    for b in 0..<nb {
        inp.withUnsafeBytes { setInput($0.baseAddress!.assumingMemoryBound(to: Float16.self) + b * HID * T) }
        ts.append(runBlock(p0: P0OFF + b * T))
        out.withUnsafeMutableBytes { getOutput($0.baseAddress!.assumingMemoryBound(to: Float16.self) + b * HID * T) }
    }
    try! out.write(to: URL(fileURLWithPath: args[4]))
    let s = ts.sorted()
    print(String(format: "%d layers, %d blocks: GPU ms per block median %.2f (first %.2f, last %.2f)", layers.count, nb, s[s.count / 2], ts[0], ts[nb - 1]))
    if PROF {
        let tot = profT.values.reduce(0, +)
        for (n, t) in profT.sorted(by: { $0.value > $1.value }) {
            print(String(format: "  %-22@ %8.2f ms/block  %5.1f%%  (%d calls/block)", n, t * 1e3 / Double(nb), 100 * t / tot, profN[n]! / nb))
        }
    }
} else {
    // slots file: [slot][in [5120][64] f16 | out [5120][64] f16]
    let fd = open(args[3], O_RDWR); precondition(fd >= 0)
    var st = stat(); fstat(fd, &st)
    let raw = mmap(nil, Int(st.st_size), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)!
    slotBuf = dev.makeBuffer(bytesNoCopy: raw, length: (Int(st.st_size) + Int(getpagesize()) - 1) / Int(getpagesize()) * Int(getpagesize()),
                             options: .storageModeShared, deallocator: nil)!
    print("ready \(layers.count) layers")
    while let line = readLine() {
        let w = line.split(separator: " ")
        if w.first == "quit" { break }
        if w.first == "reset" { for l in layers { l.reset() }; print("ok"); continue }
        if w.first == "state" {  // DeltaNet states in the ANE's layout: conv rows [3][10240] f16, rec [48][dk][dv] f16 x GDN_SV (64)
            var out = Data()
            for l in layers where l.m.kind == "gdn" {
                let cs = l.CS!.contents().bindMemory(to: Float.self, capacity: 3 * 10240)
                var c16 = [Float16](repeating: 0, count: 3 * 10240)
                for i in 0..<(3 * 10240) { c16[i] = Float16(cs[i]) }
                let s = l.S!.contents().bindMemory(to: Float.self, capacity: 48 * 128 * 128)
                var r16 = [Float16](repeating: 0, count: 48 * 128 * 128)
                for h in 0..<48 { for kk in 0..<128 { for v in 0..<128 { r16[(h * 128 + kk) * 128 + v] = Float16(64 * s[(h * 128 + v) * 128 + kk]) } } }
                c16.withUnsafeBytes { out.append(contentsOf: $0) }; r16.withUnsafeBytes { out.append(contentsOf: $0) }
            }
            try! out.write(to: URL(fileURLWithPath: String(w[1]))); print("ok"); continue
        }
        let slot = Int(w[1])!, p0 = Int(w[2])!
        let t0 = Date()
        let g = runBlock(p0: p0, slot: slot)
        // the new K / V rows of this call's positions, per attention layer, after the slot's in / out halves
        let cb2 = q.makeCommandBuffer()!, bl = cb2.makeBlitCommandEncoder()!
        var f = 0
        for l in layers where l.m.kind != "gdn" {
            for (c, src) in [l.Kc!, l.Vc!].enumerated() {
                for h in 0..<4 {
                    let dst = (slot * slotHalves + 2 * HID * T + ((f * 2 + c) * 4 + h) * T * 256) * 2
                    bl.copy(from: src, sourceOffset: ((h * Lmax + p0) * 256) * 2, to: slotBuf!, destinationOffset: dst, size: T * 256 * 2)
                }
            }
            f += 1
        }
        bl.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()
        print(String(format: "%.3f %.3f", g, -t0.timeIntervalSinceNow * 1e3))
    }
}
