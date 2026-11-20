// V8 GPU side (dylib): per-layer MIXER of Qwen3.8-27B release layers (DeltaNet / attention, residual add) on the
// M6 GPU, for the V8 schedule (the ANE runs every FFN). Derived from gpu_layers.swift (same kernels and encoding).
// Env: V8_MANIFEST, V8_KERNELS (kernels.metal), NT, LMAX, TKV. C API: v8_init, v8_wrap, v8_mix, v8_reset, v8_state, v8_kv.
// gpu_v8_fold.swift (18 Nov 2026) = gpu_v8.swift (libgpu_v8_srv3) + the softmax folded into QK / PV on the KV8 + K8V16
// path: kernels_fold.metal (next to V8_KERNELS) appended to the source; FOLD=64|64s|128 (PV chunk form), FOLD_MIN=keys
// (fold a call only when it attends to at least that many keys); v8_fold(mode, min) changes both at run time.
// gpu_v8_g256.swift (19 Nov 2026, queue 9b, not served; module gpuv8g): the DeltaNet-only engine instance of the 256-row
// prototype (v8_engine_g256.py). V8G_NT (default NT) sets T (256 here), V8G_MANIFEST (default V8_MANIFEST) the layers
// (the GDN layers only); v8_submit_quad runs one GDN layer on four 64-row ANE blocks (256 rows: the projection GEMMs read
// each weight once per 256 tokens), nv = real rows (the rest padding). Otherwise gpu_v8_f8q.swift.
// gpu_v8_f8q.swift (19 Nov 2026) = gpu_v8_f8.swift + kernels_qk.metal: QKE=tg runs Q.K as attn_qk8isk_tg (the same
// INT8 Q.K, the scores stored through threadgroup memory as coalesced half4 rows; bit-identical). QKE unset = served.
// gpu_v8_f8.swift (18 Nov 2026) = gpu_v8_fold.swift + FP8 P.V (kernels_f8.metal): F8=c (V copy = e4m3 INT8 codes,
// value scales folded into P per tile) or F8=v (V copy = e4m3 values x F8VS); the V copy is one byte per value.
// (gpu_layers.swift header follows)
// GPU layer engine: a contiguous range of Qwen3.8-27B release layers on the M6 GPU, weights read zero-copy from the
// ANE's compiled programs (model.hwx __KERN, palettized matrices) plus small dense tensors from files (manifest.json,
// written by prep_engine.py). One command buffer per 64-token block; DeltaNet and KV state persist across blocks.
//   gpu_layers <manifest.json> file <in.f16> <out.f16> <nblocks>     blocks of [5120][64] channel-major fp16
//   gpu_layers <manifest.json> serve <slots.f16>                       stdin: "run <slot> <p0>" | "reset" | "quit"
import Foundation
import Metal
import Accelerate

struct Hwx: Codable { let path: String; let off: Int; let size: Int }
struct Mat: Codable { let hwx: Int; let tab: String; let bits: Int; let G: Int; let cd: Int; let cin: Int; let cout: Int; let fmul: Float
                      let lr_a: String?; let lr_b: String?
                      let xoff: Int?   // prep_ffn.py FFN down chunks: first input channel (x read from there on)
}
struct Dense: Codable { let w: String; let cout: Int; let cin: Int; let lr_a: String?; let lr_b: String?
                        let w8: String?; let sc: String?   // prep_kv8.py: the export's int8 weights + per-row scales (I8KV)
}
struct Layer: Codable { let i: Int; let kind: String; let ln1: String; let ln2: String; let rot: [String: String]
                        let mats: [String: Mat]; let dense: [String: Dense]
                        let conv_w: String?; let neg_a: String?; let dt: String?; let normw: String?; let qn: String?; let kn: String? }
struct Manifest: Codable { let dir: String; let Lmax: Int; let eps: Float; let rope_inv: [Float]; let hwx: [Hwx]; let layers: [Layer] }

let man = try! JSONDecoder().decode(Manifest.self, from: Data(contentsOf: URL(fileURLWithPath: ProcessInfo.processInfo.environment["V8G_MANIFEST"] ?? ProcessInfo.processInfo.environment["V8_MANIFEST"]!)))
let dev = MTLCreateSystemDefaultDevice()!
let T = Int(ProcessInfo.processInfo.environment["V8G_NT"] ?? ProcessInfo.processInfo.environment["NT"] ?? "64")!, HID = 5120, INTER = 17408
let Lmax = Int(ProcessInfo.processInfo.environment["LMAX"] ?? "\(man.Lmax)")!
let TK = min(Int(ProcessInfo.processInfo.environment["TKV"] ?? "4096")!, Lmax)
func path(_ f: String) -> String { f.hasPrefix("/") ? f : man.dir + "/" + f }
func fileBuf(_ f: String) -> MTLBuffer {
    let d = try! Data(contentsOf: URL(fileURLWithPath: path(f)))
    return d.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: d.count, options: .storageModeShared)! }
}
func buf(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: max(n, 16), options: .storageModeShared)! }
// HWX weight segments, zero-copy
let kern: [MTLBuffer] = man.hwx.map { h in
    let page = Int(getpagesize()), size = (h.size + page - 1) / page * page
    let fd = open(h.path, O_RDONLY); precondition(fd >= 0, "open \(h.path)")
    let p = mmap(nil, size, PROT_READ, MAP_SHARED, fd, off_t(h.off)); precondition(p != MAP_FAILED)
    return dev.makeBuffer(bytesNoCopy: p!, length: size, options: .storageModeShared, deallocator: nil)!
}
// pipelines
let kpath = ProcessInfo.processInfo.environment["V8_KERNELS"]!
let src = try! String(contentsOfFile: kpath, encoding: .utf8)
    + (try! String(contentsOfFile: (kpath as NSString).deletingLastPathComponent + "/kernels_fold.metal", encoding: .utf8))
    + (try! String(contentsOfFile: (kpath as NSString).deletingLastPathComponent + "/kernels_f8.metal", encoding: .utf8))
    + (try! String(contentsOfFile: (kpath as NSString).deletingLastPathComponent + "/kernels_qk.metal", encoding: .utf8))
    + (try! String(contentsOfFile: (kpath as NSString).deletingLastPathComponent + "/kernels_g256.metal", encoding: .utf8))
    + (try! String(contentsOfFile: (kpath as NSString).deletingLastPathComponent + "/kernels_gdnp.metal", encoding: .utf8))
let opts: MTLCompileOptions = { let o = MTLCompileOptions(); o.languageVersion = .version4_1; return o }()
let lib = try! dev.makeLibrary(source: src.replacingOccurrences(of: "#define T 64", with: "#define T \(T)"), options: opts)
var psoCache: [String: MTLComputePipelineState] = [:]
var psoName: [ObjectIdentifier: String] = [:]
let PROF = ProcessInfo.processInfo.environment["PROFILE"] != nil
let SKIP = Set((ProcessInfo.processInfo.environment["SKIP"] ?? "").split(separator: ",").map(String.init))
// CONC=1: concurrent compute encoder (a barrier after every dispatch unless a dispatch opts out: independent kernels
// overlap, e.g. the latency-bound DeltaNet recurrence with the z projection)
let CONC = ProcessInfo.processInfo.environment["CONC"] != "0"  // default on since 14 Nov (-2% GPU time, same output); CONC=0 serial
func mixEncoder(_ cb: MTLCommandBuffer) -> MTLComputeCommandEncoder {
    CONC ? cb.makeComputeCommandEncoder(dispatchType: .concurrent)! : cb.makeComputeCommandEncoder()!
}
var profT: [String: Double] = [:], profN: [String: Int] = [:]
let q = dev.makeCommandQueue()!
func pso(_ name: String, bits: Int = 0, g: Int = 0, cin: Int = 0, acc: Bool? = nil, cd: Int = 2) -> MTLComputePipelineState {
    let key = "\(name)/\(bits)/\(g)/\(cin)/\(String(describing: acc))/\(cd)"
    if let p = psoCache[key] { return p }
    let fc = MTLFunctionConstantValues()
    var b = UInt32(bits), gg = UInt32(g), c = UInt32(cin), a = acc ?? false, cdv = UInt32(cd)
    if name.hasPrefix("lut_gemm") { fc.setConstantValue(&a, type: .bool, index: 3); fc.setConstantValue(&b, type: .uint, index: 0); fc.setConstantValue(&gg, type: .uint, index: 1); fc.setConstantValue(&c, type: .uint, index: 2); fc.setConstantValue(&cdv, type: .uint, index: 4) }
    if name == "dense_gemm" { fc.setConstantValue(&a, type: .bool, index: 3) }
    let f = try! lib.makeFunction(name: name, constantValues: fc)
    let p = try! dev.makeComputePipelineState(function: f)
    psoCache[key] = p
    psoName[ObjectIdentifier(p)] = name + (name.hasPrefix("lut_gemm") ? "_\(bits)b_cd\(cd)" : "")
    return p
}
// KV8=1: attention layers read / write the ANE's own INT8 KV cache (segments registered with v8_kv8_seg), no fp16 cache
let KV8 = ProcessInfo.processInfo.environment["KV8"] == "1"
struct Seg { let start: Int, rows: Int, k: MTLBuffer, v: MTLBuffer, ks: MTLBuffer, vs: MTLBuffer, kh: UInt32, kd: UInt32, vh: UInt32, vr: UInt32, sh: UInt32 }
// per-layer weights and state
final class L {
    let m: Layer
    var bufs: [String: MTLBuffer] = [:]
    var S: MTLBuffer? = nil, CS: MTLBuffer? = nil, Kc: MTLBuffer? = nil, Vc: MTLBuffer? = nil
    var K8c: MTLBuffer? = nil, KSc: MTLBuffer? = nil   // QK8: int8 keys + per (kv head, key) scales
    var segs: [Seg] = []   // KV8: the ANE's INT8 KV segments of this attention layer
    init(_ m: Layer) {
        self.m = m
        for (k, f) in [("ln1", m.ln1), ("ln2", m.ln2)] { bufs[k] = fileBuf(f) }
        for (k, f) in m.rot { bufs["rot_" + k] = fileBuf(f) }
        for (k, mt) in m.mats { bufs["tab_" + k] = fileBuf(mt.tab); if let a = mt.lr_a { bufs["lra_" + k] = fileBuf(a); bufs["lrb_" + k] = fileBuf(mt.lr_b!) } }
        for (k, d) in m.dense { if I8KV, let w8 = d.w8, let sc = d.sc { bufs["w8_" + k] = fileBuf(w8); bufs["sc_" + k] = fileBuf(sc) } else { bufs["w_" + k] = fileBuf(d.w) }; if let a = d.lr_a { bufs["lra_" + k] = fileBuf(a); bufs["lrb_" + k] = fileBuf(d.lr_b!) } }
        for (k, f) in [("conv_w", m.conv_w), ("neg_a", m.neg_a), ("dt", m.dt), ("normw", m.normw), ("qn", m.qn), ("kn", m.kn)] { if let f { bufs[k] = fileBuf(f) } }
        if LRM, m.kind == "gdn", let bq = bufs["lrb_qkv"], let bz = bufs["lrb_z"] {   // one [128][cin] low-rank B for qkv + z
            let bb = buf(bq.length + bz.length)
            memcpy(bb.contents(), bq.contents(), bq.length); memcpy(bb.contents() + bq.length, bz.contents(), bz.length)
            bufs["lrb_qkvz"] = bb; bufs["lrb_qkv"] = nil; bufs["lrb_z"] = nil
        }
        if m.kind == "gdn" { S = buf(48 * 128 * 128 * 4); CS = buf(3 * 10240 * 4) } else if KV8 && K8V16 { Vc = buf(4 * Lmax * 256 * (F8.isEmpty ? 2 : 1)) } else if !KV8 { Kc = buf(4 * Lmax * 256 * 2); Vc = buf(4 * Lmax * 256 * 2); if QK8 { K8c = buf(4 * Lmax * 256); KSc = buf(4 * Lmax * 4) } }
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
let OR = buf(24 * T * 256 * 4), STT = buf(2 * 24 * T * 4), MD = buf(2 * 24 * T * 4), MXB = buf(24 * T * (TK / 64) * 4)
let SMFV = ProcessInfo.processInfo.environment["SMF"] ?? "0", SMF = SMFV == "1" || SMFV == "2" || SMFV == "3"  // softmax folded into QK (tile row max) and PV (exp on load)
let OPART = buf(16 * 24 * T * 256 * 4), STPART = buf(16 * 2 * 24 * T * 4)
let FLASH = ProcessInfo.processInfo.environment["FLASH"] == "1"
// FA2=<RM>_<KN> (e.g. 16_32): register-resident flash attention attn_fa2_<RM>_<KN> (4 simdgroups x RM rows per
// threadgroup); FA2NS overrides the key-split count
let FA2 = ProcessInfo.processInfo.environment["FA2"] ?? ""
// QKT=RxK (attn_qk_RxK[s8]), PVT=RxD (attn_pv_RxD[s8]): tile variants of the two-pass attention matmuls
let QKT = ProcessInfo.processInfo.environment["QKT"] ?? "", PVT = ProcessInfo.processInfo.environment["PVT"] ?? ""
func tdims(_ s: String) -> (Int, Int, Int) { let p = s.replacingOccurrences(of: "s8", with: "").split(separator: "x").map { Int($0)! }; return (p[0], p[1], s.hasSuffix("s8") ? 256 : (p[0] * p[1] >= 16384 ? 256 : 128)) }
let LUTV = ProcessInfo.processInfo.environment["LUTV"] ?? ""
let QX = ProcessInfo.processInfo.environment["QX"] == "1"   // two QK passes, P written once (attn_qk_mx / attn_qk_ex)
let SUB = buf(24 * T * (TK / 64) * 4)
let FA2P = FA2.split(separator: "_").map { Int($0) ?? 0 }   // [RM, KN] or s_..: [0, RM, KN, NSG]
let FA2RM = FA2.isEmpty ? 0 : (FA2P[0] == 0 ? FA2P[1] : FA2P[0]), FA2NSG = (FA2P.first == 0 && FA2P.count > 3) ? FA2P[3] : 4
let FA2NS = Int(ProcessInfo.processInfo.environment["FA2NS"] ?? "0")!
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
    func run(_ p: MTLComputePipelineState, _ bs: [MTLBuffer?], _ consts: [(Any, Int)] = [], grid: MTLSize, tg: MTLSize, groups: Bool = true, offsets: [Int: Int] = [:], bar: Bool = true) {
        if !SKIP.isEmpty, SKIP.contains(psoName[ObjectIdentifier(p)] ?? "") { return }  // timing ablation only
        let cb1 = PROF ? q.makeCommandBuffer()! : nil
        let e = PROF ? cb1!.makeComputeCommandEncoder()! : self.e
        e.setComputePipelineState(p)
        for (i, b) in bs.enumerated() where b != nil { e.setBuffer(b, offset: offsets[i] ?? 0, index: i) }
        for (v, idx) in consts {
            if var u = v as? UInt32 { e.setBytes(&u, length: 4, index: idx) } else if var f = v as? Float { e.setBytes(&f, length: 4, index: idx) }
            else if var u2 = v as? SIMD2<UInt32> { e.setBytes(&u2, length: 8, index: idx) } else if var u4 = v as? SIMD4<UInt32> { e.setBytes(&u4, length: 16, index: idx) }
        }
        if groups { e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg) } else { e.dispatchThreads(grid, threadsPerThreadgroup: tg) }
        if PROF {
            e.endEncoding(); cb1!.commit(); cb1!.waitUntilCompleted()
            let n = psoName[ObjectIdentifier(p)] ?? "?"
            profT[n, default: 0] += cb1!.gpuEndTime - cb1!.gpuStartTime; profN[n, default: 0] += 1
        } else if bar || !CONC { e.memoryBarrier(scope: .buffers) }
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
let EPI = ProcessInfo.processInfo.environment["EPI"] == "1"  // output projection accumulates into X (no add_into)
let REC16 = ProcessInfo.processInfo.environment["REC16"] == "1"  // gdn_rec16: 16 steps staged per barrier pair
let REC8 = ProcessInfo.processInfo.environment["REC8"] == "1"  // gdn_rec8: 8 lanes per state row
let RECV = ProcessInfo.processInfo.environment["REC"] ?? ""  // REC=pf: gdn_rec_pf (all inputs one step ahead, bit-same); px: 4 partial sums
let LUTP = ProcessInfo.processInfo.environment["LUTP"] == "1"  // scalar 4-bit LUT layers (cd 1): lut_gemm32p pair-table decode
// FUSE=rms,conv,norm,add,enc | all: fused glue kernels (rms_fwht4, gdn_conv_qkv, gdn_norm_fwht4, to_half(_pair)_add) and
// the fp16 <-> fp32 conversions inside the mixer encoder (one encoder per call instead of three)
let FUSEV = Set((ProcessInfo.processInfo.environment["FUSE"] ?? "").split(separator: ",").map(String.init))
func FU(_ s: String) -> Bool { FUSEV.contains(s) || FUSEV.contains("all") }
// I8=1: DeltaNet projections (qkv, z, out) as W8A8 LUT GEMMs (lut_gemm32_i8; activations quantized per 1024-channel
// block and token by quant8_blk); low-rank corrections stay fp16. 128-row calls only.
let I8 = ProcessInfo.processInfo.environment["I8"] == "1"
let I8T = ProcessInfo.processInfo.environment["I8T"] == "1"   // with I8: lut_gemm32_i8t (channel-major tile, vector stores)
let I8TV = ProcessInfo.processInfo.environment["V8G_I8TV"] ?? ProcessInfo.processInfo.environment["I8TV"] ?? ""   // queue 9b: V8G_I8TV for this instance only (e.g. _k256ws)
let I8A = ProcessInfo.processInfo.environment["I8A"] == "1"   // with I8: attention layers' q and o projections too (full mixer path)
// I8KV=1 (with I8 + I8A): attention k / v as int8 x int8 from the export's int8 weights (manifest from prep_kv8.py; the
// fp16 copies are then not loaded). LRM=1: DeltaNet low-rank B of qkv and z (same input) as one 128-row multiply.
let I8KV = ProcessInfo.processInfo.environment["I8KV"] == "1"
let LRM = ProcessInfo.processInfo.environment["LRM"] == "1"
let QK8 = ProcessInfo.processInfo.environment["QK8"] == "1"
let KV8D = ProcessInfo.processInfo.environment["KV8D"] == "1"
let K8V16 = KV8 && QK8 && ProcessInfo.processInfo.environment["K8V16"] == "1"   // INT8 K (ANE) + fp16 V copy (patch_k8v16.py)
let F8: String = ProcessInfo.processInfo.environment["F8"] ?? ""   // "" (fp16 P.V) | "c" | "v"
let F8K: String = { let f = ProcessInfo.processInfo.environment["F8"] ?? ""; return (f == "w" || f == "x") ? "v" : f }()   // w = v + vectorized softmax
let QKE: String = ProcessInfo.processInfo.environment["QKE"] ?? ""   // "" served Q.K | "tg" staged epilogue (kernels_qk.metal)
let QKN: String = QKE == "tg" ? "attn_qk8isk_tg" : "attn_qk8isk"
let F8VS: Float = Float(ProcessInfo.processInfo.environment["F8VS"] ?? "1") ?? 1
let P8B = buf(24 * T * TK), CBB = buf(64)
var FOLDV: String = ProcessInfo.processInfo.environment["FOLD"] ?? ""
var FOLD_MIN: Int = Int(ProcessInfo.processInfo.environment["FOLD_MIN"] ?? "0") ?? 0
let PVWW = ProcessInfo.processInfo.environment["PVWW"] == "1"   // PV over 256 dims per threadgroup (patch_pvw.py)   // KV8 core: key scales in QK, V dequantized on chip (patch_kv8b.py)   // int8 x int8 attention scores (attn_prep16q8 + attn_qk8i)
let Q8B = buf(24 * T * 256), QSB = buf(24 * T * 4)
let PART8 = buf(5 * 1024 * T * 4), TR2 = buf(128 * T * 4), TR2_16 = buf(128 * T * 2)
func denseI8(_ e: Enc, _ l: L, _ key: String, _ y: MTLBuffer) {
    let d = l.m.dense[key]!
    e.run(pso("dense_i8_splitk"), [l.bufs["w8_" + key]!, X8A, PART8, l.bufs["sc_" + key]!, XSA], [(UInt32(d.cout), 5), (UInt32(d.cin), 6)],
          grid: sz(d.cout / 32, d.cin / 1024), tg: sz(128))
    e.run(pso("splitk_reduce"), [PART8, y, nil, nil, nil, y], [(UInt32(d.cout * T), 2), (UInt32(d.cin / 1024), 3), (UInt32(0), 4)],
          grid: sz(d.cout * T), tg: sz(256), groups: false)
    lowrank(e, l, key, HQ16, y, cout: d.cout, cin: d.cin)
}
let LUTC = ProcessInfo.processInfo.environment["LUTC"] == "1"  // fp16 LUT GEMM (TM 32): lut_gemm32c (K column fastest in decode)
let X8A = buf(HID * T), XSA = buf(HID / 1024 * T * 4), X8B = buf(6144 * T), XSB = buf(6 * T * 4)
func lutI8(_ e: Enc, _ l: L, _ key: String, _ x8: MTLBuffer, _ xs: MTLBuffer, _ x16: MTLBuffer, _ y: MTLBuffer, acc: Bool = false) {
    precondition(T >= 128, "I8: 128-row calls only")
    let m = l.m.mats[key]!, nsub = l.bufs["tab_" + key]!.length / 12
    e.run(pso(I8T ? "lut_gemm32_i8t" + I8TV : "lut_gemm32_i8", bits: m.bits, g: m.G, cin: m.cin, acc: acc, cd: m.cd), [kern[m.hwx], l.bufs["tab_" + key]!, x8, y, nil, xs], [(m.fmul, 4)],
          grid: sz(nsub / ((I8T && I8TV.contains("m64") ? 64 : 32) / (m.cd * m.G))), tg: sz(I8TV.hasSuffix("ws") ? 256 : 128))
    lowrank(e, l, key, x16, y, cout: m.cout, cin: m.cin)
}
// I8F=1: the GPU FFN (chunked shared build, prep_ffn.py) as W8A8 LUT GEMMs (lut_gemm32_i8t + I8TV)
let I8F = ProcessInfo.processInfo.environment["I8F"] == "1"
let X8C = buf(INTER * T), XSC = buf(INTER / 1024 * T * 4)
/// One FFN chunk matrix (prep_ffn.py): rows are absolute (tab row0), x is read from channel m.xoff on.
func lutChunk(_ e: Enc, _ l: L, _ key: String, _ x16: MTLBuffer, _ x8: MTLBuffer?, _ xs: MTLBuffer?, _ y: MTLBuffer, acc: Bool) {
    precondition(T >= 128, "chunked GPU FFN: 128-row calls only")
    let m = l.m.mats[key]!, nsub = l.bufs["tab_" + key]!.length / 12, xo = m.xoff ?? 0
    if let x8, let xs {
        e.run(pso("lut_gemm32_i8t" + I8TV, bits: m.bits, g: m.G, cin: m.cin, acc: acc, cd: m.cd), [kern[m.hwx], l.bufs["tab_" + key]!, x8, y, nil, xs],
              [(m.fmul, 4)], grid: sz(nsub / (32 / (m.cd * m.G))), tg: sz(128), offsets: [2: xo * T, 5: xo / 1024 * T * 4])
    } else {
        e.run(pso(LUTC ? "lut_gemm32c" : "lut_gemm32", bits: m.bits, g: m.G, cin: m.cin, acc: acc, cd: m.cd), [kern[m.hwx], l.bufs["tab_" + key]!, x16, y],
              [(m.fmul, 4)], grid: sz(nsub / (32 / (m.cd * m.G))), tg: sz(128), offsets: [2: xo * T * 2])
    }
}
// LRF=1: the low-rank correction folded into the LUT GEMM (every matrix); LRF=att: only attention layers' (q, o)
let LRFV = ProcessInfo.processInfo.environment["LRF"] ?? "", LRF = LRFV == "1" || LRFV == "att"
func lut(_ e: Enc, _ l: L, _ key: String, _ x16: MTLBuffer, _ y: MTLBuffer, acc: Bool = false) {
    let m = l.m.mats[key]!
    let nsub = l.bufs["tab_" + key]!.length / 12
    let tm = T >= 128 ? 32 : 64
    if LRF, LRFV == "1" || l.m.kind != "gdn", let a = l.bufs["lra_" + key], let b = l.bufs["lrb_" + key] {
        denseSplit(e, b, x16, TR, M: 64, K: m.cin, mode: 2, y16: TR16)
        e.run(pso(tm == 32 ? "lut_gemm32_lr" : "lut_gemm_lr", bits: m.bits, g: m.G, cin: m.cin, acc: acc, cd: m.cd),
              [kern[m.hwx], l.bufs["tab_" + key]!, x16, y, nil, a, TR16], [(m.fmul, 4)], grid: sz(nsub / (tm / (m.cd * m.G))), tg: sz(128))
        return
    }
    if !LUTV.isEmpty {   // LUTV=32_k128 | 32_db | 32_k128db | 64_db: LUT GEMM K-step / double-buffer variants
        let tv = LUTV.hasPrefix("64") ? 64 : 32
        e.run(pso("lut_gemm" + LUTV, bits: m.bits, g: m.G, cin: m.cin, acc: acc, cd: m.cd), [kern[m.hwx], l.bufs["tab_" + key]!, x16, y], [(m.fmul, 4)],
              grid: sz(nsub / (tv / (m.cd * m.G))), tg: sz(128))
        lowrank(e, l, key, x16, y, cout: m.cout, cin: m.cin)
        return
    }
    let pair = LUTP && tm == 32 && m.cd == 1 && m.bits == 4 && tm / m.G <= 2
    e.run(pso(pair ? "lut_gemm32p" : tm == 32 ? (LUTC ? "lut_gemm32c" : "lut_gemm32") : "lut_gemm", bits: m.bits, g: m.G, cin: m.cin, acc: acc, cd: m.cd), [kern[m.hwx], l.bufs["tab_" + key]!, x16, y], [(m.fmul, 4)],
          grid: sz(nsub / (tm / (m.cd * m.G))), tg: sz(128))
    lowrank(e, l, key, x16, y, cout: m.cout, cin: m.cin)
}
func dense(_ e: Enc, _ l: L, _ key: String, _ x16: MTLBuffer, _ y: MTLBuffer) {
    let d = l.m.dense[key]!
    denseSplit(e, l.bufs["w_" + key]!, x16, y, M: d.cout, K: d.cin, mode: 0)
    lowrank(e, l, key, x16, y, cout: d.cout, cin: d.cin)
}
/// core: attention layers only, the attention core alone (q / k / v projections already in BIG / KP / VP, from the
/// ANE's ffnq2 entry): stops at ON = the gated attention output before the mix_out rotation (the ANE's offn2 input).
func encodeMixer(_ e: Enc, _ l: L, p0: Int, core: Bool = false, deferAdd: Bool = false) {
    let eps = man.eps
    if !core {
    e.run(pso(T == 256 ? "rms_partial_g256" : "rms_partial"), [X, RP], grid: sz(HID / 320), tg: sz(256))
    if FU("rms"), let s = l.bufs["rot_mix_in"] {
        e.run(pso("rms_fwht4"), [X, l.bufs["ln1"], s, HQ16, H16, RP], [(UInt32(HID), 6), (eps, 7)], grid: sz(HID / 1024, T / 4), tg: sz(512))
    } else {
    e.run(pso("rms_apply"), [X, l.bufs["ln1"], H32, H16, RP], [(UInt32(HID), 5), (eps, 6)], grid: sz(HID * T), tg: sz(256), groups: false)
    rotate(e, l, "mix_in", H32, HQ16, HID)
    }
    }
    if l.m.kind == "gdn" {
        if I8 { e.run(pso("quant8_blk"), [HQ16, X8A, XSA], grid: sz(HID / 1024, T / 32), tg: sz(256)); lutI8(e, l, "qkv", X8A, XSA, HQ16, BIG) }
        else { lut(e, l, "qkv", HQ16, BIG) }
        if let bb = l.bufs["lrb_qkvz"] {   // LRM: B of qkv and z together (same input); A of qkv now, A of z after its GEMM
            denseSplit(e, bb, HQ16, TR2, M: 128, K: HID, mode: 2, y16: TR2_16)
            e.run(pso("dense_gemm", acc: true), [l.bufs["lra_qkv"]!, TR2_16, BIG], [(UInt32(10240), 3), (UInt32(64), 4)], grid: sz(10240 / 64), tg: sz(128))
        }
        func lrmZ() { if l.bufs["lrb_qkvz"] != nil { e.run(pso("dense_gemm", acc: true), [l.bufs["lra_z"]!, TR2_16, Z], [(UInt32(6144), 3), (UInt32(64), 4)], grid: sz(6144 / 64), tg: sz(128), offsets: [1: 64 * T * 2]) } }
        if !CONC { if I8 { lutI8(e, l, "z", X8A, XSA, HQ16, Z) } else { lut(e, l, "z", HQ16, Z) }; lrmZ() }
        dense(e, l, "ab", H16, AB)
        if FU("conv") {   // no barrier: gdn_gates is independent of it (the one after gdn_gates covers both)
            if A8NV != UInt32(T) { e.run(pso("gdn_conv_qkv_nv"), [BIG, l.CS, l.bufs["conv_w"], QH, KH, VH], [(A8NV, 6)], grid: sz(80, T / 16), tg: sz(128), bar: false) }
            else { e.run(pso("gdn_conv_qkv"), [BIG, l.CS, l.bufs["conv_w"], QH, KH, VH], grid: sz(80, T / 16), tg: sz(128), bar: false) }
        } else {
        e.run(pso("gdn_conv"), [BIG, l.CS, l.bufs["conv_w"], CONV], grid: sz(10240), tg: sz(256), groups: false)
        e.run(pso("gdn_qkv"), [CONV, QH, KH, VH], grid: sz(80, T), tg: sz(128))
        }
        precondition(A8NV == UInt32(T) || FU("conv"), "valid rows need the fused conv")
        if A8NV != UInt32(T) { e.run(pso("gdn_gates_nv"), [AB, l.bufs["dt"], l.bufs["neg_a"], BETA, DEC], [(A8NV, 5)], grid: sz(48, T), tg: sz(48, 1), groups: false) }
        else { e.run(pso("gdn_gates"), [AB, l.bufs["dt"], l.bufs["neg_a"], BETA, DEC], grid: sz(48, T), tg: sz(48, 1), groups: false) }
        if REC16 { e.run(pso("gdn_rec16"), [QH, KH, VH, DEC, BETA, l.S, O], grid: sz(48, 4), tg: sz(128), bar: false) }
        else if REC8 { e.run(pso("gdn_rec8"), [QH, KH, VH, DEC, BETA, l.S, O], grid: sz(48, 8), tg: sz(128), bar: false) }
        else if !RECV.isEmpty { e.run(pso("gdn_rec_" + RECV), [QH, KH, VH, DEC, BETA, l.S, O], grid: sz(48, 4), tg: sz(128), bar: false) }
        else { e.run(pso("gdn_rec"), [QH, KH, VH, DEC, BETA, l.S, O], grid: sz(48, 4), tg: sz(128), bar: false) }
        if CONC { if I8 { lutI8(e, l, "z", X8A, XSA, HQ16, Z) } else { lut(e, l, "z", HQ16, Z) }; lrmZ() }  // overlaps the recurrence (independent: HQ16 -> Z)
        if FU("norm"), let s = l.bufs["rot_mix_out"] {
            e.run(pso("gdn_norm_fwht4"), [O, Z, l.bufs["normw"], s, ON16], [(eps, 5)], grid: sz(6144 / 1024, T / 4), tg: sz(512))
        } else {
        e.run(pso("gdn_norm"), [O, Z, l.bufs["normw"], ON], [(eps, 4)], grid: sz(48, T), tg: sz(128))
        rotate(e, l, "mix_out", ON, ON16, 6144)
        }
        if A8NOOUT { return }   // A8: the output projection runs on the ANE (ON16 -> AO)
        if I8 { e.run(pso("quant8_blk"), [ON16, X8B, XSB], grid: sz(6144 / 1024, T / 32), tg: sz(256)); lutI8(e, l, "out", X8B, XSB, ON16, EPI ? X : Y, acc: EPI) }
        else { lut(e, l, "out", ON16, EPI ? X : Y, acc: EPI) }
    } else {
        let L_ = UInt32(p0 + T)
        if !core {
            if I8 && I8A { e.run(pso("quant8_blk"), [HQ16, X8A, XSA], grid: sz(HID / 1024, T / 32), tg: sz(256)); lutI8(e, l, "q", X8A, XSA, HQ16, BIG) }
            else { lut(e, l, "q", HQ16, BIG) }
            if I8 && I8A && I8KV && l.bufs["w8_k"] != nil { denseI8(e, l, "k", KP); denseI8(e, l, "v", VP) }
            else { dense(e, l, "k", HQ16, KP); dense(e, l, "v", HQ16, VP) }
        }
        // core + KV8 (A8 --kv8): the core writes / reads the ANE's INT8 segments like the full mixer
        if KV8 {
            let g0 = l.segs.first { p0 >= $0.start && p0 < $0.start + $0.rows }!
            precondition(p0 - g0.start + T <= g0.rows, "a call's rows cross a KV segment")
            e.run(pso(K8V16 ? (F8.isEmpty ? "attn_prep8qv" : "attn_prep8qv_f8" + F8K) : QK8 ? "attn_prep8q" : "attn_prep8"), [BIG, KP, VP, l.bufs["qn"], l.bufs["kn"], QA, g0.k, g0.v, g0.ks, g0.vs, nil, nil, nil, csAll, Q8B, QSB, l.Vc],
                  [(SIMD4<UInt32>(g0.kh, g0.kd, g0.vh, g0.vr), 10), (SIMD2<UInt32>(g0.sh, UInt32(p0 - g0.start)), 11), (eps, 12),
                   (SIMD2<UInt32>(UInt32(Lmax), UInt32(p0)), 17), (F8VS, 18)],
                  grid: sz(32, T), tg: sz(256), offsets: [13: p0 * 64 * 2 * 4])
            var a0 = 0
            while a0 < Int(L_) {
                let n = min(TK, Int(L_) - a0)
                let g = l.segs.first { a0 >= $0.start && a0 < $0.start + $0.rows }!
                let off = a0 - g.start
                precondition(off + n <= g.rows, "a key tile crosses a KV segment")
                let fold = K8V16 && F8.isEmpty && !FOLDV.isEmpty && Int(L_) >= FOLD_MIN
                if fold { e.run(pso("attn_qk8iskm"), [Q8B, g.k, P16, nil, nil, QSB, g.ks, nil, MXB], [(SIMD2<UInt32>(g.kh, g.kd), 3), (UInt32(TK), 4), (g.sh, 7), (SIMD2<UInt32>(UInt32(a0), UInt32(p0)), 9)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128), offsets: [1: off, 6: off * 2]) }
                else if QK8 && (KV8D || K8V16) { e.run(pso(QKN), [Q8B, g.k, P16, nil, nil, QSB, g.ks], [(SIMD2<UInt32>(g.kh, g.kd), 3), (UInt32(TK), 4), (g.sh, 7)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128), offsets: [1: off, 6: off * 2]) }
                else if QK8 { e.run(pso("attn_qk8is"), [Q8B, g.k, P16, nil, nil, QSB], [(SIMD2<UInt32>(g.kh, g.kd), 3), (UInt32(TK), 4)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128), offsets: [1: off]) }
                else { e.run(pso("attn_qk8"), [QA, g.k, P16], [(SIMD2<UInt32>(g.kh, g.kd), 3), (UInt32(TK), 4)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128), offsets: [1: off]) }
                if K8V16 && !F8.isEmpty {   // FP8 probabilities x the FP8 V copy (kernels_f8.metal)
                    e.run(pso("attn_softmax8n_f8" + F8), [P16, STT, nil, nil, nil, nil, P8B, nil, CBB, g.vs],
                          [(UInt32(TK), 2), (UInt32(a0), 3), (UInt32(n), 4), (UInt32(p0), 5), (F8VS, 7), (g.sh, 10)],
                          grid: sz(24 * T), tg: sz(256), offsets: [9: off * 2])
                    e.run(pso("attn_pv_f8" + F8K), [P8B, l.Vc, OA, nil, nil, nil, CBB], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(n), 5)],
                          grid: sz(2, 6 * T / 64, 4), tg: sz(128))
                } else if fold {   // softmax folded: PV computes exp(s - m) on load (row max from attn_qk8iskm), no P pass
                    let f = (FOLDV == "128" && n % 128 != 0) ? "64" : FOLDV
                    e.run(pso("attn_pv_wm8_" + f), [P16, l.Vc, OA, MXB, STT], [(UInt32(Lmax), 5), (UInt32(a0), 6), (UInt32(TK), 7), (UInt32(n), 8), (UInt32(p0), 9)],
                          grid: sz(2, 6 * T / 64, 4), tg: sz(128))
                } else if K8V16 {   // plain softmax, fp16-V PV over the GPU's copy (absolute key rows)
                    e.run(pso("attn_softmax8n"), [P16, STT], [(UInt32(TK), 2), (UInt32(a0), 3), (UInt32(n), 4), (UInt32(p0), 5)], grid: sz(24 * T), tg: sz(256))
                    e.run(pso("attn_pv_w"), [P16, l.Vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(2, 6 * T / 64, 4), tg: sz(128))
                } else if QK8 && KV8D {
                    e.run(pso("attn_softmax8n"), [P16, STT], [(UInt32(TK), 2), (UInt32(a0), 3), (UInt32(n), 4), (UInt32(p0), 5)], grid: sz(24 * T), tg: sz(256))
                    e.run(pso("attn_pv8d"), [P16, g.v, OA, nil, nil, nil, g.vs], [(SIMD2<UInt32>(g.vh, g.vr), 3), (UInt32(TK), 4), (UInt32(n), 5), (g.sh, 7)],
                          grid: sz(2, 6 * T / 64, 4), tg: sz(128), offsets: [1: off * Int(g.vr), 6: off * 2])
                } else {
                e.run(pso("attn_softmax8"), [P16, STT, nil, nil, nil, nil, g.ks, g.vs], [(UInt32(TK), 2), (UInt32(a0), 3), (UInt32(n), 4), (UInt32(p0), 5), (g.sh, 8)],
                      grid: sz(24 * T), tg: sz(256), offsets: [6: off * 2, 7: off * 2])
                if PVWW { e.run(pso("attn_pv8w"), [P16, g.v, OA], [(SIMD2<UInt32>(g.vh, g.vr), 3), (UInt32(TK), 4), (UInt32(n), 5)], grid: sz(1, 6 * T / 64, 4), tg: sz(256),
                      offsets: [1: off * Int(g.vr)]) }
                else { e.run(pso("attn_pv8"), [P16, g.v, OA], [(SIMD2<UInt32>(g.vh, g.vr), 3), (UInt32(TK), 4), (UInt32(n), 5)], grid: sz(2, 6 * T / 64, 4), tg: sz(128),
                      offsets: [1: off * Int(g.vr)]) }
                }
                e.run(pso("attn_combine"), [OA, STT, OR, MD], [(UInt32(a0 == 0 ? 1 : 0), 4)], grid: sz(256, 24 * T), tg: sz(256, 1, 1), groups: false)
                a0 += n
            }
            e.run(pso("attn_finish"), [OR, MD, BIG, ON], grid: sz(256, T, 24), tg: sz(64, 1, 1), groups: false)
            if core { return }   // A8 attention core: AO (ANE) does the rest
            rotate(e, l, "mix_out", ON, ON16, 6144)
            if A8NOOUT { return }
            if I8 && I8A { e.run(pso("quant8_blk"), [ON16, X8B, XSB], grid: sz(6144 / 1024, T / 32), tg: sz(256)); lutI8(e, l, "o", X8B, XSB, ON16, EPI ? X : Y, acc: EPI) }
            else { lut(e, l, "o", ON16, EPI ? X : Y, acc: EPI) }
            if !EPI && !deferAdd { e.run(pso("add_into"), [X, Y], grid: sz(HID * T), tg: sz(256), groups: false) }
            return
        }
        if QK8 { e.run(pso("attn_prep16q8"), [BIG, KP, VP, l.bufs["qn"], l.bufs["kn"], QA, l.Kc, l.Vc, nil, nil, nil, csAll, Q8B, QSB, l.K8c, l.KSc],
              [(UInt32(p0), 8), (UInt32(Lmax), 9), (eps, 10)], grid: sz(32, T), tg: sz(256), offsets: [11: p0 * 64 * 2 * 4]) } else {
        e.run(pso("attn_prep16"), [BIG, KP, VP, l.bufs["qn"], l.bufs["kn"], QA, l.Kc, l.Vc, nil, nil, nil, csAll],
              [(UInt32(p0), 8), (UInt32(Lmax), 9), (eps, 10)], grid: sz(32, T), tg: sz(256), offsets: [11: p0 * 64 * 2 * 4]) }
        let ntile = (Int(p0) + T + 63) / 64
        let ns = max(1, min(16, ntile / 16))   // key splits: at least 16 key tiles each
        if !FA2.isEmpty {
            let nsf = FA2NS > 0 ? FA2NS : ns
            e.run(pso(FA2.hasPrefix("s_") ? "attn_fa2" + FA2 : "attn_fa2_" + FA2), [QA, l.Kc, l.Vc, OPART, STPART], [(UInt32(Lmax), 5), (UInt32(p0), 6), (UInt32(nsf), 7)],
                  grid: sz(6 * T / (FA2NSG * FA2RM), 4, nsf), tg: sz(32 * FA2NSG))
            e.run(pso("attn_combine_splits"), [OPART, STPART, OR, MD], [(UInt32(nsf), 4)], grid: sz(256, 24 * T), tg: sz(256, 1, 1), groups: false)
        } else if FLASH {  // fused: scores stay on chip (less DRAM traffic next to the ANE; slower GPU time alone)
            e.run(pso("attn_flash"), [QA, l.Kc, l.Vc, OPART, STPART], [(UInt32(Lmax), 5), (UInt32(p0), 6), (UInt32(ns), 7)],
                  grid: sz(6 * T / 64, 4, ns), tg: sz(256))
            e.run(pso("attn_combine_splits"), [OPART, STPART, OR, MD], [(UInt32(ns), 4)], grid: sz(256, 24 * T), tg: sz(256, 1, 1), groups: false)
        }
        var a0 = (FLASH || !FA2.isEmpty) ? Int(L_) : 0
        while a0 < Int(L_) {
            let n = min(TK, Int(L_) - a0)
            if SMFV == "3" {
                e.run(pso("attn_qk_h"), [QA, l.Kc, P16], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128))
                e.run(pso("attn_softmax_hr"), [P16, STT], [(UInt32(TK), 2), (UInt32(a0), 3), (UInt32(n), 4), (UInt32(p0), 5)], grid: sz(24 * T), tg: sz(256))
                if PVWW { e.run(pso("attn_pv_ww"), [P16, l.Vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(1, 6 * T / 64, 4), tg: sz(256)) }
                else { e.run(pso("attn_pv_w"), [P16, l.Vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(2, 6 * T / 64, 4), tg: sz(128)) }
            } else if SMFV == "2" {
                e.run(pso("attn_qk_hm"), [QA, l.Kc, P16, MXB], [(UInt32(Lmax), 4), (UInt32(a0), 5), (UInt32(TK), 6), (UInt32(p0), 7)],
                      grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128))
                e.run(pso("attn_softmax_hm"), [P16, STT, MXB], [(UInt32(TK), 3), (UInt32(a0), 4), (UInt32(n), 5), (UInt32(p0), 6)], grid: sz(24 * T), tg: sz(256))
                if PVWW { e.run(pso("attn_pv_ww"), [P16, l.Vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(1, 6 * T / 64, 4), tg: sz(256)) }
                else { e.run(pso("attn_pv_w"), [P16, l.Vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(2, 6 * T / 64, 4), tg: sz(128)) }
            } else if SMF {
                e.run(pso("attn_qk_hm"), [QA, l.Kc, P16, MXB], [(UInt32(Lmax), 4), (UInt32(a0), 5), (UInt32(TK), 6), (UInt32(p0), 7)],
                      grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128))
                e.run(pso("attn_pv_wm"), [P16, l.Vc, OA, MXB, STT], [(UInt32(Lmax), 5), (UInt32(a0), 6), (UInt32(TK), 7), (UInt32(n), 8), (UInt32(p0), 9)],
                      grid: sz(2, 6 * T / 64, 4), tg: sz(128))
            } else if QX {
                e.run(pso("attn_qk_mx"), [QA, l.Kc, MXB], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(p0), 6)],
                      grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128))
                e.run(pso("attn_qk_ex"), [QA, l.Kc, P16, MXB, SUB], [(UInt32(Lmax), 5), (UInt32(a0), 6), (UInt32(TK), 7), (UInt32(p0), 8), (UInt32(n), 9)],
                      grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128))
                e.run(pso("attn_qx_st"), [MXB, SUB, STT], [(UInt32(TK), 3), (UInt32(n), 4)], grid: sz(24 * T), tg: sz(64), groups: false)
                if PVWW { e.run(pso("attn_pv_ww"), [P16, l.Vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(1, 6 * T / 64, 4), tg: sz(256)) }
                else { e.run(pso("attn_pv_w"), [P16, l.Vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(2, 6 * T / 64, 4), tg: sz(128)) }
            } else {
            if QK8 { e.run(pso("attn_qk8i"), [Q8B, l.K8c, P16, nil, nil, nil, QSB, l.KSc], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128)) }
            else if QKT.isEmpty { e.run(pso("attn_qk_h"), [QA, l.Kc, P16], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5)], grid: sz(n / 64, 6 * T / 64, 4), tg: sz(128)) }
            else { let (r, k, th) = tdims(QKT); e.run(pso("attn_qk_" + QKT), [QA, l.Kc, P16], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5)], grid: sz(n / k, 6 * T / r, 4), tg: sz(th)) }
            e.run(pso("attn_softmax_h"), [P16, STT], [(UInt32(TK), 2), (UInt32(a0), 3), (UInt32(n), 4), (UInt32(p0), 5)], grid: sz(24 * T), tg: sz(256))
            if !PVT.isEmpty { let (r, dn, th) = tdims(PVT); e.run(pso("attn_pv_" + PVT), [P16, l.Vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(256 / dn, 6 * T / r, 4), tg: sz(th)) }
            else if PVW { e.run(pso("attn_pv_w"), [P16, l.Vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(2, 6 * T / 64, 4), tg: sz(128)) }
            else { e.run(pso("attn_pv_t"), [P16, l.Vc, OA], [(UInt32(Lmax), 3), (UInt32(a0), 4), (UInt32(TK), 5), (UInt32(n), 6)], grid: sz(4, 6 * T / 64, 4), tg: sz(128)) }
            }
            e.run(pso("attn_combine"), [OA, STT, OR, MD], [(UInt32(a0 == 0 ? 1 : 0), 4)], grid: sz(256, 24 * T), tg: sz(256, 1, 1), groups: false)
            a0 += n
        }
        e.run(pso("attn_finish"), [OR, MD, BIG, ON], grid: sz(256, T, 24), tg: sz(64, 1, 1), groups: false)
        if core { return }
        rotate(e, l, "mix_out", ON, ON16, 6144)
        if A8NOOUT { return }   // A8: the output projection runs on the ANE (ON16 -> AO)
        if I8 && I8A { e.run(pso("quant8_blk"), [ON16, X8B, XSB], grid: sz(6144 / 1024, T / 32), tg: sz(256)); lutI8(e, l, "o", X8B, XSB, ON16, EPI ? X : Y, acc: EPI) }
        else { lut(e, l, "o", ON16, EPI ? X : Y, acc: EPI) }
    }
    if !EPI && !deferAdd { e.run(pso("add_into"), [X, Y], grid: sz(HID * T), tg: sz(256), groups: false) }
}
// RoPE cos / sin of every position [pos][64 dims][2], in double as setRope computes them (no per-call CPU work, and
// command buffers in flight never see the table change)
let csAll: MTLBuffer = {
    let n = Lmax + T, b = buf(n * 64 * 2 * 4), p = b.contents().bindMemory(to: Float.self, capacity: n * 128)
    for t in 0..<n { for d in 0..<64 { let a = Double(t) * ropeInv[d % 32]; p[(t * 64 + d) * 2] = Float(cos(a)); p[(t * 64 + d) * 2 + 1] = Float(sin(a)) } }
    return b
}()
var inflight: [Int32: MTLCommandBuffer] = [:]
var ticket: Int32 = 0
let inflightLock = NSLock()   // v8_submit and v8_wait may run on different threads
/// The layer's FFN on the GPU too (X = X + FFN(X)), for layers the ANE skips (load balancing).
func encodeFfn(_ e: Enc, _ l: L) {
    let eps = man.eps
    e.run(pso(T == 256 ? "rms_partial_g256" : "rms_partial"), [X, RP], grid: sz(HID / 320), tg: sz(256))
    if l.m.mats["gate_c0"] != nil {   // chunked FFN of a shared-weight build (prep_ffn.py)
        if FU("rms"), let s = l.bufs["rot_mlp_in"] {
            e.run(pso("rms_fwht4"), [X, l.bufs["ln2"], s, HQ16, H16, RP], [(UInt32(HID), 6), (eps, 7)], grid: sz(HID / 1024, T / 4), tg: sz(512))
        } else {
            e.run(pso("rms_apply"), [X, l.bufs["ln2"], H32, H16, RP], [(UInt32(HID), 5), (eps, 6)], grid: sz(HID * T), tg: sz(256), groups: false)
            rotate(e, l, "mlp_in", H32, HQ16, HID)
        }
        let nc = (0..<8).filter { l.m.mats["gate_c\($0)"] != nil }.count
        var x8: MTLBuffer? = nil, xs: MTLBuffer? = nil
        if I8F { e.run(pso("quant8_blk"), [HQ16, X8A, XSA], grid: sz(HID / 1024, T / 32), tg: sz(256)); x8 = X8A; xs = XSA }
        for j in 0..<nc { lutChunk(e, l, "gate_c\(j)", HQ16, x8, xs, G, acc: false); lutChunk(e, l, "up_c\(j)", HQ16, x8, xs, U, acc: false) }
        if let s = l.bufs["rot_mlp_mid"] { e.run(pso("fwht4_swiglu"), [G, s, A16, U], grid: sz(INTER / 1024, T / 4), tg: sz(512)) }
        else { e.run(pso("swiglu"), [G, U, G], grid: sz(INTER * T), tg: sz(256), groups: false); rotate(e, l, "mlp_mid", G, A16, INTER) }
        if I8F { e.run(pso("quant8_blk"), [A16, X8C, XSC], grid: sz(INTER / 1024, T / 32), tg: sz(256)); x8 = X8C; xs = XSC }
        for j in 0..<nc { lutChunk(e, l, "down_c\(j)", A16, x8, xs, Y, acc: j > 0) }
        e.run(pso("add_into"), [X, Y], grid: sz(HID * T), tg: sz(256), groups: false)
        return
    }
    e.run(pso("rms_apply"), [X, l.bufs["ln2"], H32, H16, RP], [(UInt32(HID), 5), (eps, 6)], grid: sz(HID * T), tg: sz(256), groups: false)
    rotate(e, l, "mlp_in", H32, HQ16, HID)
    lut(e, l, "gate", HQ16, G); lut(e, l, "up", HQ16, U)
    if let s = l.bufs["rot_mlp_mid"] { e.run(pso("fwht4_swiglu"), [G, s, A16, U], grid: sz(INTER / 1024, T / 4), tg: sz(512)) }
    else { e.run(pso("swiglu"), [G, U, G], grid: sz(INTER * T), tg: sz(256), groups: false); rotate(e, l, "mlp_mid", G, A16, INTER) }
    lut(e, l, "down", A16, Y)
    e.run(pso("add_into"), [X, Y], grid: sz(HID * T), tg: sz(256), groups: false)
}
var layerIndex: [Int: Int] = [:]   // model layer i -> position in `layers`
var wrapped: [MTLBuffer] = []
@_cdecl("v8_init")
public func v8_init() -> Int32 {
    for (k, l) in layers.enumerated() { layerIndex[l.m.i] = k }
    _ = csAll.length
    _ = pso("from_half"); _ = pso("to_half")
    return Int32(layers.count)
}
/// A CPU-visible, page-aligned region (an ANE IOSurface buffer) as a Metal buffer, zero-copy. Returns its handle.
@_cdecl("v8_wrap")
public func v8_wrap(_ p: UnsafeMutableRawPointer, _ n: Int) -> Int32 {
    let page = Int(getpagesize())
    guard Int(bitPattern: p) % page == 0, let b = dev.makeBuffer(bytesNoCopy: p, length: (n + page - 1) / page * page,
                                                                 options: .storageModeShared, deallocator: nil) else { return -1 }
    wrapped.append(b); return Int32(wrapped.count - 1)
}
func encodeMix(_ layer: Int32, _ src: Int32, _ dst: Int32, _ p0: Int32) -> MTLCommandBuffer? {
    guard let k = layerIndex[Int(layer)] else { return nil }
    let l = layers[k]
    let cb = q.makeCommandBuffer()!
    let dAdd = FU("add") && !EPI   // residual add folded into the output conversion
    if FU("enc") {   // conversions inside the mixer encoder (barriers instead of encoder boundaries)
        let e = Enc(mixEncoder(cb))
        e.run(pso("from_half"), [wrapped[Int(src)], X], grid: sz(HID * T), tg: sz(256), groups: false)
        encodeMixer(e, l, p0: Int(p0), deferAdd: dAdd)
        if dAdd { e.run(pso("to_half_add"), [X, wrapped[Int(dst)], Y], grid: sz(HID * T), tg: sz(256), groups: false) }
        else { e.run(pso("to_half"), [X, wrapped[Int(dst)]], grid: sz(HID * T), tg: sz(256), groups: false) }
        e.e.endEncoding()
        return cb
    }
    let e0 = cb.makeComputeCommandEncoder()!
    e0.setComputePipelineState(pso("from_half")); e0.setBuffer(wrapped[Int(src)], offset: 0, index: 0); e0.setBuffer(X, offset: 0, index: 1)
    e0.dispatchThreads(sz(HID * T), threadsPerThreadgroup: sz(256)); e0.endEncoding()
    let e = Enc(mixEncoder(cb))
    encodeMixer(e, l, p0: Int(p0), deferAdd: dAdd)
    e.e.endEncoding()
    let e2 = cb.makeComputeCommandEncoder()!
    e2.setComputePipelineState(pso(dAdd ? "to_half_add" : "to_half")); e2.setBuffer(X, offset: 0, index: 0); e2.setBuffer(wrapped[Int(dst)], offset: 0, index: 1)
    if dAdd { e2.setBuffer(Y, offset: 0, index: 2) }
    e2.dispatchThreads(sz(HID * T), threadsPerThreadgroup: sz(256)); e2.endEncoding()
    return cb
}
/// Pair mode (NT = 128): one call covers two consecutive 64-token blocks at p0 (block A = rows 0..63, B = 64..127), read
/// from two ANE output buffers and written to two ANE input buffers.
@_cdecl("v8_submit_pair")
public func v8_submit_pair(_ layer: Int32, _ srcA: Int32, _ srcB: Int32, _ dstA: Int32, _ dstB: Int32, _ p0: Int32) -> Int32 {
    v8_submit_pair_full(layer, srcA, srcB, dstA, dstB, p0, 0)
}
/// full != 0: the layer's FFN runs on the GPU as well (dst may be src: the inputs are read before the outputs).
@_cdecl("v8_submit_pair_full")
public func v8_submit_pair_full(_ layer: Int32, _ srcA: Int32, _ srcB: Int32, _ dstA: Int32, _ dstB: Int32, _ p0: Int32, _ full: Int32) -> Int32 {
    guard let k = layerIndex[Int(layer)] else { return -1 }
    let l = layers[k]
    let cb = q.makeCommandBuffer()!
    let dAdd = FU("add") && !EPI && full == 0   // residual add folded into the output conversion (not when the FFN follows)
    if FU("enc") {   // conversions inside the mixer encoder (barriers instead of encoder boundaries)
        let e = Enc(mixEncoder(cb))
        e.run(pso("from_half_pair"), [wrapped[Int(srcA)], wrapped[Int(srcB)], X], grid: sz(HID * T), tg: sz(256), groups: false)
        encodeMixer(e, l, p0: Int(p0), deferAdd: dAdd)
        if full != 0 { encodeFfn(e, l) }
        if dAdd { e.run(pso("to_half_pair_add"), [X, wrapped[Int(dstA)], wrapped[Int(dstB)], Y], grid: sz(HID * T), tg: sz(256), groups: false) }
        else { e.run(pso("to_half_pair"), [X, wrapped[Int(dstA)], wrapped[Int(dstB)]], grid: sz(HID * T), tg: sz(256), groups: false) }
        e.e.endEncoding()
    } else {
    let e0 = cb.makeComputeCommandEncoder()!
    e0.setComputePipelineState(pso("from_half_pair")); e0.setBuffer(wrapped[Int(srcA)], offset: 0, index: 0)
    e0.setBuffer(wrapped[Int(srcB)], offset: 0, index: 1); e0.setBuffer(X, offset: 0, index: 2)
    e0.dispatchThreads(sz(HID * T), threadsPerThreadgroup: sz(256)); e0.endEncoding()
    let e = Enc(mixEncoder(cb))
    encodeMixer(e, l, p0: Int(p0), deferAdd: dAdd)
    if full != 0 { encodeFfn(e, l) }
    e.e.endEncoding()
    let e2 = cb.makeComputeCommandEncoder()!
    e2.setComputePipelineState(pso(dAdd ? "to_half_pair_add" : "to_half_pair")); e2.setBuffer(X, offset: 0, index: 0)
    e2.setBuffer(wrapped[Int(dstA)], offset: 0, index: 1); e2.setBuffer(wrapped[Int(dstB)], offset: 0, index: 2)
    if dAdd { e2.setBuffer(Y, offset: 0, index: 3) }
    e2.dispatchThreads(sz(HID * T), threadsPerThreadgroup: sz(256)); e2.endEncoding()
    }
    inflightLock.lock(); ticket += 1; let t = ticket; inflight[t] = cb; inflightLock.unlock()
    cb.commit()
    return t
}
/// Pair mode, attention layer, core only: q (12288 ch, with gate) / k / v (1024 ch each) of blocks A and B from the ANE's
/// ffnq2 outputs (fp16 [C][64] each); writes the gated attention output (6144 ch, before the mix_out rotation) to oA / oB
/// (the ANE's offn2 inputs). The K / V caches are written as in v8_submit_pair. h = [qA, qB, kA, kB, vA, vB, oA, oB].
@_cdecl("v8_submit_attcore")
public func v8_submit_attcore(_ layer: Int32, _ h: UnsafePointer<Int32>, _ p0: Int32) -> Int32 {
    guard let k = layerIndex[Int(layer)], layers[k].m.kind != "gdn" else { return -1 }
    let l = layers[k]
    let cb = q.makeCommandBuffer()!
    let e0 = cb.makeComputeCommandEncoder()!
    e0.setComputePipelineState(pso("from_half_pair"))
    for (j, (dst, c)) in [(BIG, 12288), (KP, 1024), (VP, 1024)].enumerated() {
        e0.setBuffer(wrapped[Int(h[2 * j])], offset: 0, index: 0); e0.setBuffer(wrapped[Int(h[2 * j + 1])], offset: 0, index: 1)
        e0.setBuffer(dst, offset: 0, index: 2)
        e0.dispatchThreads(sz(c * T), threadsPerThreadgroup: sz(256))
    }
    e0.endEncoding()
    let e = Enc(mixEncoder(cb))
    encodeMixer(e, l, p0: Int(p0), core: true)
    e.e.endEncoding()
    let e2 = cb.makeComputeCommandEncoder()!
    e2.setComputePipelineState(pso("to_half_pair")); e2.setBuffer(ON, offset: 0, index: 0)
    e2.setBuffer(wrapped[Int(h[6])], offset: 0, index: 1); e2.setBuffer(wrapped[Int(h[7])], offset: 0, index: 2)
    e2.dispatchThreads(sz(6144 * T), threadsPerThreadgroup: sz(256)); e2.endEncoding()
    inflightLock.lock(); ticket += 1; let t = ticket; inflight[t] = cb; inflightLock.unlock()
    cb.commit()
    return t
}
/// The layer's FFN alone (test entry): dst = x + FFN(x) (fp16 [5120][T]). GPU ms.
@_cdecl("v8_ffn")
public func v8_ffn(_ layer: Int32, _ src: Int32, _ dst: Int32) -> Double {
    guard let k = layerIndex[Int(layer)] else { return -1 }
    let l = layers[k], cb = q.makeCommandBuffer()!, e = Enc(mixEncoder(cb))
    e.run(pso("from_half"), [wrapped[Int(src)], X], grid: sz(HID * T), tg: sz(256), groups: false)
    encodeFfn(e, l)
    e.run(pso("to_half"), [X, wrapped[Int(dst)]], grid: sz(HID * T), tg: sz(256), groups: false)
    e.e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return cb.error != nil ? -2 : (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}
/// One layer's mixer for a 64-token block at p0: x = src (fp16 [5120][T]); dst = x + mixer(x) (fp16). GPU ms.
@_cdecl("v8_mix")
public func v8_mix(_ layer: Int32, _ src: Int32, _ dst: Int32, _ p0: Int32) -> Double {
    guard let cb = encodeMix(layer, src, dst, p0) else { return -1 }
    cb.commit(); cb.waitUntilCompleted()
    if cb.error != nil { return -2 }
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}
/// v8_mix without waiting: returns a ticket for v8_wait (command buffers run in submission order on one queue, and
/// share the scratch buffers: the order is the dependency).
@_cdecl("v8_submit")
public func v8_submit(_ layer: Int32, _ src: Int32, _ dst: Int32, _ p0: Int32) -> Int32 {
    guard let cb = encodeMix(layer, src, dst, p0) else { return -1 }
    inflightLock.lock(); ticket += 1; let t = ticket; inflight[t] = cb; inflightLock.unlock()
    cb.commit()
    return t
}
/// v8_wait that also reports the command buffer's GPU start / end (host seconds, the CACurrentMediaTime /
/// mach_absolute_time base that Python's time.perf_counter shares on macOS).
@_cdecl("v8_wait2")
public func v8_wait2(_ t: Int32, _ out: UnsafeMutablePointer<Double>) -> Double {
    inflightLock.lock(); let c = inflight.removeValue(forKey: t); inflightLock.unlock()
    guard let cb = c else { return -1 }
    cb.waitUntilCompleted()
    if cb.error != nil { return -2 }
    out[0] = cb.gpuStartTime; out[1] = cb.gpuEndTime
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}
@_cdecl("v8_wait")
public func v8_wait(_ t: Int32) -> Double {
    inflightLock.lock(); let c = inflight.removeValue(forKey: t); inflightLock.unlock()
    guard let cb = c else { return -1 }
    cb.waitUntilCompleted()
    if cb.error != nil { return -2 }
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}
@_cdecl("v8_reset")
public func v8_reset() { for l in layers { l.reset() } }
/// The KV rows [p0, p0 + n) of attention layer `layer`: K and V as [4][n][256] fp16 each, into kdst / vdst.
@_cdecl("v8_kv")
public func v8_kv(_ layer: Int32, _ p0: Int32, _ n: Int32, _ kdst: UnsafeMutableRawPointer, _ vdst: UnsafeMutableRawPointer) -> Int32 {
    guard let k = layerIndex[Int(layer)], layers[k].m.kind != "gdn" else { return -1 }
    let l = layers[k]
    for (src, dst) in [(l.Kc!, kdst), (l.Vc!, vdst)] {
        for h in 0..<4 { memcpy(dst + h * Int(n) * 256 * 2, src.contents() + ((h * Lmax + Int(p0)) * 256) * 2, Int(n) * 256 * 2) }
    }
    return 0
}
/// DeltaNet state of layer `layer` in the ANE's layout: conv rows [3][10240] f16 and rec [48][dk][dv] f16 x GDN_SV (64).
@_cdecl("v8_state")
public func v8_state(_ layer: Int32, _ conv: UnsafeMutablePointer<Float16>, _ rec: UnsafeMutablePointer<Float16>) -> Int32 {
    guard let k = layerIndex[Int(layer)], layers[k].m.kind == "gdn" else { return -1 }
    let l = layers[k]
    let cs = l.CS!.contents().bindMemory(to: Float.self, capacity: 3 * 10240)
    for i in 0..<(3 * 10240) { conv[i] = Float16(cs[i]) }
    let s = l.S!.contents().bindMemory(to: Float.self, capacity: 48 * 128 * 128)
    var tmp = [Float](repeating: 0, count: 128 * 128)   // per head: transpose [v][k] -> [k][v] (vDSP), then x 64 -> f16
    for h in 0..<48 {
        tmp.withUnsafeMutableBufferPointer { t in
            vDSP_mtrans(s + h * 16384, 1, t.baseAddress!, 1, 128, 128)
            let o = rec + h * 16384
            for i in 0..<16384 { o[i] = Float16(64 * t[i]) }
        }
    }
    return 0
}
/// PROFILE=1: per-kernel GPU ms accumulated so far (each dispatch in its own command buffer), printed and cleared.
/// FOLD at run time: mode 0 off, 1 = "64" (double-buffered 64-key chunks), 2 = "64s" (single buffer), 3 = "128";
/// minKeys: fold only calls that attend to at least that many keys. Returns 1 if the path can fold (KV8 + K8V16).
@_cdecl("v8_fold")
public func v8_fold(_ mode: Int32, _ minKeys: Int32) -> Int32 {
    let names = ["", "64", "64s", "128"]
    FOLDV = (mode >= 0 && Int(mode) < names.count) ? names[Int(mode)] : ""; FOLD_MIN = Int(minKeys)
    return K8V16 ? 1 : 0
}
@_cdecl("v8_prof_dump")
public func v8_prof_dump() {
    let tot = profT.values.reduce(0, +)
    for (n, t) in profT.sorted(by: { $0.value > $1.value }) {
        print(String(format: "  %-22@ %8.3f ms  %5.1f%%  (%d calls)", n, t * 1e3, 100 * t / tot, profN[n]!))
    }
    print(String(format: "  total %.3f ms", tot * 1e3)); fflush(stdout)
    profT = [:]; profN = [:]
}

/// KV8: register one INT8 KV segment of attention layer `layer` (the ANE's buffers, wrapped with v8_wrap): positions
/// [start, start + rows); strides in bytes (k: head, dim; v: head, row), scales: head stride in halves.
@_cdecl("v8_kv8_seg")
public func v8_kv8_seg(_ layer: Int32, _ start: Int32, _ rows: Int32, _ k: Int32, _ v: Int32, _ ks: Int32, _ vs: Int32,
                       _ kh: Int32, _ kd: Int32, _ vh: Int32, _ vr: Int32, _ sh: Int32) -> Int32 {
    guard KV8, let i = layerIndex[Int(layer)], layers[i].m.kind != "gdn" else { return -1 }
    let l = layers[i]
    l.segs.removeAll { $0.start == Int(start) }
    l.segs.append(Seg(start: Int(start), rows: Int(rows), k: wrapped[Int(k)], v: wrapped[Int(v)], ks: wrapped[Int(ks)], vs: wrapped[Int(vs)],
                      kh: UInt32(kh), kd: UInt32(kd), vh: UInt32(vh), vr: UInt32(vr), sh: UInt32(sh)))
    l.segs.sort { $0.start < $1.start }
    return Int32(l.segs.count)
}

// ---- A8 FFN (patch_a8.py) ----
var A8PAL: MTLBuffer? = nil, A8XS: UInt32 = 5120, A8TS: UInt32 = 2, A8YS: UInt32 = 5120
let A8EXP = ProcessInfo.processInfo.environment["A8EXP"] ?? ""   // "t": tiled expander (a8_expand_t)
let q2 = dev.makeCommandQueue()!   // weight expansion runs beside the mixer queue
/// pal8: int8 [64][3][64][2] (prep_a8.py pal8.bin); xs / ts / ys: the ANE A8 program's row strides (x bytes, tx and y halves).
@_cdecl("v8_a8_setup")
public func v8_a8_setup(_ p: UnsafeRawPointer, _ n: Int, _ xs: Int32, _ ts: Int32, _ ys: Int32) -> Int32 {
    A8PAL = dev.makeBuffer(bytes: p, length: n, options: .storageModeShared)
    A8XS = UInt32(xs); A8TS = UInt32(ts); A8YS = UInt32(ys)
    if A8EXP == "t" {   // tiled expander: tables sorted by row, 32 consecutive G-16 subgroups per 1024-row tile
        for l in layers { for m in ["gate", "up", "down"] { for j in 0..<8 {
            guard let mt = l.m.mats["\(m)_c\(j)"], let tb = l.bufs["tab_\(m)_c\(j)"] else { continue }
            let n = tb.length / 12, p = tb.contents().bindMemory(to: UInt32.self, capacity: 3 * n)
            var rows = (0..<n).map { (p[3 * $0], p[3 * $0 + 1], p[3 * $0 + 2]) }
            rows.sort { $0.2 < $1.2 }
            for (i, r) in rows.enumerated() { p[3 * i] = r.0; p[3 * i + 1] = r.1; p[3 * i + 2] = r.2 }
            if mt.G != 16 || n % 32 != 0 || mt.cin % 32 != 0 || (0..<n).contains(where: { p[3 * $0 + 2] != p[2] + UInt32(16 * $0) }) { return -3 }
            if m != "down" && (2 * p[2]) % 1024 != 0 { return -4 }
        } } }
    }
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
            ce.setComputePipelineState(pso(A8EXP == "t" ? "a8_expand_t" : "a8_expand"))
            ce.setBuffer(kern[mt.hwx], offset: 0, index: 0); ce.setBuffer(tb, offset: 0, index: 1)
            ce.setBuffer(pal, offset: (Int(layer) * 3 + mi) * 128, index: 2); ce.setBuffer(wrapped[Int(dst)], offset: 0, index: 3)
            ce.setBytes(&prm, length: 16, index: 4)
            if A8EXP == "t" { ce.dispatchThreadgroups(sz(mt.cin / 32, tb.length / 12 / 32), threadsPerThreadgroup: sz(1024)) }
            else { ce.dispatchThreads(sz(mt.cin, tb.length / 12), threadsPerThreadgroup: sz(64, 4)) }
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
        e.run(pso(T == 256 ? "rms_partial_g256" : "rms_partial"), [X, RP], grid: sz(HID / 320), tg: sz(256))
        let att = flags & 4 != 0
        e.run(pso("rms_fwht4"), [X, l.bufs[att ? "ln1" : "ln2"], l.bufs[att ? "rot_mix_in" : "rot_mlp_in"], HQ16, H16, RP], [(UInt32(HID), 6), (man.eps, 7)],
              grid: sz(HID / 1024, T / 4), tg: sz(512))
        e.run(pso("a8_quant_tm"), [HQ16, wrapped[Int(x8)], wrapped[Int(tx)]], [(SIMD4<UInt32>(UInt32(xrow), A8XS, A8TS, 0), 3)],
              grid: sz(2, T), tg: sz(256))
    }
    e.e.endEncoding()
    inflightLock.lock(); ticket += 1; let t = ticket; inflight[t] = cb; inflightLock.unlock()
    cb.commit()
    return t
}

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
    for (mi, (key, dst, mode)) in [("q", wq, 0), ("o", wo, 1)].enumerated() where dst >= 0 {
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

/// The inverse of v8_state: a DeltaNet layer's state from the ANE's layout into the GPU engine (continue from a bridge
/// state). conv [3][10240] f16, rec [48][dk][dv] f16 x 64.
@_cdecl("v8_set_state")
public func v8_set_state(_ layer: Int32, _ conv: UnsafePointer<Float16>, _ rec: UnsafePointer<Float16>) -> Int32 {
    guard let k = layerIndex[Int(layer)], layers[k].m.kind == "gdn" else { return -1 }
    let l = layers[k]
    let cs = l.CS!.contents().bindMemory(to: Float.self, capacity: 3 * 10240)
    for i in 0..<(3 * 10240) { cs[i] = Float(conv[i]) }
    let st = l.S!.contents().bindMemory(to: Float.self, capacity: 48 * 128 * 128)
    var tmp = [Float](repeating: 0, count: 128 * 128)   // per head: f16 / 64 -> [k][v], transpose -> [v][k] (vDSP)
    for h in 0..<48 {
        tmp.withUnsafeMutableBufferPointer { t in
            let r = rec + h * 16384
            for i in 0..<16384 { t[i] = Float(r[i]) / 64 }
            vDSP_mtrans(t.baseAddress!, 1, st + h * 16384, 1, 128, 128)
        }
    }
    return 0
}

// ---- valid rows in a pair call (patch_nv.py) ----
var A8NV: UInt32 = UInt32(T)   // rows of the call that are real tokens (the rest is padding); T outside v8_submit_pair_nv
/// v8_submit_pair with only the first nv rows real (the prompt's last pair): DeltaNet state and conv history stop there.
@_cdecl("v8_submit_pair_nv")
public func v8_submit_pair_nv(_ layer: Int32, _ srcA: Int32, _ srcB: Int32, _ dstA: Int32, _ dstB: Int32, _ p0: Int32, _ nv: Int32) -> Int32 {
    guard nv >= 1 && Int(nv) <= T else { return -1 }
    A8NV = UInt32(nv)
    let t = v8_submit_pair_full(layer, srcA, srcB, dstA, dstB, p0, 0)
    A8NV = UInt32(T)
    return t
}

/// Quad mode (queue 9b): one GDN layer on four consecutive 64-row blocks at p0 (T = 256), from four ANE output buffers to
/// four ANE input buffers; nv = real rows (T for a full quad; padding after: DeltaNet state and conv history stop there).
@_cdecl("v8_submit_quad")
public func v8_submit_quad(_ layer: Int32, _ s0: Int32, _ s1: Int32, _ s2: Int32, _ s3: Int32, _ d0: Int32, _ d1: Int32, _ d2: Int32, _ d3: Int32,
                           _ p0: Int32, _ nv: Int32) -> Int32 {
    guard let k = layerIndex[Int(layer)], T == 256, nv >= 1, Int(nv) <= T, FU("enc") else { return -1 }
    let l = layers[k]
    A8NV = UInt32(nv)
    defer { A8NV = UInt32(T) }
    let cb = q.makeCommandBuffer()!
    let dAdd = FU("add") && !EPI
    let e = Enc(mixEncoder(cb))
    e.run(pso("from_half_quad"), [wrapped[Int(s0)], wrapped[Int(s1)], wrapped[Int(s2)], wrapped[Int(s3)], X], grid: sz(HID * T), tg: sz(256), groups: false)
    encodeMixer(e, l, p0: Int(p0), deferAdd: dAdd)
    if dAdd { e.run(pso("to_half_quad_add"), [X, wrapped[Int(d0)], wrapped[Int(d1)], wrapped[Int(d2)], wrapped[Int(d3)], Y], grid: sz(HID * T), tg: sz(256), groups: false) }
    else { e.run(pso("to_half_quad"), [X, wrapped[Int(d0)], wrapped[Int(d1)], wrapped[Int(d2)], wrapped[Int(d3)]], grid: sz(HID * T), tg: sz(256), groups: false) }
    e.e.endEncoding()
    inflightLock.lock(); ticket += 1; let t = ticket; inflight[t] = cb; inflightLock.unlock()
    cb.commit()
    return t
}

// ---- DeltaNet state <-> the bridge's arrays on the GPU (patch_state_gpu.py) ----
/// n layers in one command buffer: GPU state -> conv / rec / pend (wrapped handles; conv / pend element counts). GPU ms.
@_cdecl("v8_state_out")
public func v8_state_out(_ n: Int32, _ lay: UnsafePointer<Int32>, _ hc: UnsafePointer<Int32>, _ hr: UnsafePointer<Int32>,
                         _ hp: UnsafePointer<Int32>, _ nconv: Int32, _ npend: Int32) -> Double {
    let cb = q.makeCommandBuffer()!, ce = cb.makeComputeCommandEncoder()!
    let pso_ = pso("gdn_state_out")
    var cnt = SIMD4<UInt32>(UInt32(nconv), 48 * 16384, UInt32(npend), 0)
    for i in 0..<Int(n) {
        guard let k = layerIndex[Int(lay[i])], layers[k].m.kind == "gdn" else { return -1 }
        let l = layers[k]
        ce.setComputePipelineState(pso_)
        ce.setBuffer(l.S!, offset: 0, index: 0); ce.setBuffer(l.CS!, offset: 0, index: 1)
        ce.setBuffer(wrapped[Int(hc[i])], offset: 0, index: 2); ce.setBuffer(wrapped[Int(hr[i])], offset: 0, index: 3)
        ce.setBuffer(wrapped[Int(hp[i])], offset: 0, index: 4); ce.setBytes(&cnt, length: 16, index: 5)
        ce.dispatchThreads(sz(Int(nconv) + 48 * 16384 + Int(npend)), threadsPerThreadgroup: sz(256))
    }
    ce.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return cb.error != nil ? -2 : (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}
/// n layers: conv / rec arrays (the bridge's) -> the GPU state (pending rows must be 0). GPU ms.
@_cdecl("v8_state_in")
public func v8_state_in(_ n: Int32, _ lay: UnsafePointer<Int32>, _ hc: UnsafePointer<Int32>, _ hr: UnsafePointer<Int32>) -> Double {
    let cb = q.makeCommandBuffer()!, ce = cb.makeComputeCommandEncoder()!
    let pso_ = pso("gdn_state_in")
    for i in 0..<Int(n) {
        guard let k = layerIndex[Int(lay[i])], layers[k].m.kind == "gdn" else { return -1 }
        let l = layers[k]
        ce.setComputePipelineState(pso_)
        ce.setBuffer(l.S!, offset: 0, index: 0); ce.setBuffer(l.CS!, offset: 0, index: 1)
        ce.setBuffer(wrapped[Int(hc[i])], offset: 0, index: 2); ce.setBuffer(wrapped[Int(hr[i])], offset: 0, index: 3)
        ce.dispatchThreads(sz(3 * 10240 + 48 * 16384), threadsPerThreadgroup: sz(256))
    }
    ce.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return cb.error != nil ? -2 : (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}

// ---- patch_fillfrom.py: the fill from a row (the rows below are the GPU's own, still valid) ----
/// Every attention layer's Vc rows [from, upto) from its registered ANE segments (from rounded down to 64). GPU ms.
@_cdecl("v8_kv8_v16_fill_from")
public func v8_kv8_v16_fill_from(_ from: Int32, _ upto: Int32) -> Double {
    guard K8V16 else { return -1 }
    let f0 = max(0, Int(from) / 64 * 64), u = Int(upto)
    if f0 >= u { return 0 }
    let cb = q.makeCommandBuffer()!, ce = cb.makeComputeCommandEncoder()!
    for l in layers where l.m.kind != "gdn" {
        guard let vc = l.Vc else { return -2 }
        for g in l.segs where g.start < u && g.start + g.rows > f0 {
            let r0 = max(f0, g.start) - g.start, n = min(g.start + g.rows, u) - g.start - r0   // segment-local rows
            if n <= 0 { continue }
            var prm = SIMD4<UInt32>(UInt32(Lmax), UInt32(g.start + r0), g.vh, g.vr), sh = g.sh
            ce.setComputePipelineState(pso(F8.isEmpty ? "kv8_v_to_v16" : "kv8_v_to_f8" + F8K)); var vsc = F8VS; ce.setBytes(&vsc, length: 4, index: 5)
            ce.setBuffer(g.v, offset: r0 * Int(g.vr), index: 0); ce.setBuffer(g.vs, offset: r0 * 2, index: 1)
            ce.setBuffer(vc, offset: 0, index: 2)
            ce.setBytes(&prm, length: 16, index: 3); ce.setBytes(&sh, length: 4, index: 4)
            ce.dispatchThreads(sz(256, n, 4), threadsPerThreadgroup: sz(256))
        }
    }
    ce.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return cb.error != nil ? -3 : (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}

/// Layer i's fp16 V copy (4 x Lmax x 256 half, shared storage), nil without one.
@_cdecl("v8_v16_ptr")
public func v8_v16_ptr(_ i: Int32) -> UnsafeMutableRawPointer? {
    guard Int(i) >= 0 && Int(i) < layers.count, let vc = layers[Int(i)].Vc else { return nil }
    return vc.contents()
}

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
            ce.setComputePipelineState(pso(F8.isEmpty ? "kv8_v_to_v16" : "kv8_v_to_f8" + F8K)); var vsc = F8VS; ce.setBytes(&vsc, length: 4, index: 5)
            ce.setBuffer(g.v, offset: 0, index: 0); ce.setBuffer(g.vs, offset: 0, index: 1); ce.setBuffer(vc, offset: 0, index: 2)
            ce.setBytes(&prm, length: 16, index: 3); ce.setBytes(&sh, length: 4, index: 4)
            ce.dispatchThreads(sz(256, n, 4), threadsPerThreadgroup: sz(256))
        }
    }
    ce.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return cb.error != nil ? -3 : (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}

/// The loaded attention variant, for run banners (18 Nov 2026): one line into buf (at most n bytes, NUL-terminated),
/// from the globals this process actually initialized (not the caller's environment), plus whether the library holds
/// each kernel the tile loop will dispatch. Returns the line's length, or -1 if a kernel is missing.
@_cdecl("v8_variant")
public func v8_variant(_ out: UnsafeMutablePointer<CChar>, _ n: Int32) -> Int32 {
    let f8 = KV8 && K8V16 && !F8.isEmpty
    let prep = K8V16 ? (F8.isEmpty ? "attn_prep8qv" : "attn_prep8qv_f8" + F8K) : (QK8 ? "attn_prep8q" : "attn_prep8")
    let sm = f8 ? "attn_softmax8n_f8" + F8 : (K8V16 ? (FOLDV.isEmpty ? "attn_softmax8n" : "folded") : "attn_softmax8n|8")
    let pv = f8 ? "attn_pv_f8" + F8K : (K8V16 ? (FOLDV.isEmpty ? "attn_pv_w" : "attn_pv_wm8_" + FOLDV) : "attn_pv8*")
    let fill = K8V16 ? (F8.isEmpty ? "kv8_v_to_v16" : "kv8_v_to_f8" + F8K) : "none"
    let need = [prep, QKN, sm, pv, fill].filter { !$0.contains("|") && !$0.contains("*") && $0 != "none" && $0 != "folded" }
    let ok = need.allSatisfy { lib.makeFunction(name: $0) != nil }
    let vfmt = !K8V16 ? "INT8 (ANE cache)" : (f8 ? "e4m3 (fp16 value x \(F8VS)), 1 B" : "fp16, 2 B")
    let mask = f8 && F8 == "x" ? "boundary chunk only" : "every element"
    let s = "F8=\(F8.isEmpty ? "off" : F8) F8VS=\(F8VS) FOLD=\(FOLDV.isEmpty ? "off" : FOLDV) KV8=\(KV8) K8V16=\(K8V16) QK8=\(QK8) TK=\(TK) Lmax=\(Lmax) T=\(T)"
        + " | prep=\(prep) qk=\(QKN) lut=\(I8T ? "lut_gemm32_i8t" + I8TV : "lut_gemm32_i8") softmax=\(sm) pv=\(pv) fill=\(fill) | V copy=\(vfmt) | mask=\(mask)"
        + " exp=\(f8 && F8 == "x" ? "half" : "float") P=\(f8 ? "e4m3" : "fp16") | kernels=\(kpath)+kernels_fold.metal+kernels_f8.metal+kernels_qk.metal (\(src.count) chars) | kernels_ok=\(ok)"
    let b = Array(s.utf8CString.prefix(Int(n) - 1)) + [0]
    for (i, c) in b.enumerated() { out[i] = c }
    return ok ? Int32(b.count - 1) : -1
}
