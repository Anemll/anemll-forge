// Flash-decoding attention for the decode verify, variant sweep (16 Nov 2026), standalone, synthetic data in the ANE
// cache layout (see decode_attn_flash.swift): 16 attention layers, 4 KV heads x 48 query vectors (padded to 64), head
// dim 256, INT8 K dim-major + per-key scale, INT8 V + per-row scale (or the fp16 V copy). One threadgroup = (KV head,
// chunk of CK keys, 256 / DSPLIT output dims), NSG simdgroups, walking SUB-key sub-tiles with an online softmax: S =
// Q8 K8^T (int8 matmul2d, cooperative) -> half in threadgroup memory (x QS x KS), P = exp(s - m) (x VS) in place,
// O += P V (matmul2d into a cooperative accumulator). A combine merges the chunks. Variants are template
// instantiations: dec_<v>_<NSG>_<DSPLIT>_<SUB> (v = i8 / f16 values). Each is checked against a CPU reference.
//   swiftc -O -o decode_attn_flash3 decode_attn_flash3.swift -framework Metal && ./decode_attn_flash3 [N ...]
//   env: VARIANTS (comma list, e.g. i8_8_2_64), CK, REPS, NLAYERS
import Foundation
import Metal

let env = ProcessInfo.processInfo.environment
let CK = Int(env["CK"] ?? "1024")!, NLB = Int(env["NLAYERS"] ?? "4")!, REPS = Int(env["REPS"] ?? "5")!
let SMAX = 98304
let NS: [Int] = CommandLine.arguments.count > 1 ? CommandLine.arguments.dropFirst().map { Int($0)! } : [8192, 32768, 65536, 98304]
let ALL = ["i8_4_1_64", "i8_8_1_64", "i8_4_2_64", "i8_8_2_64", "i8_4_4_64", "i8_8_4_64", "i8_4_2_128", "i8_8_2_128", "i8_8_4_128",
           "f16_8_2_64", "f16_8_4_128"]
let VARIANTS = (env["VARIANTS"].map { $0.split(separator: ",").map(String.init) }) ?? ALL
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
var src = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct Prm { uint kh, kd, sh, vh, vr, n, ck, nc; };

template <typename VT, bool VSC, int NSG, int DSPLIT, int SUB>
inline void body(device const int8_t *Q8, device const float *QS, device const int8_t *K8, device const half *KS,
                 device const VT *V, device const half *VS, device float *OP, device float *MP, device float *LP,
                 constant Prm &p, uint c, uint dt, uint gh, uint t,
                 threadgroup half *S, threadgroup float *alpha, threadgroup float *qs, threadgroup float *ks, threadgroup float *vs) {
    constexpr int NT = 32 * NSG, TPR = NT / 64, KPT = SUB / TPR, DN = 256 / DSPLIT;
    uint slot = (gh * p.nc + c) * DSPLIT + dt;
    uint k0 = c * p.ck, k1 = min(k0 + p.ck, p.n);
    if (t < 64) qs[t] = QS[gh * 64 + t];
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tQ((device int8_t *)Q8 + gh * 64 * 256, dextents<int32_t, 2>(256, 64));
    constexpr auto dqk = matmul2d_descriptor(64, SUB, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<dqk, execution_simdgroups<NSG>> qk;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(S, dextents<int32_t, 2>(SUB, 64), array<int32_t, 2>{1, SUB});
    constexpr auto dpv = matmul2d_descriptor(64, DN, SUB, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dpv, execution_simdgroups<NSG>> pv;
    tensor<device VT, dextents<int32_t, 2>, tensor_inline> tV0((device VT *)V + gh * p.vh + dt * DN, dextents<int32_t, 2>(DN, SUB),
                                                               array<int32_t, 2>{1, int(p.vr)});
    auto cO = pv.template get_destination_cooperative_tensor<decltype(tP), decltype(tV0), float>();
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0.0f;
    uint row = t / TPR, part = t % TPR;
    float m = -INFINITY, l = 0.0f;
    for (uint kb = k0; kb < k1; kb += SUB) {
        for (uint j = t; j < SUB; j += NT) { ks[j] = float(KS[gh * p.sh + kb + j]); vs[j] = float(VS[gh * p.sh + kb + j]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK((device int8_t *)K8 + gh * p.kh + kb, dextents<int32_t, 2>(SUB, 256),
                                                                      array<int32_t, 2>{1, int(p.kd)});
        auto cI = qk.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
        qk.run(tQ, tK, cI);
        for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
            if (!cI.is_valid_element(i)) continue;
            auto ix = cI.get_multidimensional_index(i);
            S[ix[1] * SUB + ix[0]] = half(float(cI[i]) * qs[ix[1]] * ks[ix[0]]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half *Sr = S + row * SUB + part * KPT;
        float tm = -INFINITY;
        for (int j = 0; j < KPT; ++j) tm = max(tm, float(Sr[j]));
        for (int o = 1; o < TPR; o <<= 1) tm = max(tm, simd_shuffle_xor(tm, o));
        float mn = max(m, tm), a = exp(m - mn), sum = 0.0f;
        for (int j = 0; j < KPT; ++j) {
            float e = exp(float(Sr[j]) - mn); sum += e;
            Sr[j] = half(VSC ? e * vs[part * KPT + j] : e);
        }
        for (int o = 1; o < TPR; o <<= 1) sum += simd_shuffle_xor(sum, o);
        l = l * a + sum; m = mn;
        if (part == 0) alpha[row] = a;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
            if (!cO.is_valid_element(i)) continue;
            auto ix = cO.get_multidimensional_index(i);
            cO[i] *= alpha[ix[1]];
        }
        tensor<device VT, dextents<int32_t, 2>, tensor_inline> tV((device VT *)V + gh * p.vh + kb * p.vr + dt * DN, dextents<int32_t, 2>(DN, SUB),
                                                                  array<int32_t, 2>{1, int(p.vr)});
        pv.run(tP, tV, cO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        OP[(slot * 64 + ix[1]) * DN + ix[0]] = cO[i];
    }
    if (part == 0) { MP[slot * 64 + row] = m; LP[slot * 64 + row] = l; }
}
#define DEC(NAME, VT, VSC, NSG, DSPLIT, SUB) \\
kernel void NAME(device const int8_t *Q8 [[buffer(0)]], device const float *QS [[buffer(1)]], device const int8_t *K8 [[buffer(2)]], \\
                 device const half *KS [[buffer(3)]], device const VT *V [[buffer(4)]], device const half *VS [[buffer(5)]], \\
                 device float *OP [[buffer(8)]], device float *MP [[buffer(9)]], device float *LP [[buffer(10)]], \\
                 constant Prm &p [[buffer(11)]], uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) { \\
    threadgroup half S[64 * SUB]; threadgroup float alpha[64], qs[64], ks[SUB], vs[SUB]; \\
    body<VT, VSC, NSG, DSPLIT, SUB>(Q8, QS, K8, KS, V, VS, OP, MP, LP, p, tg.x, tg.y, tg.z, t, S, alpha, qs, ks, vs); }

"""
for v in Set(VARIANTS + ["i8_4_1_64"]) {
    let f = v.split(separator: "_"); let vt = f[0] == "i8" ? "int8_t" : "half"
    src += "DEC(dec_\(v), \(vt), \(f[0] == "i8" ? "true" : "false"), \(f[1]), \(f[2]), \(f[3]))\n"
}
src += """
kernel void dec_combine(device const float *OP [[buffer(0)]], device const float *MP [[buffer(1)]], device const float *LP [[buffer(2)]],
                        device float *O [[buffer(3)]], constant Prm &p [[buffer(4)]], constant uint &ds [[buffer(5)]],
                        uint3 g [[thread_position_in_grid]]) {
    uint d = g.x, row = g.y, gh = g.z, dn = 256 / ds, dt = d / dn, dd = d % dn;
    float M = -INFINITY;
    for (uint c = 0; c < p.nc; ++c) M = max(M, MP[((gh * p.nc + c) * ds + dt) * 64 + row]);
    float num = 0.0f, den = 0.0f;
    for (uint c = 0; c < p.nc; ++c) {
        uint s = (gh * p.nc + c) * ds + dt;
        float w = exp(MP[s * 64 + row] - M);
        num += w * OP[(s * 64 + row) * dn + dd]; den += w * LP[s * 64 + row];
    }
    O[(gh * 48 + row) * 256 + d] = num / den;
}
"""
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: opts) } catch { print("compile error: \(error)"); exit(1) }
var psos: [String: MTLComputePipelineState] = [:]
for v in VARIANTS {
    do { psos[v] = try dev.makeComputePipelineState(function: lib.makeFunction(name: "dec_" + v)!) }
    catch { print("variant \(v): pipeline error \(error)") }
}
let pC = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "dec_combine")!)
func buf(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: max(n, 16), options: .storageModeShared)! }
struct Prm { var kh, kd, sh, vh, vr, n, ck, nc: UInt32 }
var rng: UInt64 = 0x9E3779B97F4A7C15
func rnd() -> UInt32 { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return UInt32(truncatingIfNeeded: rng >> 16) }
func fillH(_ b: MTLBuffer, _ lo: Float, _ hi: Float) {
    let n = b.length / 2, p = b.contents().bindMemory(to: Float16.self, capacity: n)
    for i in 0..<n { p[i] = Float16(lo + (hi - lo) * Float(rnd() >> 8) / 16777216) }
}
let Q8 = buf(4 * 64 * 256), QS = buf(4 * 64 * 4)
do {
    let p = Q8.contents().bindMemory(to: Int8.self, capacity: 4 * 64 * 256)
    for h in 0..<4 { for r in 0..<64 { for d in 0..<256 { p[(h * 64 + r) * 256 + d] = r < 48 ? Int8(truncatingIfNeeded: Int(rnd() % 255) - 127) : 0 } } }
    let s = QS.contents().bindMemory(to: Float.self, capacity: 256); for i in 0..<256 { s[i] = 0.003 }
}
let NCMAX = (SMAX + CK - 1) / CK
let OP = buf(4 * NCMAX * 4 * 64 * 256 * 4), MP = buf(4 * NCMAX * 4 * 64 * 4), LP = buf(4 * NCMAX * 4 * 64 * 4), O = buf(4 * 48 * 256 * 4)
struct Lay { let k, ks, v, vs, vc: MTLBuffer }
func makeLay(_ S: Int) -> Lay {
    let l = Lay(k: buf(4 * 256 * S), ks: buf(4 * S * 2), v: buf(4 * S * 256), vs: buf(4 * S * 2), vc: buf(4 * S * 256 * 2))
    arc4random_buf(l.k.contents(), l.k.length); arc4random_buf(l.v.contents(), l.v.length); fillH(l.ks, 0.002, 0.004); fillH(l.vs, 0.005, 0.02)
    let vc = l.vc.contents().bindMemory(to: Float16.self, capacity: 4 * S * 256), v8 = l.v.contents().bindMemory(to: Int8.self, capacity: 4 * S * 256)
    let vs = l.vs.contents().bindMemory(to: Float16.self, capacity: 4 * S)
    for h in 0..<4 { for r in 0..<S { let sc = Float(vs[h * S + r]); for d in 0..<256 { let i = (h * S + r) * 256 + d; vc[i] = Float16(Float(v8[i]) * sc) } } }
    return l
}
func encode(_ e: MTLComputeCommandEncoder, _ v: String, _ l: Lay, _ S: Int, _ n: Int) {
    let f = v.split(separator: "_"), nsg = Int(f[1])!, ds = Int(f[2])!, f16 = f[0] == "f16"
    let nc = (n + CK - 1) / CK
    var prm = Prm(kh: UInt32(256 * S), kd: UInt32(S), sh: UInt32(S), vh: UInt32(S * 256), vr: 256, n: UInt32(n), ck: UInt32(CK), nc: UInt32(nc))
    e.setComputePipelineState(psos[v]!)
    for (i, b) in [Q8, QS, l.k, l.ks, f16 ? l.vc : l.v, l.vs].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    for (i, b) in [OP, MP, LP].enumerated() { e.setBuffer(b, offset: 0, index: 8 + i) }
    e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 11)
    e.dispatchThreadgroups(MTLSize(width: nc, height: ds, depth: 4), threadsPerThreadgroup: MTLSize(width: 32 * nsg, height: 1, depth: 1))
    e.memoryBarrier(scope: .buffers)
    var dsu = UInt32(ds)
    e.setComputePipelineState(pC)
    for (i, b) in [OP, MP, LP, O].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 4); e.setBytes(&dsu, length: 4, index: 5)
    e.dispatchThreads(MTLSize(width: 256, height: 48, depth: 4), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.memoryBarrier(scope: .buffers)
}
// correctness, N = 4096
let small = makeLay(4096)
var refO = [Float](repeating: 0, count: 4 * 48 * 256)
do {
    let S = 4096, n = 4096
    let q8 = Q8.contents().bindMemory(to: Int8.self, capacity: 4 * 64 * 256), k8 = small.k.contents().bindMemory(to: Int8.self, capacity: 4 * 256 * S)
    let ks = small.ks.contents().bindMemory(to: Float16.self, capacity: 4 * S), v8 = small.v.contents().bindMemory(to: Int8.self, capacity: 4 * S * 256)
    let vs = small.vs.contents().bindMemory(to: Float16.self, capacity: 4 * S)
    for h in 0..<4 { for r in stride(from: 0, to: 48, by: 5) {
        var s = [Float](repeating: 0, count: n)
        for k in 0..<n { var acc: Int32 = 0; for d in 0..<256 { acc += Int32(q8[(h * 64 + r) * 256 + d]) * Int32(k8[h * 256 * S + d * S + k]) }; s[k] = Float(acc) * 0.003 * Float(ks[h * S + k]) }
        let mx = s.max()!; var den: Float = 0; var o = [Float](repeating: 0, count: 256)
        for k in 0..<n { let e = exp(s[k] - mx); den += e; let w = e * Float(vs[h * S + k]); for d in 0..<256 { o[d] += w * Float(v8[(h * S + k) * 256 + d]) } }
        for d in 0..<256 { refO[(h * 48 + r) * 256 + d] = o[d] / den }
    } }
}
var ok: [String] = []
for v in VARIANTS where psos[v] != nil {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    encode(e, v, small, 4096, 4096); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if let err = cb.error { print("variant \(v): run error \(err)"); continue }
    let og = O.contents().bindMemory(to: Float.self, capacity: 4 * 48 * 256)
    var me: Float = 0, mr: Float = 0
    for h in 0..<4 { for r in stride(from: 0, to: 48, by: 5) { for d in 0..<256 { let i = (h * 48 + r) * 256 + d; mr = max(mr, abs(refO[i])); me = max(me, abs(refO[i] - og[i])) } } }
    let rel = me / mr
    print(String(format: "check %-12@ max error %.2f%% %@", v, 100 * rel, rel < 0.01 ? "OK" : "WRONG"))
    if rel < 0.01 { ok.append(v) }
}
let lays = (0..<NLB).map { _ in makeLay(SMAX) }
func ms(_ v: String, _ n: Int) -> Double {
    var ts: [Double] = []
    for _ in 0..<REPS {
        let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        for li in 0..<16 { encode(e, v, lays[li % NLB], SMAX, n) }
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted(); precondition(cb.error == nil, "\(cb.error!)")
        ts.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
    }
    return ts.sorted()[ts.count / 2]
}
for v in ok {
    _ = ms(v, 8192)
    var line = String(format: "%-12@ CK %4d:", v, CK)
    for n in NS { let t = ms(v, n); line += String(format: "  %dK %6.1f ms (%.3f/1K)", n / 1024, t, t / Double(n) * 1024) }
    print(line)
}
