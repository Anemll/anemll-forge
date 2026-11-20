// Q.K variants alone (19 Nov 2026, M6): the served attn_qk8isk (INT8 Q x INT8 K read from the ANE cache layout
// [kv head][dim][key], per-key scale epilogue, fp16 scores) against an FP8 K copy (e4m3 Q x e4m3 K) and controls.
// One "call" = the served loop over the keys in TK = 2048 tiles for 4 KV heads x 768 rows (6 q heads x T = 128);
// GPU ms per call (command buffer GPU time, median), and the scores checked against a CPU reference.
//   qk_f8_bench N,N,... [REPS]
//   i8s      served: INT8, strided K (64 B rows), 64 x 64 per threadgroup, 4 simdgroups
//   i8s_ne   i8s without the epilogue (no score stores): the matmul + loads alone
//   i8d      INT8, K in a dense copy [kv][key tile][256][64] (layout effect alone)
//   f8d      FP8 e4m3 Q x e4m3 K copy, dense [kv][key tile][256][64], same epilogue (the requested prototype)
//   f8t      FP8, K copy [kv][key][256] (row-major keys), transpose_right
//   i8s_n128 / f8d_n128   64 rows x 128 keys per threadgroup, 8 simdgroups
//   i8s_m128 / f8d_m128   128 rows x 64 keys per threadgroup, 8 simdgroups
import Foundation
import Metal

let args = CommandLine.arguments
let NS = args[1].split(separator: ",").map { Int($0)! }
let REPS = args.count > 2 ? Int(args[2])! : 10
let TK = 2048, R = 3072, D = 256, LK = 98304, KH = 4
let src = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#define T 128
#define LK \(LK)
using TI = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;
using TF = tensor<device metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline>;
#define ARGS device uchar *Q [[buffer(0)]], device uchar *K [[buffer(1)]], device half *S [[buffer(2)]], \\
    constant uint2 &kst [[buffer(3)]], constant uint &TK [[buffer(4)]], device const float *QS [[buffer(5)]], \\
    device const half *KS [[buffer(6)]], constant uint &ssh [[buffer(7)]], constant uint &a0 [[buffer(8)]], \\
    device int *o [[buffer(9)]], uint3 tg [[threadgroup_position_in_grid]]

template <typename C>
inline void epi(thread C &cI, device half *S, uint r0, uint TK, uint k0, device const float *QS, device const half *ks) {
    for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
        if (!cI.is_valid_element(i)) continue;
        auto ix = cI.get_multidimensional_index(i);
        uint key = ix[0], row = ix[1];
        S[(r0 + row) * TK + k0 + key] = half(float(cI[i]) * QS[r0 + row] * float(ks[key]));
    }
}
// served attn_qk8isk (+ a0 instead of buffer offsets)
template <uint MR, uint NK, uint SG, bool EPI>
inline void qk_i8s_t(ARGS) {
    uint kt = tg.x, rt = tg.y, gh = tg.z, r0 = gh * 6 * T + rt * MR;
    TI tQ((device int8_t *)Q + r0 * 256, dextents<int32_t, 2>(256, MR));
    TI tK((device int8_t *)K + gh * kst.x + a0 + kt * NK, dextents<int32_t, 2>(NK, 256), array<int32_t, 2>{1, int(kst.y)});
    constexpr auto d = matmul2d_descriptor(MR, NK, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<SG>> op;
    auto cI = op.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
    op.run(tQ, tK, cI);
    if (EPI) { epi(cI, S, r0, TK, kt * NK, QS, KS + gh * ssh + a0 + kt * NK); }
    else { int s = 0; for (uint16_t i = 0; i < cI.get_capacity(); ++i) if (cI.is_valid_element(i)) s += cI[i]; if (s == 123456789) o[0] = s; }
}
kernel void qk_i8s(ARGS) { qk_i8s_t<64, 64, 4, true>(Q, K, S, kst, TK, QS, KS, ssh, a0, o, tg); }
kernel void qk_i8s_ne(ARGS) { qk_i8s_t<64, 64, 4, false>(Q, K, S, kst, TK, QS, KS, ssh, a0, o, tg); }
kernel void qk_i8s_n128(ARGS) { qk_i8s_t<64, 128, 8, true>(Q, K, S, kst, TK, QS, KS, ssh, a0, o, tg); }
kernel void qk_i8s_m128(ARGS) { qk_i8s_t<128, 64, 8, true>(Q, K, S, kst, TK, QS, KS, ssh, a0, o, tg); }
// dense K copy [kv][LK / NK][256][NK] (INT8 or e4m3 bytes)
template <typename TT, typename DT, uint MR, uint NK, uint SG>
inline void qk_dense_t(ARGS) {
    uint kt = tg.x, rt = tg.y, gh = tg.z, r0 = gh * 6 * T + rt * MR;
    TT tQ((typename TT::data_handle_type)(Q + r0 * 256), dextents<int32_t, 2>(256, MR));
    TT tK((typename TT::data_handle_type)(K + (ulong(gh) * (LK / NK) + (a0 / NK + kt)) * 256 * NK), dextents<int32_t, 2>(NK, 256));
    constexpr auto d = matmul2d_descriptor(MR, NK, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<SG>> op;
    auto cI = op.template get_destination_cooperative_tensor<TT, TT, DT>();
    op.run(tQ, tK, cI);
    epi(cI, S, r0, TK, kt * NK, QS, KS + gh * ssh + a0 + kt * NK);
}
kernel void qk_i8d(ARGS) { qk_dense_t<TI, int32_t, 64, 64, 4>(Q, K, S, kst, TK, QS, KS, ssh, a0, o, tg); }
kernel void qk_f8d(ARGS) { qk_dense_t<TF, float, 64, 64, 4>(Q, K, S, kst, TK, QS, KS, ssh, a0, o, tg); }
kernel void qk_f8d_n128(ARGS) { qk_dense_t<TF, float, 64, 128, 8>(Q, K, S, kst, TK, QS, KS, ssh, a0, o, tg); }
kernel void qk_f8d_m128(ARGS) { qk_dense_t<TF, float, 128, 64, 8>(Q, K, S, kst, TK, QS, KS, ssh, a0, o, tg); }
// K copy [kv][key][256], transpose_right
kernel void qk_f8t(ARGS) {
    uint kt = tg.x, rt = tg.y, gh = tg.z, r0 = gh * 6 * T + rt * 64;
    TF tQ((typename TF::data_handle_type)(Q + r0 * 256), dextents<int32_t, 2>(256, 64));
    TF tK((typename TF::data_handle_type)(K + (ulong(gh) * LK + a0 + kt * 64) * 256), dextents<int32_t, 2>(256, 64));
    constexpr auto d = matmul2d_descriptor(64, 64, 256, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    auto cI = op.template get_destination_cooperative_tensor<TF, TF, float>();
    op.run(tQ, tK, cI);
    epi(cI, S, r0, TK, kt * 64, QS, KS + gh * ssh + a0 + kt * 64);
}
// threadgroup-staged epilogue: the accumulators to threadgroup memory ([row][key]), then coalesced 4-wide scaled stores
template <typename TT, typename DT, uint MR, uint NK, uint SG, bool STRIDED, bool KSC>
inline void qk_tg_t(ARGS, uint tid, threadgroup DT *tile) {
    uint kt = tg.x, rt = tg.y, gh = tg.z, r0 = gh * 6 * T + rt * MR;
    TT tQ((typename TT::data_handle_type)(Q + r0 * 256), dextents<int32_t, 2>(256, MR));
    constexpr auto d = matmul2d_descriptor(MR, NK, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<SG>> op;
    auto cI = op.template get_destination_cooperative_tensor<TT, TT, DT>();
    if (STRIDED) { TT tK((typename TT::data_handle_type)(K + gh * kst.x + a0 + kt * NK), dextents<int32_t, 2>(NK, 256), array<int32_t, 2>{1, int(kst.y)}); op.run(tQ, tK, cI); }
    else { TT tK((typename TT::data_handle_type)(K + (ulong(gh) * (LK / NK) + (a0 / NK + kt)) * 256 * NK), dextents<int32_t, 2>(NK, 256)); op.run(tQ, tK, cI); }
    tensor<threadgroup DT, dextents<int32_t, 2>, tensor_inline> tT(tile, dextents<int32_t, 2>(NK, MR));
    cI.store(tT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const half *ks = KS + gh * ssh + a0 + kt * NK;
    constexpr uint NT_ = SG * 32, C4 = NK / 4;
    for (uint i = tid; i < MR * C4; i += NT_) {
        uint row = i / C4, c = (i % C4) * 4;
        float4 v = float4(*(threadgroup vec<DT, 4> *)(tile + row * NK + c)) * QS[r0 + row];
        if (KSC) v *= float4(*(device const half4 *)(ks + c));
        *(device half4 *)(S + (r0 + row) * TK + kt * NK + c) = half4(v);
    }
}
#define TGA uint tid [[thread_index_in_threadgroup]]
kernel void qk_i8s_tg(ARGS, TGA) { threadgroup int tile[64 * 64]; qk_tg_t<TI, int32_t, 64, 64, 4, true, true>(Q, K, S, kst, TK, QS, KS, ssh, a0, o, tg, tid, tile); }
kernel void qk_i8s_m128_tg(ARGS, TGA) { threadgroup int tile[128 * 64]; qk_tg_t<TI, int32_t, 128, 64, 8, true, true>(Q, K, S, kst, TK, QS, KS, ssh, a0, o, tg, tid, tile); }
kernel void qk_i8s_n128_tg(ARGS, TGA) { threadgroup int tile[64 * 128]; qk_tg_t<TI, int32_t, 64, 128, 8, true, true>(Q, K, S, kst, TK, QS, KS, ssh, a0, o, tg, tid, tile); }
kernel void qk_i8d_tg(ARGS, TGA) { threadgroup int tile[64 * 64]; qk_tg_t<TI, int32_t, 64, 64, 4, false, true>(Q, K, S, kst, TK, QS, KS, ssh, a0, o, tg, tid, tile); }
kernel void qk_f8d_tg(ARGS, TGA) { threadgroup float tile[64 * 64]; qk_tg_t<TF, float, 64, 64, 4, false, true>(Q, K, S, kst, TK, QS, KS, ssh, a0, o, tg, tid, tile); }
kernel void qk_f8g_tg(ARGS, TGA) { threadgroup float tile[64 * 64]; qk_tg_t<TF, float, 64, 64, 4, false, false>(Q, K, S, kst, TK, QS, KS, ssh, a0, o, tg, tid, tile); }
kernel void qk_f8g_m128_tg(ARGS, TGA) { threadgroup float tile[128 * 64]; qk_tg_t<TF, float, 128, 64, 8, false, false>(Q, K, S, kst, TK, QS, KS, ssh, a0, o, tg, tid, tile); }
// FP8 K with one global scale (no per-key scale load in the epilogue)
kernel void qk_f8g(ARGS) {
    uint kt = tg.x, rt = tg.y, gh = tg.z, r0 = gh * 6 * T + rt * 64;
    TF tQ((typename TF::data_handle_type)(Q + r0 * 256), dextents<int32_t, 2>(256, 64));
    TF tK((typename TF::data_handle_type)(K + (ulong(gh) * (LK / 64) + (a0 / 64 + kt)) * 256 * 64), dextents<int32_t, 2>(64, 256));
    constexpr auto d = matmul2d_descriptor(64, 64, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    auto cI = op.template get_destination_cooperative_tensor<TF, TF, float>();
    op.run(tQ, tK, cI);
    for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
        if (!cI.is_valid_element(i)) continue;
        auto ix = cI.get_multidimensional_index(i);
        S[(r0 + ix[1]) * TK + kt * 64 + ix[0]] = half(cI[i] * QS[r0 + ix[1]]);
    }
}
"""
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: opts) } catch { print("COMPILE FAILED: \(error)".prefix(3000)); exit(1) }
func buf(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: n, options: .storageModeShared)! }

// ---- data: INT8 codes, the same values as e4m3 where exact; seeded ----
var rng = SystemRandomNumberGenerator()
struct LCG { var s: UInt64; mutating func next() -> UInt64 { s = s &* 6364136223846793005 &+ 1442695040888963407; return s >> 33 } }
var g = LCG(s: 12345)
// e4m3 encode/decode (CPU)
func e4m3(_ x: Float) -> UInt8 {
    let s: UInt8 = x < 0 ? 0x80 : 0; let a = min(abs(x), 448)
    if a < 0.015625 { return s | UInt8(min(7, (a * 512).rounded(.toNearestOrEven))) }
    var b = a.bitPattern; b = b &+ 0x7FFFF &+ ((b >> 20) & 1)
    let r = (((b >> 23) - 120) << 3) | ((b >> 20) & 7); return s | UInt8(min(r, 0x7E))
}
func de4m3(_ c: UInt8) -> Float {
    let s: Float = c & 0x80 != 0 ? -1 : 1, e = Int((c >> 3) & 0xF), m = Float(c & 7)
    return e == 0 ? s * m / 512 : s * (1 + m / 8) * Float(pow(2.0, Double(e - 7)))
}
let Q8 = buf(R * D), QF = buf(R * D), QS = buf(R * 4)
let K8 = buf(KH * D * LK), KF64 = buf(KH * D * LK), KF128 = buf(KH * D * LK), KFT = buf(KH * D * LK), KI64 = buf(KH * D * LK), KS = buf(KH * LK * 2)
let S = buf(R * TK * 2), O = buf(16)
let q8 = Q8.contents().bindMemory(to: Int8.self, capacity: R * D), qf = QF.contents().bindMemory(to: UInt8.self, capacity: R * D)
let qs = QS.contents().bindMemory(to: Float.self, capacity: R)
for i in 0..<(R * D) { let v = Int8(Int(g.next() % 255) - 127); q8[i] = v; qf[i] = e4m3(Float(v)) }
for i in 0..<R { qs[i] = 0.001 + Float(g.next() % 1000) * 1e-6 }
let k8 = K8.contents().bindMemory(to: Int8.self, capacity: KH * D * LK)
let kf64 = KF64.contents().bindMemory(to: UInt8.self, capacity: KH * D * LK), kf128 = KF128.contents().bindMemory(to: UInt8.self, capacity: KH * D * LK)
let kft = KFT.contents().bindMemory(to: UInt8.self, capacity: KH * D * LK), ki64 = KI64.contents().bindMemory(to: Int8.self, capacity: KH * D * LK)
let ks = KS.contents().bindMemory(to: Float16.self, capacity: KH * LK)
for h in 0..<KH { for key in 0..<LK { ks[h * LK + key] = Float16(0.004 + Float(g.next() % 1000) * 1.6e-5) } }
var e8 = [UInt8](repeating: 0, count: 256); for c in 0..<256 { e8[c] = e4m3(Float(Int8(bitPattern: UInt8(c)))) }
for h in 0..<KH {
    for d in 0..<D {
        for key in 0..<LK {
            let v = Int8(Int(g.next() % 255) - 127)
            k8[(h * D + d) * LK + key] = v                                           // ANE layout [kv][dim][key]
            let c = e8[Int(UInt8(bitPattern: v))]
            ki64[((h * (LK / 64) + key / 64) * 256 + d) * 64 + key % 64] = v
            kf64[((h * (LK / 64) + key / 64) * 256 + d) * 64 + key % 64] = c
            kf128[((h * (LK / 128) + key / 128) * 256 + d) * 128 + key % 128] = c
            kft[(h * LK + key) * 256 + d] = c
        }
    }
}
// (variant, K buffer, Q buffer, keys per threadgroup, rows per threadgroup, threads, fp8)
let V: [(String, MTLBuffer, MTLBuffer, Int, Int, Int, Bool)] = [
    ("qk_i8s", K8, Q8, 64, 64, 128, false), ("qk_i8s_ne", K8, Q8, 64, 64, 128, false), ("qk_i8d", KI64, Q8, 64, 64, 128, false),
    ("qk_f8d", KF64, QF, 64, 64, 128, true), ("qk_f8t", KFT, QF, 64, 64, 128, true),
    ("qk_i8s_n128", K8, Q8, 128, 64, 256, false), ("qk_f8d_n128", KF128, QF, 128, 64, 256, true),
    ("qk_i8s_m128", K8, Q8, 64, 128, 256, false), ("qk_f8d_m128", KF64, QF, 64, 128, 256, true),
    ("qk_f8g", KF64, QF, 64, 64, 128, true),
    ("qk_i8s_tg", K8, Q8, 64, 64, 128, false), ("qk_i8s_m128_tg", K8, Q8, 64, 128, 256, false), ("qk_i8s_n128_tg", K8, Q8, 128, 64, 256, false),
    ("qk_i8d_tg", KI64, Q8, 64, 64, 128, false), ("qk_f8d_tg", KF64, QF, 64, 64, 128, true), ("qk_f8g_tg", KF64, QF, 64, 64, 128, true),
    ("qk_f8g_m128_tg", KF64, QF, 64, 128, 256, true)]
func call(_ pso: MTLComputePipelineState, _ v: (String, MTLBuffer, MTLBuffer, Int, Int, Int, Bool), _ n: Int, until: Int? = nil) -> Double {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso)
    e.setBuffer(v.2, offset: 0, index: 0); e.setBuffer(v.1, offset: 0, index: 1); e.setBuffer(S, offset: 0, index: 2)
    var kst = SIMD2<UInt32>(UInt32(D * LK), UInt32(LK)); e.setBytes(&kst, length: 8, index: 3)
    var tk = UInt32(TK); e.setBytes(&tk, length: 4, index: 4)
    e.setBuffer(QS, offset: 0, index: 5); e.setBuffer(KS, offset: 0, index: 6)
    var ssh = UInt32(LK); e.setBytes(&ssh, length: 4, index: 7); e.setBuffer(O, offset: 0, index: 9)
    var a0 = 0
    while a0 < (until ?? n) {
        let m = min(TK, n - a0)
        var a = UInt32(a0); e.setBytes(&a, length: 4, index: 8)
        e.dispatchThreadgroups(MTLSize(width: m / v.3, height: 6 * 128 / v.4, depth: 4), threadsPerThreadgroup: MTLSize(width: v.5, height: 1, depth: 1))
        a0 += m
    }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if let err = cb.error { print("GPU error \(err)"); exit(1) }
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}
// correctness: the first tile (a0 = 0), sampled scores vs CPU (exact for INT8 codes; e4m3(code) for FP8)
let sp = S.contents().bindMemory(to: Float16.self, capacity: R * TK)
for v in V where v.0 != "qk_i8s_ne" {
    let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: v.0)!)
    memset(S.contents(), 0, S.length)
    _ = call(pso, v, TK, until: 1)
    var maxrel: Float = 0
    for _ in 0..<2000 {
        let row = Int(g.next() % UInt64(R)), key = Int(g.next() % UInt64(TK)), h = row / 768
        var acc: Float = 0
        for d in 0..<D {
            let qv = Float(q8[row * D + d]), kv = Float(k8[(h * D + d) * LK + key])
            acc += v.6 ? de4m3(e4m3(qv)) * de4m3(e4m3(kv)) : qv * kv
        }
        let ref = acc * qs[row] * (v.0.hasPrefix("qk_f8g") ? 1 : Float(ks[h * LK + key])), got = Float(sp[row * TK + key])
        maxrel = max(maxrel, abs(got - ref) / max(abs(ref), 0.05))
    }
    print(String(format: "check %-12@ max |err| / max(|ref|, 0.05) = %.2e  (threads %d)", v.0, maxrel, pso.maxTotalThreadsPerThreadgroup))
}
var out: [[String: Any]] = []
do {   // warm the GPU clock (2 s of back-to-back calls) before any timed call
    let w = V[0], pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: w.0)!)
    let t0 = Date(); while Date().timeIntervalSince(t0) < 2 { _ = call(pso, w, 32768) }
}
for n in NS {
    for v in V {
        let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: v.0)!)
        _ = call(pso, v, n); _ = call(pso, v, n)
        var ms: [Double] = []
        for _ in 0..<REPS { ms.append(call(pso, v, n)) }
        ms.sort()
        let med = ms[ms.count / 2], tops = 2.0 * Double(R) * Double(n) * Double(D) / (med * 1e-3) / 1e12
        print(String(format: "keys %6d  %-12@ %8.3f ms  (min %.3f)  %5.1f TOPS", n, v.0, med, ms[0], tops))
        out.append(["keys": n, "variant": v.0, "ms_median": med, "ms_min": ms[0], "tops": tops])
    }
}
if let p = ProcessInfo.processInfo.environment["OUT"] {
    try! JSONSerialization.data(withJSONObject: out, options: .prettyPrinted).write(to: URL(fileURLWithPath: p))
}
