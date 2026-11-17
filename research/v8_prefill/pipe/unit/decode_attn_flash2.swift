// Flash-decoding attention on the GPU for the decode verify (16 Nov 2026), standalone: the M6 target's shapes (16
// attention layers, 4 KV heads with 48 query vectors each = 8 verify rows x 6 query heads, padded to 64; head dim
// 256), the ANE cache layout (INT8 keys dim-major [kvh][256][S] + a per-key scale, INT8 values [kvh][S][256] + a
// per-row scale; or the K8V16 fp16 value copy [kvh][S][256]), the actual key count N. One threadgroup = one KV head x
// one chunk of CK keys, walking 64-key sub-tiles: S = Q8 K8^T (int8 matmul2d) x QS x KS -> online softmax (P = exp(s -
// m) x value scale) -> O += P V (matmul2d into a cooperative accumulator); a combine merges the chunks. Checked
// against a CPU reference, then timed for N = 8K .. 96K (16 layers per command buffer).
//   swiftc -O -o decode_attn_flash decode_attn_flash.swift -framework Metal && ./decode_attn_flash
import Foundation
import Metal

let env = ProcessInfo.processInfo.environment
let STAGE = Int(env["STAGE"] ?? "3")!   // timing ablation: 1 = QK only, 2 = + softmax, 3 = full
let CK = Int(env["CK"] ?? "1024")!, NLB = Int(env["NLAYERS"] ?? "4")!, REPS = Int(env["REPS"] ?? "7")!
let SMAX = 98304
let NS: [Int] = CommandLine.arguments.count > 1 ? CommandLine.arguments.dropFirst().map { Int($0)! }
    : [8192, 16384, 32768, 49152, 65536, 81920, 98304]
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct Prm { uint kh, kd, sh, vh, vr, n, ck, nc, stage; };

template <typename VT, bool VSC>
inline void dec_attn_body(device const int8_t *Q8, device const float *QS, device const int8_t *K8, device const half *KS,
                          device const VT *V, device const half *VS, device float *SC, device half *PS,
                          device float *OP, device float *MP, device float *LP, constant Prm &p,
                          uint c, uint gh, uint t, threadgroup float *alpha, threadgroup float *S, threadgroup half *P) {
    // online softmax over 64-key sub-tiles, scores and probabilities on chip (threadgroup memory), O in registers
    uint slot = gh * p.nc + c;
    uint k0 = c * p.ck, k1 = min(k0 + p.ck, p.n);
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tQ((device int8_t *)Q8 + gh * 64 * 256, dextents<int32_t, 2>(256, 64));
    constexpr auto dqk = matmul2d_descriptor(64, 64, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<dqk, execution_simdgroups<4>> qk;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(P, dextents<int32_t, 2>(64, 64), array<int32_t, 2>{1, 64});
    constexpr auto dpv = matmul2d_descriptor(64, 256, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dpv, execution_simdgroups<4>> pv;
    tensor<device VT, dextents<int32_t, 2>, tensor_inline> tV0((device VT *)V + gh * p.vh, dextents<int32_t, 2>(256, 64), array<int32_t, 2>{1, int(p.vr)});
    auto cO = pv.template get_destination_cooperative_tensor<decltype(tP), decltype(tV0), float>();
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0.0f;
    uint row = t >> 1, hf = t & 1;
    float m = -INFINITY, l = 0.0f, qs = QS[gh * 64 + row];
    for (uint kb = k0; kb < k1; kb += 64) {
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK((device int8_t *)K8 + gh * p.kh + kb, dextents<int32_t, 2>(64, 256),
                                                                      array<int32_t, 2>{1, int(p.kd)});
        auto cI = qk.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
        qk.run(tQ, tK, cI);
        for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
            if (!cI.is_valid_element(i)) continue;
            auto ix = cI.get_multidimensional_index(i);
            S[ix[1] * 64 + ix[0]] = float(cI[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const half *ks = KS + gh * p.sh + kb + hf * 32;
        device const half *vs = VS + gh * p.sh + kb + hf * 32;
        threadgroup float *Sr = S + row * 64 + hf * 32;
        float sc[32], tm = -INFINITY;
        for (uint j = 0; j < 32; ++j) { sc[j] = Sr[j] * qs * float(ks[j]); tm = max(tm, sc[j]); }
        tm = max(tm, simd_shuffle_xor(tm, 1));
        float mn = max(m, tm), a = exp(m - mn), sum = 0.0f;
        threadgroup half *Pr = P + row * 64 + hf * 32;
        for (uint j = 0; j < 32; ++j) { float e = exp(sc[j] - mn); sum += e; Pr[j] = half(VSC ? e * float(vs[j]) : e); }
        sum += simd_shuffle_xor(sum, 1);
        l = l * a + sum; m = mn;
        if (hf == 0) alpha[row] = a;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
            if (!cO.is_valid_element(i)) continue;
            auto ix = cO.get_multidimensional_index(i);
            cO[i] *= alpha[ix[1]];
        }
        tensor<device VT, dextents<int32_t, 2>, tensor_inline> tV((device VT *)V + gh * p.vh + kb * p.vr, dextents<int32_t, 2>(256, 64),
                                                                  array<int32_t, 2>{1, int(p.vr)});
        pv.run(tP, tV, cO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        OP[(slot * 64 + ix[1]) * 256 + ix[0]] = cO[i];
    }
    if (hf == 0) { MP[slot * 64 + row] = m; LP[slot * 64 + row] = l; }
}
kernel void dec_attn8(device const int8_t *Q8 [[buffer(0)]], device const float *QS [[buffer(1)]], device const int8_t *K8 [[buffer(2)]],
                      device const half *KS [[buffer(3)]], device const int8_t *V [[buffer(4)]], device const half *VS [[buffer(5)]],
                      device float *SC [[buffer(6)]], device half *PS [[buffer(7)]], device float *OP [[buffer(8)]],
                      device float *MP [[buffer(9)]], device float *LP [[buffer(10)]], constant Prm &p [[buffer(11)]],
                      uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    threadgroup float alpha[64]; threadgroup float S[64 * 64]; threadgroup half P[64 * 64];
    dec_attn_body<int8_t, true>(Q8, QS, K8, KS, V, VS, SC, PS, OP, MP, LP, p, tg.x, tg.z, t, alpha, S, P);
}
kernel void dec_attn16(device const int8_t *Q8 [[buffer(0)]], device const float *QS [[buffer(1)]], device const int8_t *K8 [[buffer(2)]],
                       device const half *KS [[buffer(3)]], device const half *V [[buffer(4)]], device const half *VS [[buffer(5)]],
                       device float *SC [[buffer(6)]], device half *PS [[buffer(7)]], device float *OP [[buffer(8)]],
                       device float *MP [[buffer(9)]], device float *LP [[buffer(10)]], constant Prm &p [[buffer(11)]],
                       uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    threadgroup float alpha[64]; threadgroup float S[64 * 64]; threadgroup half P[64 * 64];
    dec_attn_body<half, false>(Q8, QS, K8, KS, V, VS, SC, PS, OP, MP, LP, p, tg.x, tg.z, t, alpha, S, P);
}
// merge the chunks: O[gh][row][d] = sum_c exp(m_c - M) O_c / sum_c exp(m_c - M) l_c (48 real rows)
kernel void dec_combine(device const float *OP [[buffer(0)]], device const float *MP [[buffer(1)]], device const float *LP [[buffer(2)]],
                        device float *O [[buffer(3)]], constant Prm &p [[buffer(4)]], uint3 g [[thread_position_in_grid]]) {
    uint d = g.x, row = g.y, gh = g.z;
    float M = -INFINITY;
    for (uint c = 0; c < p.nc; ++c) M = max(M, MP[(gh * p.nc + c) * 64 + row]);
    float num = 0.0f, den = 0.0f;
    for (uint c = 0; c < p.nc; ++c) {
        uint s = gh * p.nc + c;
        float w = exp(MP[s * 64 + row] - M);
        num += w * OP[(s * 64 + row) * 256 + d]; den += w * LP[s * 64 + row];
    }
    O[(gh * 48 + row) * 256 + d] = num / den;
}
"""
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
let lib = try! dev.makeLibrary(source: src, options: opts)
func pso(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let pA8 = pso("dec_attn8"), pA16 = pso("dec_attn16"), pC = pso("dec_combine")
func buf(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: max(n, 16), options: .storageModeShared)! }
struct Prm { var kh, kd, sh, vh, vr, n, ck, nc, stage: UInt32 }

var rng: UInt64 = 0x9E3779B97F4A7C15
func rnd() -> UInt32 { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return UInt32(truncatingIfNeeded: rng >> 16) }
func fillI8(_ b: MTLBuffer, count: Int? = nil) { arc4random_buf(b.contents(), count ?? b.length) }
func fillH(_ b: MTLBuffer, _ lo: Float, _ hi: Float) {
    let n = b.length / 2, p = b.contents().bindMemory(to: Float16.self, capacity: n)
    for i in 0..<n { p[i] = Float16(lo + (hi - lo) * Float(rnd() >> 8) / 16777216) }
}
// shared per-call buffers
let Q8 = buf(4 * 64 * 256), QS = buf(4 * 64 * 4)
do {   // 48 real query rows per head, 16 zero rows
    let p = Q8.contents().bindMemory(to: Int8.self, capacity: 4 * 64 * 256)
    for h in 0..<4 { for r in 0..<64 { for d in 0..<256 { p[(h * 64 + r) * 256 + d] = r < 48 ? Int8(truncatingIfNeeded: Int(rnd() % 255) - 127) : 0 } } }
    let s = QS.contents().bindMemory(to: Float.self, capacity: 256); for i in 0..<256 { s[i] = 0.003 }
}
let NCMAX = (SMAX + CK - 1) / CK
let SC = buf(16), PS = buf(16)
let OP = buf(4 * NCMAX * 64 * 256 * 4), MP = buf(4 * NCMAX * 64 * 4), LP = buf(4 * NCMAX * 64 * 4), O = buf(4 * 48 * 256 * 4)
struct Lay { let k, ks, v, vs, vc: MTLBuffer }
func makeLay(_ S: Int) -> Lay {
    let l = Lay(k: buf(4 * 256 * S), ks: buf(4 * S * 2), v: buf(4 * S * 256), vs: buf(4 * S * 2), vc: buf(4 * S * 256 * 2))
    fillI8(l.k); fillI8(l.v); fillH(l.ks, 0.002, 0.004); fillH(l.vs, 0.005, 0.02)
    let vc = l.vc.contents().bindMemory(to: Float16.self, capacity: 4 * S * 256), v8 = l.v.contents().bindMemory(to: Int8.self, capacity: 4 * S * 256)
    let vs = l.vs.contents().bindMemory(to: Float16.self, capacity: 4 * S)
    for h in 0..<4 { for r in 0..<S { let sc = Float(vs[h * S + r]); for d in 0..<256 { let i = (h * S + r) * 256 + d; vc[i] = Float16(Float(v8[i]) * sc) } } }
    return l
}
func encode(_ e: MTLComputeCommandEncoder, _ l: Lay, _ S: Int, _ n: Int, _ v16: Bool) {
    let nc = (n + CK - 1) / CK
    var prm = Prm(kh: UInt32(256 * S), kd: UInt32(S), sh: UInt32(S), vh: UInt32(S * 256), vr: 256, n: UInt32(n), ck: UInt32(CK), nc: UInt32(nc), stage: UInt32(STAGE))
    e.setComputePipelineState(v16 ? pA16 : pA8)
    for (i, b) in [Q8, QS, l.k, l.ks, v16 ? l.vc : l.v, l.vs, SC, PS, OP, MP, LP].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 11)
    e.dispatchThreadgroups(MTLSize(width: nc, height: 1, depth: 4), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    e.memoryBarrier(scope: .buffers)
    e.setComputePipelineState(pC)
    for (i, b) in [OP, MP, LP, O].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 4)
    e.dispatchThreads(MTLSize(width: 256, height: 48, depth: 4), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.memoryBarrier(scope: .buffers)
}
// ---- correctness: one layer, N = 4096, against a CPU reference ----
do {
    let S = 4096, n = 4096, l = makeLay(S)
    for v16 in [false, true] {
        let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        encode(e, l, S, n, v16); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        precondition(cb.error == nil, "\(cb.error!)")
        let q8 = Q8.contents().bindMemory(to: Int8.self, capacity: 4 * 64 * 256), k8 = l.k.contents().bindMemory(to: Int8.self, capacity: 4 * 256 * S)
        let ks = l.ks.contents().bindMemory(to: Float16.self, capacity: 4 * S), v8 = l.v.contents().bindMemory(to: Int8.self, capacity: 4 * S * 256)
        let vs = l.vs.contents().bindMemory(to: Float16.self, capacity: 4 * S), og = O.contents().bindMemory(to: Float.self, capacity: 4 * 48 * 256)
        var maxErr: Float = 0, maxRef: Float = 0
        for h in 0..<4 { for r in stride(from: 0, to: 48, by: 7) {
            var s = [Float](repeating: 0, count: n)
            for k in 0..<n { var acc: Int32 = 0; for d in 0..<256 { acc += Int32(q8[(h * 64 + r) * 256 + d]) * Int32(k8[h * 256 * S + d * S + k]) }
                s[k] = Float(acc) * 0.003 * Float(ks[h * S + k]) }
            let mx = s.max()!; var den: Float = 0; var o = [Float](repeating: 0, count: 256)
            for k in 0..<n { let e = exp(s[k] - mx); den += e; let w = e * Float(vs[h * S + k]); for d in 0..<256 { o[d] += w * Float(v8[(h * S + k) * 256 + d]) } }
            for d in 0..<256 { let ref = o[d] / den; maxRef = max(maxRef, abs(ref)); maxErr = max(maxErr, abs(ref - og[(h * 48 + r) * 256 + d])) }
        } }
        print(String(format: "check %@ V, N %d: max |ref| %.4f, max error %.5f (%.2f%%)", v16 ? "fp16" : "int8", n, maxRef, maxErr, 100 * maxErr / maxRef))
    }
}
// ---- timing: 16 layers per command buffer, NLB distinct layers of SMAX keys cycled ----
let lays = (0..<NLB).map { _ in makeLay(SMAX) }
func ms(_ n: Int, _ v16: Bool) -> Double {
    var ts: [Double] = []
    for _ in 0..<REPS {
        let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        for li in 0..<16 { encode(e, lays[li % NLB], SMAX, n, v16) }
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted(); precondition(cb.error == nil, "\(cb.error!)")
        ts.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
    }
    return ts.sorted()[ts.count / 2]
}
_ = ms(8192, false); _ = ms(8192, true)
for n in NS {
    let a = ms(n, false), b = ms(n, true)
    print(String(format: "N %6d (CK %d): int8 V %6.2f ms (%.3f ms/1K, %3.0f GB/s)   fp16 V %6.2f ms (%.3f ms/1K, %3.0f GB/s)", n, CK,
                 a, a / Double(n) * 1024, Double(16 * 4 * n * 512) / a / 1e6, b, b / Double(n) * 1024, Double(16 * 4 * n * 768) / b / 1e6))
}
