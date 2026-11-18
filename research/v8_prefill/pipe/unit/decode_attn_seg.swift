// Decode-verify attention on the GPU over the ANE's segmented INT8 KV cache (16 Nov 2026): the chosen flash-decoding
// kernel (decode_attn_flash5.swift: 48 query vectors per KV head in 3 threadgroups of 16 rows, 4 simdgroups, 256-key
// sub-tiles, CK-key chunks) with what the real verify needs: the engine's query layout (Q [24 heads][8 rows][256],
// KV head g = query heads 6g .. 6g + 5, so its 48 vectors are contiguous: index j = head_local * 8 + row), a causal
// mask among the 8 verify rows (row t at position p0 + t sees keys <= p0 + t), any key count N = p0 + 8 (keys >= N
// masked), and the cache as segments with their own buffers and strides (keys [kvh][256][rows] dim-major + key scale
// [kvh][rows], values [kvh][rows][256] + row scale). Checked against a CPU reference on a small segment layout with
// unaligned N, then timed on the real one (8K, 8K, 16K, 32K, 32K) for 16 layers.
//   swiftc -O -o decode_attn_seg decode_attn_seg.swift -framework Metal && ./decode_attn_seg
import Foundation
import Metal

let env = ProcessInfo.processInfo.environment
let CK = Int(env["CK"] ?? "2048")!, REPS = Int(env["REPS"] ?? "5")!, NLB = Int(env["NLAYERS"] ?? "4")!
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#define NSEG 5
struct DPrm { uint n, p0, ck, nc; uint start[NSEG]; uint rows[NSEG]; };
constant constexpr int NSG = 4, SUB = 256, M = 16, QSP = 3, NT = 32 * NSG, TPR = NT / M, KPT = SUB / TPR;

kernel void dec_attn(device const int8_t *Q8 [[buffer(0)]], device const float *QS [[buffer(1)]],
                     device const int8_t *K0 [[buffer(2)]], device const int8_t *K1 [[buffer(3)]], device const int8_t *K2 [[buffer(4)]],
                     device const int8_t *K3 [[buffer(5)]], device const int8_t *K4 [[buffer(6)]],
                     device const half *KS0 [[buffer(7)]], device const half *KS1 [[buffer(8)]], device const half *KS2 [[buffer(9)]],
                     device const half *KS3 [[buffer(10)]], device const half *KS4 [[buffer(11)]],
                     device const int8_t *V0 [[buffer(12)]], device const int8_t *V1 [[buffer(13)]], device const int8_t *V2 [[buffer(14)]],
                     device const int8_t *V3 [[buffer(15)]], device const int8_t *V4 [[buffer(16)]],
                     device const half *VS0 [[buffer(17)]], device const half *VS1 [[buffer(18)]], device const half *VS2 [[buffer(19)]],
                     device const half *VS3 [[buffer(20)]], device const half *VS4 [[buffer(21)]],
                     device float *OP [[buffer(22)]], device float *MP [[buffer(23)]], device float *LP [[buffer(24)]],
                     constant DPrm &p [[buffer(25)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    threadgroup half S[M * SUB]; threadgroup float alpha[M], qs[M], ks[SUB], vs[SUB];
    uint c = tg.x, qi = tg.y, gh = tg.z, r0 = qi * M;
    uint k0 = c * p.ck, k1 = min(k0 + p.ck, p.n);
    uint sg = 0;
    for (uint i = 1; i < NSEG; ++i) if (p.rows[i] && k0 >= p.start[i]) sg = i;   // a chunk never crosses a segment
    uint R = p.rows[sg], s0 = p.start[sg];
    device const int8_t *K = sg == 0 ? K0 : sg == 1 ? K1 : sg == 2 ? K2 : sg == 3 ? K3 : K4;
    device const half *KS = sg == 0 ? KS0 : sg == 1 ? KS1 : sg == 2 ? KS2 : sg == 3 ? KS3 : KS4;
    device const int8_t *V = sg == 0 ? V0 : sg == 1 ? V1 : sg == 2 ? V2 : sg == 3 ? V3 : V4;
    device const half *VS = sg == 0 ? VS0 : sg == 1 ? VS1 : sg == 2 ? VS2 : sg == 3 ? VS3 : VS4;
    uint slot = (gh * p.nc + c) * QSP + qi;
    if (t < M) qs[t] = QS[gh * 48 + r0 + t];
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tQ((device int8_t *)Q8 + (gh * 48 + r0) * 256, dextents<int32_t, 2>(256, M));
    constexpr auto dqk = matmul2d_descriptor(M, SUB, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<dqk, execution_simdgroups<NSG>> qk;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(S, dextents<int32_t, 2>(SUB, M), array<int32_t, 2>{1, SUB});
    constexpr auto dpv = matmul2d_descriptor(M, 256, SUB, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dpv, execution_simdgroups<NSG>> pv;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tV0((device int8_t *)V + gh * R * 256, dextents<int32_t, 2>(256, SUB),
                                                                   array<int32_t, 2>{1, 256});
    auto cO = pv.template get_destination_cooperative_tensor<decltype(tP), decltype(tV0), float>();
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0.0f;
    uint row = t / TPR, part = t % TPR;
    float m = -INFINITY, l = 0.0f;
    for (uint kb = k0; kb < k1; kb += SUB) {
        uint lb = kb - s0;                                    // segment-local row of the sub-tile
        for (uint j = t; j < SUB; j += NT) {
            bool in = kb + j < p.n;
            ks[j] = in ? float(KS[gh * R + lb + j]) : 0.0f; vs[j] = in ? float(VS[gh * R + lb + j]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK((device int8_t *)K + gh * 256 * R + lb, dextents<int32_t, 2>(SUB, 256),
                                                                      array<int32_t, 2>{1, int(R)});
        auto cI = qk.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
        qk.run(tQ, tK, cI);
        for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
            if (!cI.is_valid_element(i)) continue;
            auto ix = cI.get_multidimensional_index(i);
            uint key = kb + ix[0], tok = (r0 + ix[1]) & 7;    // the query's verify row
            bool vis = key < p.n && key <= p.p0 + tok;         // keys before p0 are all visible
            S[ix[1] * SUB + ix[0]] = vis ? half(float(cI[i]) * qs[ix[1]] * ks[ix[0]]) : half(-INFINITY);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half *Sr = S + row * SUB + part * KPT;
        float tm = -INFINITY;
        for (int j = 0; j < KPT; ++j) tm = max(tm, float(Sr[j]));
        for (int o = 1; o < TPR; o <<= 1) tm = max(tm, simd_shuffle_xor(tm, o));
        float mn = max(m, tm), a = (mn == -INFINITY) ? 1.0f : exp(m - mn), sum = 0.0f;
        for (int j = 0; j < KPT; ++j) {
            float sv = float(Sr[j]);
            float e = (sv == -INFINITY) ? 0.0f : exp(sv - mn); sum += e;
            Sr[j] = half(e * vs[part * KPT + j]);
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
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tV((device int8_t *)V + (gh * R + lb) * 256, dextents<int32_t, 2>(256, SUB),
                                                                      array<int32_t, 2>{1, 256});
        pv.run(tP, tV, cO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        OP[(slot * M + ix[1]) * 256 + ix[0]] = cO[i];
    }
    if (part == 0) { MP[slot * M + row] = m; LP[slot * M + row] = l; }
}
// O [4 kv heads][48][256] = the 24 query heads x 8 rows in the engine's order
kernel void dec_combine(device const float *OP [[buffer(0)]], device const float *MP [[buffer(1)]], device const float *LP [[buffer(2)]],
                        device float *O [[buffer(3)]], constant DPrm &p [[buffer(4)]], uint3 g [[thread_position_in_grid]]) {
    uint d = g.x, j = g.y, gh = g.z, qi = j / M, r = j % M;
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
    O[(gh * 48 + j) * 256 + d] = num / den;
}
"""
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
let lib = try! dev.makeLibrary(source: src, options: opts)
let pA = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "dec_attn")!)
let pC = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "dec_combine")!)
func buf(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: max(n, 16), options: .storageModeShared)! }
struct DPrm { var n, p0, ck, nc: UInt32; var start: (UInt32, UInt32, UInt32, UInt32, UInt32); var rows: (UInt32, UInt32, UInt32, UInt32, UInt32) }
var rng: UInt64 = 0x9E3779B97F4A7C15
func rnd() -> UInt32 { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return UInt32(truncatingIfNeeded: rng >> 16) }
func fillH(_ b: MTLBuffer, _ lo: Float, _ hi: Float) {
    let n = b.length / 2, p = b.contents().bindMemory(to: Float16.self, capacity: n)
    for i in 0..<n { p[i] = Float16(lo + (hi - lo) * Float(rnd() >> 8) / 16777216) }
}
struct Seg { let rows: Int, k, ks, v, vs: MTLBuffer }
func makeSegs(_ sizes: [Int]) -> [Seg] {
    sizes.map { r in
        let s = Seg(rows: r, k: buf(4 * 256 * r), ks: buf(4 * r * 2), v: buf(4 * r * 256), vs: buf(4 * r * 2))
        arc4random_buf(s.k.contents(), s.k.length); arc4random_buf(s.v.contents(), s.v.length); fillH(s.ks, 0.002, 0.004); fillH(s.vs, 0.005, 0.02)
        return s
    }
}
let Q8 = buf(4 * 48 * 256), QS = buf(4 * 48 * 4)
do {
    arc4random_buf(Q8.contents(), Q8.length)
    let s = QS.contents().bindMemory(to: Float.self, capacity: 192); for i in 0..<192 { s[i] = 0.002 + 0.002 * Float(rnd() % 1000) / 1000 }
}
let NCMAX = 98304 / 256 + 1
let OP = buf(4 * NCMAX * 3 * 16 * 256 * 4), MP = buf(4 * NCMAX * 3 * 16 * 4), LP = buf(4 * NCMAX * 3 * 16 * 4), O = buf(4 * 48 * 256 * 4)
func prm(_ segs: [Seg], _ n: Int, _ ck: Int) -> DPrm {
    var st = [UInt32](repeating: 0, count: 5), rw = [UInt32](repeating: 0, count: 5), a = 0
    for (i, s) in segs.enumerated() { st[i] = UInt32(a); rw[i] = UInt32(s.rows); a += s.rows }
    return DPrm(n: UInt32(n), p0: UInt32(n - 8), ck: UInt32(ck), nc: UInt32((n + ck - 1) / ck),
                start: (st[0], st[1], st[2], st[3], st[4]), rows: (rw[0], rw[1], rw[2], rw[3], rw[4]))
}
func encode(_ e: MTLComputeCommandEncoder, _ segs: [Seg], _ n: Int, _ ck: Int) {
    var p = prm(segs, n, ck)
    e.setComputePipelineState(pA)
    e.setBuffer(Q8, offset: 0, index: 0); e.setBuffer(QS, offset: 0, index: 1)
    for i in 0..<5 {
        let s = segs[min(i, segs.count - 1)]
        e.setBuffer(s.k, offset: 0, index: 2 + i); e.setBuffer(s.ks, offset: 0, index: 7 + i)
        e.setBuffer(s.v, offset: 0, index: 12 + i); e.setBuffer(s.vs, offset: 0, index: 17 + i)
    }
    e.setBuffer(OP, offset: 0, index: 22); e.setBuffer(MP, offset: 0, index: 23); e.setBuffer(LP, offset: 0, index: 24)
    e.setBytes(&p, length: MemoryLayout<DPrm>.size, index: 25)
    e.dispatchThreadgroups(MTLSize(width: Int(p.nc), height: 3, depth: 4), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    e.memoryBarrier(scope: .buffers)
    e.setComputePipelineState(pC)
    for (i, b) in [OP, MP, LP, O].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    e.setBytes(&p, length: MemoryLayout<DPrm>.size, index: 4)
    e.dispatchThreads(MTLSize(width: 256, height: 48, depth: 4), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.memoryBarrier(scope: .buffers)
}
// ---- correctness: segments 2048, 2048, 4096; unaligned N; small chunks too ----
let tsegs = makeSegs([2048, 2048, 4096])
func ref(_ segs: [Seg], _ n: Int) -> [Float] {
    var out = [Float](repeating: 0, count: 4 * 48 * 256)
    let q8 = Q8.contents().bindMemory(to: Int8.self, capacity: 4 * 48 * 256), qs = QS.contents().bindMemory(to: Float.self, capacity: 192)
    func at(_ key: Int) -> (Seg, Int) { var a = 0; for s in segs { if key < a + s.rows { return (s, key - a) }; a += s.rows }; fatalError() }
    let p0 = n - 8
    for h in 0..<4 { for j in 0..<48 {
        let tok = j % 8, last = p0 + tok
        var s = [Float](repeating: -Float.infinity, count: last + 1)
        for k in 0...last {
            let (sg, r) = at(k); let k8 = sg.k.contents().bindMemory(to: Int8.self, capacity: 4 * 256 * sg.rows)
            let ks = sg.ks.contents().bindMemory(to: Float16.self, capacity: 4 * sg.rows)
            var acc: Int32 = 0; for d in 0..<256 { acc += Int32(q8[(h * 48 + j) * 256 + d]) * Int32(k8[h * 256 * sg.rows + d * sg.rows + r]) }
            s[k] = Float(acc) * qs[h * 48 + j] * Float(ks[h * sg.rows + r])
        }
        let mx = s.max()!; var den: Float = 0; var o = [Float](repeating: 0, count: 256)
        for k in 0...last {
            let (sg, r) = at(k); let v8 = sg.v.contents().bindMemory(to: Int8.self, capacity: 4 * sg.rows * 256)
            let vs = sg.vs.contents().bindMemory(to: Float16.self, capacity: 4 * sg.rows)
            let e = exp(s[k] - mx); den += e; let w = e * Float(vs[h * sg.rows + r])
            for d in 0..<256 { o[d] += w * Float(v8[(h * sg.rows + r) * 256 + d]) }
        }
        for d in 0..<256 { out[(h * 48 + j) * 256 + d] = o[d] / den }
    } }
    return out
}
var allOK = true
for (n, ck) in [(7000, 2048), (6151, 1024), (4100, 2048), (8192, 2048), (2056, 512)] {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    encode(e, tsegs, n, ck); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    precondition(cb.error == nil, "\(cb.error!)")
    let r = ref(tsegs, n), og = O.contents().bindMemory(to: Float.self, capacity: 4 * 48 * 256)
    var me: Float = 0, mr: Float = 0, nan = 0
    for i in 0..<r.count { if !og[i].isFinite { nan += 1 }; mr = max(mr, abs(r[i])); me = max(me, abs(r[i] - og[i])) }
    let okk = nan == 0 && me / mr < 0.01; allOK = allOK && okk
    print(String(format: "check N %5d (p0 %5d, CK %4d, segments 2K/2K/4K): max error %.3f%% of %.4f, non-finite %d  %@", n, n - 8, ck, 100 * me / mr, mr, nan, okk ? "OK" : "WRONG"))
}
guard allOK else { exit(1) }
// ---- timing on the real segment layout, 16 layers (NLB distinct layers cycled) ----
let real = (0..<NLB).map { _ in makeSegs([8192, 8192, 16384, 32768, 32768]) }
func ms(_ n: Int) -> Double {
    var ts: [Double] = []
    for _ in 0..<REPS {
        let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        for li in 0..<16 { encode(e, real[li % NLB], n, CK) }
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted(); precondition(cb.error == nil, "\(cb.error!)")
        ts.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
    }
    return ts.sorted()[ts.count / 2]
}
_ = ms(16384)
for n in [8200, 16392, 24584, 32776, 33000, 49160, 65544, 65608, 81928, 98304] {
    let t = ms(n)
    print(String(format: "N %6d: %6.2f ms for 16 layers (%.3f ms per 1K keys)", n, t, t / Double(n) * 1024))
}
