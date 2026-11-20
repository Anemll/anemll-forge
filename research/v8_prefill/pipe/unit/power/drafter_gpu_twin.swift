// GPU cost twin of the DFlash2 drafter body (19 Nov 2026, M6): every LUT matrix of one draft call at its real shape
// (drafter_all3_gptq_qmixr: vector LUT, 64-entry palette of row pairs, one U8 index per pair, per-row scale; fc 25600 ->
// 5120 once; 5 layers of q 4096, k 1024, v 1024, o 5120 (from 4096), gate / up 17408, down 5120 (from 17408), and the two
// INT8 conv kernel projections 1280 x 5120), each as a decode-to-threadgroup + matmul2d GEMM over R rows (16 = the 8
// context rows + 8 query rows of a draft call), fp16 activations, fp32 accumulate. SYNTHETIC indices / palettes (no
// weights are loaded: timing only). Attention over the 2048-slot ring is not included (about 1 GFLOP per call).
// The U8-per-pair index is 4 bits per weight (865 MB per call); the ANE package packs 6-bit pairs (3 bits per weight,
// 718 MB with the INT8 convs): the report scales by bytes.
//   drafter_gpu_twin [ROWS] [SECONDS] [PERIOD_MS]     PERIOD_MS > 0: one call every PERIOD_MS (duty-cycled), else
//   back to back. Prints the median call ms and the achieved index-stream GB/s.
import Foundation
import Metal

let a = CommandLine.arguments
let R = a.count > 1 ? Int(a[1])! : 16, secs = a.count > 2 ? Double(a[2])! : 3.0, period = a.count > 3 ? Double(a[3])! : 0
let src = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#define R \(R)
#define TM 32
#define KS 256
// y[r][o] = sum_k x[k][r] W[o][k], W[o][k] = lut[idx[o / 2][k]][o % 2] * sc[o]; x is [K][R] fp16, y [R][O] fp32
kernel void lutg(device const uchar *idx [[buffer(0)]], device const half *lut [[buffer(1)]], device const half *sc [[buffer(2)]],
                 device const half *x [[buffer(3)]], device float *y [[buffer(4)]], constant uint2 &ok [[buffer(5)]],
                 uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W[TM * KS] __attribute__((aligned(16)));
    threadgroup half2 P[64];
    uint O = ok.x, K = ok.y, o0 = tg * TM;
    if (tid < 64) P[tid] = ((device const half2 *)lut)[tid];
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tA(W, dextents<int32_t, 2>(KS, TM));
    constexpr auto d = matmul2d_descriptor(TM, R, KS, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    auto c = op.template get_destination_cooperative_tensor<decltype(tA), tensor<device half, dextents<int32_t, 2>, tensor_inline>, float>();
    for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
    for (uint k0 = 0; k0 < K; k0 += KS) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 16 pair rows x 256 columns of indices (4 KB) -> 32 x 256 halves; 128 threads, 32 indices each (uchar4 loads)
        for (uint i = tid; i < (TM / 2) * (KS / 4); i += 128) {
            uint pr = i / (KS / 4), c4 = (i % (KS / 4)) * 4;
            uchar4 q = *(device const uchar4 *)(idx + ulong(o0 / 2 + pr) * K + k0 + c4);
            half2 w0 = P[q.x & 63], w1 = P[q.y & 63], w2 = P[q.z & 63], w3 = P[q.w & 63];
            *(threadgroup half4 *)(W + (2 * pr) * KS + c4) = half4(w0.x, w1.x, w2.x, w3.x);
            *(threadgroup half4 *)(W + (2 * pr + 1) * KS + c4) = half4(w0.y, w1.y, w2.y, w3.y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tB((device half *)(x + ulong(k0) * R), dextents<int32_t, 2>(R, KS));
        op.run(tA, tB, c);
    }
    for (uint16_t i = 0; i < c.get_capacity(); ++i) {
        if (!c.is_valid_element(i)) continue;
        auto ix = c.get_multidimensional_index(i);
        uint r = ix[0], o = o0 + ix[1];
        y[r * O + o] = c[i] * float(sc[o]);
    }
}

// v2: the engine's lut_gemm32_i8t pattern: 6-bit pair indices packed (the ANE package's 3 bits per weight), per tile of
// TM = 32 outputs (16 pairs) and column c: 16 x 6 bits; 8 indices per 64-bit read; pair palette of INT8 codes (ushort);
// W tile [KS][TM] int8 (transpose_left); x int8 [K][R]; INT8 x INT8 -> int32
kernel void lutg8(device const uchar *k [[buffer(0)]], device const half *lut [[buffer(1)]], device const half *sc [[buffer(2)]],
                  device const int8_t *x [[buffer(3)]], device float *y [[buffer(4)]], constant uint2 &ok [[buffer(5)]],
                  uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int8_t W[TM * KS] __attribute__((aligned(16)));
    threadgroup ushort PP[64];
    uint O = ok.x, K = ok.y, o0 = tg * TM;
    if (tid < 64) { half2 p = ((device const half2 *)lut)[tid]; PP[tid] = ushort(uchar(int8_t(rint(float(p.x) * 100.0f)))) | ushort(ushort(uchar(int8_t(rint(float(p.y) * 100.0f)))) << 8); }
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(W, dextents<int32_t, 2>(TM, KS));
    constexpr auto d = matmul2d_descriptor(TM, R, KS, true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    auto c = op.template get_destination_cooperative_tensor<decltype(tA), tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>, int32_t>();
    for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
    device const uint *w = (device const uint *)(k + ulong(tg) * K * 12);
    for (uint k0 = 0; k0 < K; k0 += KS) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint it = tid; it < KS * 2; it += 128) {       // item = (column, half): 8 pair indices
            uint cc = it >> 1, h = it & 1;
            uint b = ((k0 + cc) * 16 + 8 * h) * 6, wi = b >> 5, sh = b & 31;
            ulong v = (ulong(w[wi]) | (ulong(w[wi + 1]) << 32)) >> sh;
            uint ix[8];
            for (uint r = 0; r < 8; r++) ix[r] = uint(v >> (r * 6)) & 63u;
            uint4 o = uint4(uint(PP[ix[0]]) | (uint(PP[ix[1]]) << 16), uint(PP[ix[2]]) | (uint(PP[ix[3]]) << 16),
                            uint(PP[ix[4]]) | (uint(PP[ix[5]]) << 16), uint(PP[ix[6]]) | (uint(PP[ix[7]]) << 16));
            *(threadgroup uint4 *)(W + cc * TM + 16 * h) = o;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB((device int8_t *)(x + ulong(k0) * R), dextents<int32_t, 2>(R, KS));
        op.run(tA, tB, c);
    }
    for (uint16_t i = 0; i < c.get_capacity(); ++i) {
        if (!c.is_valid_element(i)) continue;
        auto ix2 = c.get_multidimensional_index(i);
        uint r = ix2[0], o = o0 + ix2[1];
        y[r * O + o] = float(c[i]) * float(sc[o]);
    }
}
"""
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: opts) } catch { print("COMPILE FAILED: \(error)".prefix(2000)); exit(1) }
let V2 = ProcessInfo.processInfo.environment["V2"] == "1"
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: V2 ? "lutg8" : "lutg")!)
// (name, out, in, count per call)
var mats: [(String, Int, Int, Int)] = [("fc", 5120, 25600, 1)]
for _ in 0..<5 { mats += [("q", 4096, 5120, 1), ("k", 1024, 5120, 1), ("v", 1024, 5120, 1), ("o", 5120, 4096, 1),
                          ("gate", 17408, 5120, 1), ("up", 17408, 5120, 1), ("down", 5120, 17408, 1), ("conv", 1280, 5120, 2)] }
var g: UInt64 = 0x9E3779B97F4A7C15
func rnd() -> UInt64 { g ^= g << 13; g ^= g >> 7; g ^= g << 17; return g }
struct M { let idx: MTLBuffer, lut: MTLBuffer, sc: MTLBuffer, x: MTLBuffer, y: MTLBuffer, o: Int, k: Int }
var bufs: [M] = []; var idxBytes = 0, convBytes = 0
for (name, o, k, n) in mats {
    for _ in 0..<n {
        let ib = dev.makeBuffer(length: V2 ? o / 32 * k * 12 + 16 : o / 2 * k, options: .storageModeShared)!
        let p = ib.contents().bindMemory(to: UInt64.self, capacity: ib.length / 8)
        for i in 0..<(ib.length / 8) { p[i] = rnd() & 0x3F3F3F3F3F3F3F3F }
        let lb = dev.makeBuffer(length: 64 * 2 * 2, options: .storageModeShared)!
        let lp = lb.contents().bindMemory(to: Float16.self, capacity: 128); for i in 0..<128 { lp[i] = Float16(Float(Int(rnd() % 255) - 127) / 1270) }
        let sb = dev.makeBuffer(length: o * 2, options: .storageModeShared)!
        let sp = sb.contents().bindMemory(to: Float16.self, capacity: o); for i in 0..<o { sp[i] = Float16(0.01) }
        let xb = dev.makeBuffer(length: k * R * 2, options: .storageModeShared)!
        if V2 { let xq = xb.contents().bindMemory(to: Int8.self, capacity: k * R); for i in 0..<(k * R) { xq[i] = Int8(Int(rnd() % 200) - 100) } }
        else { let xp = xb.contents().bindMemory(to: Float16.self, capacity: k * R); for i in 0..<(k * R) { xp[i] = Float16(Float(Int(rnd() % 200) - 100) / 100) } }
        bufs.append(M(idx: ib, lut: lb, sc: sb, x: xb, y: dev.makeBuffer(length: o * R * 4, options: .storageModeShared)!, o: o, k: k))
        if name == "conv" { convBytes += V2 ? o * k * 3 / 8 : o * k } else { idxBytes += V2 ? o / 32 * k * 12 : o / 2 * k }
    }
}
func call() -> Double {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso)
    for m in bufs {
        e.setBuffer(m.idx, offset: 0, index: 0); e.setBuffer(m.lut, offset: 0, index: 1); e.setBuffer(m.sc, offset: 0, index: 2)
        e.setBuffer(m.x, offset: 0, index: 3); e.setBuffer(m.y, offset: 0, index: 4)
        var ok = SIMD2<UInt32>(UInt32(m.o), UInt32(m.k)); e.setBytes(&ok, length: 8, index: 5)
        e.dispatchThreadgroups(MTLSize(width: m.o / 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if let err = cb.error { print("GPU error \(err)"); exit(1) }
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}
// correctness of the kernel on a small matrix (the last conv): y[r][o] vs CPU
if !V2 {
    let m = bufs.last!; _ = call()
    let ip = m.idx.contents().bindMemory(to: UInt8.self, capacity: m.idx.length), lp = m.lut.contents().bindMemory(to: Float16.self, capacity: 128)
    let xp = m.x.contents().bindMemory(to: Float16.self, capacity: m.k * R), yp = m.y.contents().bindMemory(to: Float.self, capacity: m.o * R)
    var maxe: Float = 0, maxr: Float = 0
    for t in 0..<200 {
        let r = t % R, o = Int(rnd() % UInt64(m.o)); var acc: Float = 0
        for k in 0..<m.k { acc += Float(xp[k * R + r]) * Float(lp[Int(ip[(o / 2) * m.k + k] & 63) * 2 + o % 2]) }
        acc *= 0.01; maxe = max(maxe, abs(acc - yp[r * m.o + o])); maxr = max(maxr, abs(acc))
    }
    print(String(format: "check: max |err| %.2e of max |y| %.2e", maxe, maxr))
}
// GO_FILE: buffers ready -> wait for the file to exist before any GPU work (overlap tests start the twin early);
// SIGTERM ends the timed loop and still prints the result; WARM = warm-up seconds (default 1.5; 0 for overlap tests)
var stopFlag = false
signal(SIGTERM, SIG_IGN)
let sigSrc = DispatchSource.makeSignalSource(signal: SIGTERM, queue: DispatchQueue.global()); sigSrc.setEventHandler { stopFlag = true }; sigSrc.resume()
if let gf = ProcessInfo.processInfo.environment["GO_FILE"] { print("ready", terminator: "\n"); fflush(stdout); while !FileManager.default.fileExists(atPath: gf) && !stopFlag { usleep(2000) } }
let warm = Double(ProcessInfo.processInfo.environment["WARM"] ?? "1.5") ?? 1.5
let t0 = Date(); while Date().timeIntervalSince(t0) < warm { _ = call() }
if ProcessInfo.processInfo.environment["PERMAT"] == "1" {   // GPU ms per matrix type (each type alone in its own command buffer, x20)
    var names: [String] = []; for (n, _, _, c) in mats { for _ in 0..<c { names.append(n) } }
    for t in ["fc", "q", "k", "v", "o", "gate", "up", "down", "conv"] {
        var best = 1e9
        for _ in 0..<20 {
            let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
            e.setComputePipelineState(pso)
            for (i, m) in bufs.enumerated() where names[i] == t {
                e.setBuffer(m.idx, offset: 0, index: 0); e.setBuffer(m.lut, offset: 0, index: 1); e.setBuffer(m.sc, offset: 0, index: 2)
                e.setBuffer(m.x, offset: 0, index: 3); e.setBuffer(m.y, offset: 0, index: 4)
                var ok = SIMD2<UInt32>(UInt32(m.o), UInt32(m.k)); e.setBytes(&ok, length: 8, index: 5)
                e.dispatchThreadgroups(MTLSize(width: m.o / 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
            e.endEncoding(); cb.commit(); cb.waitUntilCompleted(); best = min(best, (cb.gpuEndTime - cb.gpuStartTime) * 1e3)
        }
        let by = bufs.enumerated().filter { names[$0.offset] == t }.map { V2 ? $0.element.o / 32 * $0.element.k * 12 : $0.element.o / 2 * $0.element.k }.reduce(0, +)
        print(String(format: "  %-5@ %7.3f ms (all of this type in one call)  %6.1f MB  %5.1f GB/s  %d tgs each", t, best, Double(by) / 1e6, Double(by) / best / 1e6, bufs[names.firstIndex(of: t)!].o / 32))
    }
}
var ms: [Double] = []; var walls: [Double] = []
let t1 = Date()
while Date().timeIntervalSince(t1) < secs && !stopFlag {
    let w0 = Date(); ms.append(call()); walls.append(Date().timeIntervalSince(w0) * 1e3)
    if period > 0 { let rest = period / 1e3 - Date().timeIntervalSince(w0); if rest > 0 { Thread.sleep(forTimeInterval: rest) } }
}
if ms.isEmpty { print("no calls"); exit(0) }
ms.sort(); walls.sort()
let med = ms[ms.count / 2]
print(String(format: "drafter twin R %d%@: %d calls, GPU %.3f ms median (p10 %.3f, p90 %.3f), wall %.3f | index %.0f MB + INT8 conv %.0f MB -> %.1f GB/s | scaled to the 6-bit package (718 MB): %.2f ms",
             R, period > 0 ? String(format: ", every %.0f ms", period) : "", ms.count, med, ms[ms.count / 10], ms[ms.count * 9 / 10], walls[walls.count / 2],
             Double(idxBytes) / 1e6, Double(convBytes) / 1e6, Double(idxBytes + convBytes) / med / 1e6, V2 ? med : med * (Double(idxBytes) * 0.75 + Double(convBytes)) / Double(idxBytes + convBytes)))
if V2 { print("(V2: 6-bit packed indices, the package's byte count; the INT8 conv projections modelled as 3-bit LUT matrices)") }
