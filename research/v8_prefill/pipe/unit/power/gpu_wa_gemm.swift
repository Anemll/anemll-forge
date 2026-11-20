// Weight / activation data types on the GPU at prefill shapes (18 Nov 2026, M6): Y[TOK x N] = X[TOK x K] W[K x N],
// matmul2d with the activations X as the left operand (TOK tokens per call) and the weights W as the right operand
// (the formats the M6 takes for weights: half, bfloat, int8, int4b, fp8 e4m3, fp4 e2m1), every weight read from DRAM
// once per dispatch (N = 16384 outputs, K = 8192: 64 MB at 4 bits up to 256 MB at 16), activations in cache.
// Prints "ready", then one line every 0.5 s: "<epoch> GPU <mode> <TOPS> TOPS busy <pct> wgbs <weight GB/s>".
//   gpu_wa_gemm MODE TOK SECONDS      MODE: w16a16 w8a16 w4a16 w8a8 w4a8 f8f8 w4f8? (see table)   TOK: 32 64 128
import Foundation
import Metal

let a = CommandLine.arguments
let mode = a[1], tok = Int(a[2])!, secs = Double(a[3])!
// mode: (activation type, weight type, accumulator, activation bits, weight bits)
let M: [String: (String, String, String, Int, Int)] = [
    "w16a16": ("half", "half", "float", 16, 16),
    "bf16": ("bfloat", "bfloat", "float", 16, 16),
    "w8a16": ("half", "int8_t", "float", 16, 8),
    "w4a16": ("half", "int4b_format", "float", 16, 4),
    "w8a8": ("int8_t", "int8_t", "int32_t", 8, 8),
    "w4a8": ("int8_t", "int4b_format", "int32_t", 8, 4),
    "f8f8": ("metal_fp8_e4m3_format", "metal_fp8_e4m3_format", "float", 8, 8),
    "wf8a16": ("half", "metal_fp8_e4m3_format", "float", 16, 8),
    "wf4a16": ("half", "metal_fp4_e2m1_format", "float", 16, 4),
    "f4f4": ("metal_fp4_e2m1_format", "metal_fp4_e2m1_format", "float", 4, 4),
]
guard let (xt, wt, ct, xb, wb) = M[mode] else { fatalError("mode") }
let N = 16384, K = 8192, TN = 64, KB = 256
let src = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#define TOK \(tok)
#define TN \(TN)
#define KB \(KB)
#define K \(K)
kernel void wa(device uchar *X [[buffer(0)]], device uchar *W [[buffer(1)]], device int *o [[buffer(2)]],
               uint tg [[threadgroup_position_in_grid]]) {
    constexpr auto d = matmul2d_descriptor(TOK, TN, KB, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    using TX = tensor<device \(xt), dextents<int32_t, 2>, tensor_inline>;
    using TW = tensor<device \(wt), dextents<int32_t, 2>, tensor_inline>;
    auto c = op.template get_destination_cooperative_tensor<TX, TW, \(ct)>();
    for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
    for (uint ks = 0; ks < K / KB; ks++) {
        TX tX((typename TX::data_handle_type)(X + ks * (TOK * KB * \(xb) / 8)), dextents<int32_t, 2>(KB, TOK));
        TW tW((typename TW::data_handle_type)(W + (tg * (K / KB) + ks) * (KB * TN * \(wb) / 8)), dextents<int32_t, 2>(TN, KB));
        op.run(tX, tW, c);
    }
    float s = 0;
    for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) s += float(c[i]);
    if (s == 123456.0f) o[tg] = int(s);
}
"""
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: opts) } catch { print("GPU \(mode) TOK \(tok): COMPILE FAILED: \(error)".prefix(500)); exit(1) }
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "wa")!)
let xbytes = tok * K * xb / 8, wbytes = N * K * wb / 8
let X = dev.makeBuffer(length: xbytes, options: .storageModeShared)!, W = dev.makeBuffer(length: wbytes, options: .storageModeShared)!
let o = dev.makeBuffer(length: (N / TN) * 4, options: .storageModeShared)!
func fill(_ b: MTLBuffer, _ t: String) {   // 1.0 (or 1) in each format, no NaN / Inf patterns
    let p = b.contents().bindMemory(to: UInt8.self, capacity: b.length)
    switch t {
    case "half": for i in stride(from: 0, to: b.length, by: 2) { p[i] = 0x00; p[i + 1] = 0x3C }
    case "bfloat": for i in stride(from: 0, to: b.length, by: 2) { p[i] = 0x80; p[i + 1] = 0x3F }
    case "metal_fp8_e4m3_format": memset(p, 0x38, b.length)
    case "metal_fp4_e2m1_format": memset(p, 0x22, b.length)
    case "int4b_format": memset(p, 0x11, b.length)
    default: memset(p, 0x01, b.length)
    }
}
fill(X, xt); fill(W, wt)
let reps = max(1, Int(4e-3 / (2.0 * Double(tok) * Double(N) * Double(K) / 30e12)))   // about 4 ms per command buffer
let work = 2.0 * Double(tok) * Double(N) * Double(K) * Double(reps) / 1e12, wwork = Double(wbytes * reps) / 1e9
func run() -> Double {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso); e.setBuffer(X, offset: 0, index: 0); e.setBuffer(W, offset: 0, index: 1); e.setBuffer(o, offset: 0, index: 2)
    for _ in 0..<reps { e.dispatchThreadgroups(MTLSize(width: N / TN, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)) }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return cb.gpuEndTime - cb.gpuStartTime
}
_ = run()
print(String(format: "ready %.3f", Date().timeIntervalSince1970)); fflush(stdout)
let t0 = Date(); var w0 = Date(), ww = 0.0, wg = 0.0, rates: [Double] = [], gall = 0.0
while Date().timeIntervalSince(t0) < secs {
    let g = run(); ww += work; wg += g; gall += g
    let el = Date().timeIntervalSince(w0)
    if el >= 0.5 {
        rates.append(ww / wg)
        print(String(format: "%.3f GPU %@ %.2f TOPS busy %.0f wgbs %.1f", Date().timeIntervalSince1970, mode, ww / wg, 100 * wg / el, ww / work * wwork / wg)); fflush(stdout)
        w0 = Date(); ww = 0; wg = 0
    }
}
let s = rates.sorted()
let med: Double = s.isEmpty ? 0 : s[s.count / 2]
let wgbs: Double = med * 1e12 / (2.0 * Double(tok)) * Double(wb) / 8.0 / 1e9   // weight bytes per second at that rate
let mbw: Double = Double(wbytes) / 1048576.0
print(String(format: "GPU %@ TOK %d median %.2f TOPS, weights %.0f MB (%d bits) -> %.1f GB/s of weights", mode, tok, med, mbw, wb, wgbs))
