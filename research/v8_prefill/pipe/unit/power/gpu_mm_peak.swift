// GPU matrix-unit peak by element type (18 Nov 2026, M6): Metal 4 tensor ops matmul2d (MPP) with every operand
// combination of interest. Each threadgroup computes C[TM x 64] = A[TM x K] B[K x 64] over K = 4096 in KB steps, A and
// B straight from device memory as contiguous tiles (packed formats need implied strides), all in cache (A: 8
// distinct tiles, B: one 4096 x 64 block), so this is the compute peak, not memory.
//   gpu_mm_peak TYPE TM SECONDS      TYPE: i8 f16 bf16 f8 f8h f4 h_i4 i8_i4 h_f8      TM: 32 64 128
import Foundation
import Metal

let a = CommandLine.arguments
let ty = a[1], tm = Int(a[2])!, secs = Double(a[3])!
// name: (left type, right type, destination type, left bits, right bits, kb)
let T: [String: (String, String, String, Int, Int, Int)] = [
    "i8": ("int8_t", "int8_t", "int32_t", 8, 8, 256),
    "f16": ("half", "half", "float", 16, 16, 128),
    "bf16": ("bfloat", "bfloat", "float", 16, 16, 128),
    "f8": ("metal_fp8_e4m3_format", "metal_fp8_e4m3_format", "float", 8, 8, 256),
    "f8h": ("metal_fp8_e4m3_format", "metal_fp8_e4m3_format", "half", 8, 8, 256),
    "f4": ("metal_fp4_e2m1_format", "metal_fp4_e2m1_format", "float", 4, 4, 256),
    "h_i4": ("half", "int4b_format", "float", 16, 4, 128),
    "i8_i4": ("int8_t", "int4b_format", "int32_t", 8, 4, 256),
    "h_f8": ("half", "metal_fp8_e4m3_format", "float", 16, 8, 128),
]
guard let (lt, rt, dt, lb, rb, kb) = T[ty] else { fatalError("type") }
let K = 4096, N = 64, NA = 8, NTG = 4096 * 128 / tm
let src = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#define TM \(tm)
#define KB \(kb)
#define K \(K)
#define NA \(NA)
kernel void mm(device uchar *A [[buffer(0)]], device uchar *B [[buffer(1)]], device int *o [[buffer(2)]],
               uint tg [[threadgroup_position_in_grid]]) {
    constexpr auto d = matmul2d_descriptor(TM, 64, KB, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    using TA = tensor<device \(lt), dextents<int32_t, 2>, tensor_inline>;
    using TB = tensor<device \(rt), dextents<int32_t, 2>, tensor_inline>;
    auto c = op.template get_destination_cooperative_tensor<TA, TB, \(dt)>();
    for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
    const uint a0 = (tg % NA) * (K / KB);
    for (uint ks = 0; ks < K / KB; ks++) {
        TA tA((typename TA::data_handle_type)(A + (a0 + ks) * (TM * KB * \(lb) / 8)), dextents<int32_t, 2>(KB, TM));
        TB tB((typename TB::data_handle_type)(B + ks * (KB * 64 * \(rb) / 8)), dextents<int32_t, 2>(64, KB));
        op.run(tA, tB, c);
    }
    float s = 0;
    for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) s += float(c[i]);
    if (s == 123456.0f) o[tg] = int(s);
}
"""
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: opts) } catch { print("GPU \(ty) TM \(tm): COMPILE FAILED: \(error)".prefix(600)); exit(0) }
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "mm")!)
let abytes = NA * (K / kb) * tm * kb * lb / 8, bbytes = K * N * rb / 8
let A = dev.makeBuffer(length: abytes, options: .storageModeShared)!, B = dev.makeBuffer(length: bbytes, options: .storageModeShared)!
let o = dev.makeBuffer(length: NTG * 4, options: .storageModeShared)!
// values = 1.0 (or 1) in each format: no NaN / Inf patterns
func fill(_ b: MTLBuffer, _ t: String) {
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
fill(A, lt); fill(B, rt)
let work = 2.0 * Double(tm) * Double(N) * Double(K) * Double(NTG) / 1e12
func run() -> Double {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso); e.setBuffer(A, offset: 0, index: 0); e.setBuffer(B, offset: 0, index: 1); e.setBuffer(o, offset: 0, index: 2)
    e.dispatchThreadgroups(MTLSize(width: NTG, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return cb.gpuEndTime - cb.gpuStartTime
}
_ = run()
var rates: [Double] = []; let t0 = Date()
while Date().timeIntervalSince(t0) < secs { rates.append(work / run()) }
rates.sort()
print(String(format: "GPU %-6@ TM %3d  %@ x %@ -> %@ : median %.2f T(FL)OPS, best %.2f  (%d calls)", ty, tm, lt, rt, dt,
             rates[rates.count / 2], rates.last!, rates.count))
