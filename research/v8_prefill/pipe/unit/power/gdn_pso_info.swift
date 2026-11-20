// Pipeline limits of the served DeltaNet kernels (19 Nov 2026): the engine's Metal source (kernels.metal + fold + f8, T =
// 128) compiled as the engine does; per kernel the register-limited max threads per threadgroup, the static threadgroup
// memory and the SIMD width.   gdn_pso_info KERNELS_DIR
import Foundation
import Metal
let dir = CommandLine.arguments[1]
let src = ["kernels.metal", "kernels_fold.metal", "kernels_f8.metal"].map { try! String(contentsOfFile: dir + "/" + $0, encoding: .utf8) }.joined()
let dev = MTLCreateSystemDefaultDevice()!
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
let lib = try! dev.makeLibrary(source: src.replacingOccurrences(of: "#define T 64", with: "#define T 128"), options: opts)
print("device \(dev.name), maxThreadgroupMemoryLength \(dev.maxThreadgroupMemoryLength), maxThreadsPerThreadgroup \(dev.maxThreadsPerThreadgroup)")
for (name, lut, cin, acc) in [("lut_gemm32_i8t_k256", true, 5120, false), ("lut_gemm32_i8t_k256", true, 6144, true), ("lut_gemm32_i8t_m64k128", true, 5120, false), ("lut_gemm32_i8t_k256db", true, 5120, false),
                              ("gdn_rec", false, 0, false), ("gdn_rec16", false, 0, false), ("gdn_rec8", false, 0, false), ("gdn_rec_pf", false, 0, false),
                              ("gdn_norm_fwht4", false, 0, false), ("rms_fwht4", false, 0, false), ("gdn_conv_qkv", false, 0, false), ("dense_gemm", false, 0, true),
                              ("dense_splitk", false, 0, false), ("splitk_reduce", false, 0, false), ("quant8_blk", false, 0, false), ("gdn_gates", false, 0, false)] {
    let fc = MTLFunctionConstantValues()
    var b: UInt32 = 4, g: UInt32 = 8, c = UInt32(cin), a = acc, cd: UInt32 = 2
    if lut { fc.setConstantValue(&a, type: .bool, index: 3); fc.setConstantValue(&b, type: .uint, index: 0); fc.setConstantValue(&g, type: .uint, index: 1); fc.setConstantValue(&c, type: .uint, index: 2); fc.setConstantValue(&cd, type: .uint, index: 4) }
    if name == "dense_gemm" { fc.setConstantValue(&a, type: .bool, index: 3) }
    guard let f = try? lib.makeFunction(name: name, constantValues: fc), let p = try? dev.makeComputePipelineState(function: f) else { print("\(name): n/a"); continue }
    print(String(format: "%-24@ cin %5d  maxThreads/tg %4d  static tg mem %6d B  simd %d", name, cin, p.maxTotalThreadsPerThreadgroup, p.staticThreadgroupMemoryLength, p.threadExecutionWidth))
}
