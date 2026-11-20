// gdn_rec occupancy probe (19 Nov 2026): the served recurrence kernel (engine source, T = 128) with grid (H, 4) for H
// heads; GPU ms per dispatch (median of 50). Flat time up to some H = all threadgroups resident at once (one wave); the
// step where it grows shows how many simdgroups a core holds for this kernel.   gdn_rec_scale KERNELS_DIR
import Foundation
import Metal
let dir = CommandLine.arguments[1]
let src = ["kernels.metal"].map { try! String(contentsOfFile: dir + "/" + $0, encoding: .utf8) }.joined()
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
let lib = try! dev.makeLibrary(source: src.replacingOccurrences(of: "#define T 64", with: "#define T 128"), options: opts)
let HM = 192, T = 128
func buf(_ n: Int, _ v: Float) -> MTLBuffer { let b = dev.makeBuffer(length: n * 4, options: .storageModeShared)!; let p = b.contents().bindMemory(to: Float.self, capacity: n); for i in 0..<n { p[i] = v * Float((i * 7919) % 101) / 101 }; return b }
let Q = buf(HM / 3 * T * 128, 0.1), K = buf(HM / 3 * T * 128, 0.1), V = buf(HM * T * 128, 0.5), G = buf(HM * T, 0.99), B = buf(HM * T, 0.5)
let S = buf(HM * 128 * 128, 0.01), O = buf(HM * T * 128, 0)
for name in ["gdn_rec", "gdn_rec_pf", "gdn_rec16"] {
    let p = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    var line = "\(name):"
    for H in [3, 6, 12, 24, 36, 48, 72, 96, 144, 192] {
        var ms: [Double] = []
        for it in 0..<60 {
            let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
            e.setComputePipelineState(p)
            for (i, b) in [Q, K, V, G, B, S, O].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.dispatchThreadgroups(MTLSize(width: H, height: 4, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            if it >= 10 { ms.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3) }
        }
        ms.sort(); line += String(format: "  H%d %.3f", H, ms[ms.count / 2])
    }
    print(line + "  (ms per dispatch; served H = 48)")
}
