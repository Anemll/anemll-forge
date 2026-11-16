"""DeltaNet state handover on the GPU (16 Nov 2026): the host writes into the bridge's IOSurface state arrays cost
~170 ms per handover (CPU writes to IOSurface memory); gdn_state_out / gdn_state_in copy between the GPU engine's
state (S fp32 [48][v][k], CS fp32 [3][10240]) and the bridge's arrays (conv f16 [11][10240]: rows 0..2 the history,
the rest 0; rec f16 [48][k][v] x 64; pend f16 zeroed). C API v8_state_out / v8_state_in (n layers in one command
buffer, wrapped handles). Backups *.pre_state_gpu.*"""
from pathlib import Path
E = Path(__file__).resolve().parent
k = E / "kernels.metal"; s = k.read_text()
if "gdn_state_out" not in s:
    s += r'''
// ---- DeltaNet state <-> the bridge's arrays on the GPU (patch_state_gpu.py) ----
// n = (conv elements 11 x 10240, rec 48 x 128 x 128, pend elements); one thread per output element
kernel void gdn_state_out(device const float *S [[buffer(0)]], device const float *CS [[buffer(1)]], device half *conv [[buffer(2)]],
                          device half *rec [[buffer(3)]], device half *pend [[buffer(4)]], constant uint4 &n [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < n.x) { conv[i] = i < 3 * 10240 ? half(CS[i]) : half(0.0f); return; }
    uint j = i - n.x;
    if (j < n.y) { uint h = j / 16384, kk = (j / 128) % 128, v = j % 128; rec[j] = half(64.0f * S[(h * 128 + v) * 128 + kk]); return; }
    j -= n.y;
    if (j < n.z) pend[j] = half(0.0f);
}
kernel void gdn_state_in(device float *S [[buffer(0)]], device float *CS [[buffer(1)]], device const half *conv [[buffer(2)]],
                         device const half *rec [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < 3 * 10240) { CS[i] = float(conv[i]); return; }
    uint j = i - 3 * 10240;
    if (j < 48 * 16384) { uint h = j / 16384, v = (j / 128) % 128, kk = j % 128; S[j] = float(rec[(h * 128 + kk) * 128 + v]) / 64.0f; }
}
'''
    k.write_text(s); print("kernels patched")
w = E / "gpu_v8.swift"; s = w.read_text()
if "v8_state_out" not in s:
    s += r'''
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
'''
    w.write_text(s); print("swift patched")
