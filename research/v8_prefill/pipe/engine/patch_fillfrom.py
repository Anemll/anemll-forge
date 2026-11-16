"""K8V16 incremental V fill (16 Nov 2026): v8_kv8_v16_fill_from(from, upto) converts only the rows [from, upto) of the
ANE's INT8 V into the GPU's fp16 copy (v8_kv8_v16_fill(upto) rebuilt rows [0, upto) on every warm start, 60 to 400 ms
at 22K to 61K). Rows below `from` keep what the GPU wrote itself (its own fp16 V of earlier V8 runs, or an earlier
fill). The same kernel (kv8_v_to_v16) on a buffer offset: `from` is rounded down to a multiple of 64 (offsets stay
aligned). v8_v16_ptr(layer) returns the layer's fp16 V copy (shared storage) for the V8_FILL_CHECK comparison.
Backup: gpu_v8.pre_fillfrom.swift. Build: libgpu_v8_srv3.dylib (the running server keeps libgpu_v8_srv2.dylib)."""
from pathlib import Path

E = Path(__file__).resolve().parent
src = E / "gpu_v8.swift"
s = src.read_text()
assert "v8_kv8_v16_fill_from" not in s, "already applied"
bak = E / "gpu_v8.pre_fillfrom.swift"
if not bak.exists():
    bak.write_text(s)
anchor = '// ---- K8V16 (patch_k8v16.py): the GPU\'s fp16 V copy from the ANE\'s INT8 V (warm start) ----'
assert s.count(anchor) == 1
add = '''// ---- patch_fillfrom.py: the fill from a row (the rows below are the GPU's own, still valid) ----
/// Every attention layer's Vc rows [from, upto) from its registered ANE segments (from rounded down to 64). GPU ms.
@_cdecl("v8_kv8_v16_fill_from")
public func v8_kv8_v16_fill_from(_ from: Int32, _ upto: Int32) -> Double {
    guard K8V16 else { return -1 }
    let f0 = max(0, Int(from) / 64 * 64), u = Int(upto)
    if f0 >= u { return 0 }
    let cb = q.makeCommandBuffer()!, ce = cb.makeComputeCommandEncoder()!
    for l in layers where l.m.kind != "gdn" {
        guard let vc = l.Vc else { return -2 }
        for g in l.segs where g.start < u && g.start + g.rows > f0 {
            let r0 = max(f0, g.start) - g.start, n = min(g.start + g.rows, u) - g.start - r0   // segment-local rows
            if n <= 0 { continue }
            var prm = SIMD4<UInt32>(UInt32(Lmax), UInt32(g.start + r0), g.vh, g.vr), sh = g.sh
            ce.setComputePipelineState(pso("kv8_v_to_v16"))
            ce.setBuffer(g.v, offset: r0 * Int(g.vr), index: 0); ce.setBuffer(g.vs, offset: r0 * 2, index: 1)
            ce.setBuffer(vc, offset: 0, index: 2)
            ce.setBytes(&prm, length: 16, index: 3); ce.setBytes(&sh, length: 4, index: 4)
            ce.dispatchThreads(sz(256, n, 4), threadsPerThreadgroup: sz(256))
        }
    }
    ce.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return cb.error != nil ? -3 : (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}

/// Layer i's fp16 V copy (4 x Lmax x 256 half, shared storage), nil without one.
@_cdecl("v8_v16_ptr")
public func v8_v16_ptr(_ i: Int32) -> UnsafeMutableRawPointer? {
    guard Int(i) >= 0 && Int(i) < layers.count, let vc = layers[Int(i)].Vc else { return nil }
    return vc.contents()
}

'''
s = s.replace(anchor, add + anchor)
src.write_text(s)
print("patched", src)
