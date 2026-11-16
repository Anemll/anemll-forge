"""Server support (16 Nov 2026): v8_set_state, the inverse of v8_state: a DeltaNet layer's state from the ANE's layout
(conv rows [3][10240] f16, rec [48][dk][dv] f16 x GDN_SV 64) into the GPU engine, so a V8 prefill can continue from a
bridge state (prompt-cache snapshot). Backups: *.pre_srv.*"""
from pathlib import Path
E = Path(__file__).resolve().parent
w = E / "gpu_v8.swift"; s = w.read_text()
if "v8_set_state" not in s:
    s += r'''
/// The inverse of v8_state: a DeltaNet layer's state from the ANE's layout into the GPU engine (continue from a bridge
/// state). conv [3][10240] f16, rec [48][dk][dv] f16 x 64.
@_cdecl("v8_set_state")
public func v8_set_state(_ layer: Int32, _ conv: UnsafePointer<Float16>, _ rec: UnsafePointer<Float16>) -> Int32 {
    guard let k = layerIndex[Int(layer)], layers[k].m.kind == "gdn" else { return -1 }
    let l = layers[k]
    let cs = l.CS!.contents().bindMemory(to: Float.self, capacity: 3 * 10240)
    for i in 0..<(3 * 10240) { cs[i] = Float(conv[i]) }
    let st = l.S!.contents().bindMemory(to: Float.self, capacity: 48 * 128 * 128)
    for h in 0..<48 { for kk in 0..<128 { for v in 0..<128 { st[(h * 128 + v) * 128 + kk] = Float(rec[(h * 128 + kk) * 128 + v]) / 64 } } }
    return 0
}
'''
    w.write_text(s); print("swift patched")
