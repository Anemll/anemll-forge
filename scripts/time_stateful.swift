// Time compiled Core ML models on the ANE, including stateful ones (MLState: KV caches, recurrent states).
// Every float input is filled with a deterministic pattern; integer inputs are zero. Models are timed
// interleaved over several rounds; the best round median per model is reported.
//   swiftc -O -parse-as-library time_stateful.swift -o time_stateful
//   ./time_stateful --iters 30 --rounds 5 a.mlmodelc b.mlmodelc
// --iosurface: fp16 inputs and output backings are IOSurface-backed (CVPixelBuffer) arrays, the zero-copy path for
// tensors that are passed from call to call (e.g. recurrent states as I/O).
import CoreML
import CoreVideo
import Foundation

func surfaceArray(_ shape: [NSNumber]) throws -> MLMultiArray {
    let w = shape.last!.intValue, h = shape.dropLast().reduce(1) { $0 * $1.intValue }
    var pb: CVPixelBuffer?
    let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
    guard CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_OneComponent16Half, attrs, &pb) == kCVReturnSuccess,
          let buf = pb else { throw NSError(domain: "pb", code: 1) }
    return MLMultiArray(pixelBuffer: buf, shape: shape)
}

var useSurfaces = false

func inputs(_ model: MLModel) throws -> MLFeatureProvider {
    var features: [String: MLFeatureValue] = [:]
    for (name, d) in model.modelDescription.inputDescriptionsByName {
        guard let c = d.multiArrayConstraint else { continue }
        let a = (useSurfaces && c.dataType == .float16) ? try surfaceArray(c.shape) : try MLMultiArray(shape: c.shape, dataType: c.dataType)
        if a.dataType == .float16 && useSurfaces {
            CVPixelBufferLockBaseAddress(a.pixelBuffer!, [])
            let base = CVPixelBufferGetBaseAddress(a.pixelBuffer!)!.bindMemory(to: Float16.self, capacity: 1)
            let rowElems = CVPixelBufferGetBytesPerRow(a.pixelBuffer!) / 2
            let w = c.shape.last!.intValue, h = a.count / w
            for r in 0..<h { for i in 0..<w { base[r * rowElems + i] = Float16(Float(((r * w + i) * 7919) % 2001) / 4000 - 0.25) } }
            CVPixelBufferUnlockBaseAddress(a.pixelBuffer!, [])
        } else if a.dataType == .float16 {
            let p = a.dataPointer.bindMemory(to: Float16.self, capacity: a.count)
            for i in 0..<a.count { p[i] = Float16(Float((i * 7919) % 2001) / 4000 - 0.25) }
        } else if a.dataType == .float32 {
            let p = a.dataPointer.bindMemory(to: Float.self, capacity: a.count)
            for i in 0..<a.count { p[i] = Float((i * 7919) % 2001) / 4000 - 0.25 }
        } else if a.dataType == .int32 {
            let p = a.dataPointer.bindMemory(to: Int32.self, capacity: a.count)
            for i in 0..<a.count { p[i] = 0 }
        }
        features[name] = MLFeatureValue(multiArray: a)
    }
    return try MLDictionaryFeatureProvider(dictionary: features)
}

func median(_ v: [Double]) -> Double { let s = v.sorted(); return (s[(s.count - 1) / 2] + s[s.count / 2]) / 2 }

@main
struct TimeStateful {
    static func main() async throws {
        var iters = 30, rounds = 5
        var paths: [String] = []
        var args = CommandLine.arguments.dropFirst().makeIterator()
        while let a = args.next() {
            switch a {
            case "--iters": iters = Int(args.next() ?? "") ?? iters
            case "--rounds": rounds = Int(args.next() ?? "") ?? rounds
            case "--iosurface": useSurfaces = true
            default: paths.append(a)
            }
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        var models: [(String, MLModel, MLFeatureProvider, MLState?, MLPredictionOptions)] = []
        for p in paths {
            let url = URL(fileURLWithPath: p)
            let m = try MLModel(contentsOf: url, configuration: config)
            let state = m.modelDescription.stateDescriptionsByName.isEmpty ? nil : m.makeState()
            let x = try inputs(m)
            let opts = MLPredictionOptions()
            if useSurfaces {
                var backings: [String: Any] = [:]
                for (name, d) in m.modelDescription.outputDescriptionsByName {
                    if let c = d.multiArrayConstraint, c.dataType == .float16 { backings[name] = try surfaceArray(c.shape) }
                }
                opts.outputBackings = backings
            }
            for _ in 0..<5 {
                if let s = state { _ = try await m.prediction(from: x, using: s, options: opts) } else { _ = try await m.prediction(from: x, options: opts) }
            }
            models.append((url.deletingPathExtension().lastPathComponent, m, x, state, opts))
        }
        var best = [Double](repeating: .infinity, count: models.count)
        for _ in 0..<rounds {
            for (i, (_, m, x, s, o)) in models.enumerated() {
                var t: [Double] = []
                for _ in 0..<iters {
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    if let s = s { _ = try await m.prediction(from: x, using: s, options: o) } else { _ = try await m.prediction(from: x, options: o) }
                    t.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
                }
                best[i] = min(best[i], median(t))
            }
        }
        for (i, (name, _, _, s, _)) in models.enumerated() {
            print(String(format: "%@  %.3f ms%@", name, best[i], s == nil ? "" : "  (stateful)"))
        }
    }
}
