import CoreAI
import Foundation
let url = URL(fileURLWithPath: CommandLine.arguments[1])
let opts: SpecializationOptions = CommandLine.arguments.count > 2 && CommandLine.arguments[2] == "default" ? .default : SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
print("loading \(url.lastPathComponent) with \(opts.preferredComputeUnitKind.map { "\($0)" } ?? "default")")
let m = try await AIModel(contentsOf: url, options: opts)
print("functions", m.functionNames)
let f = try m.loadFunction(named: m.functionNames[0])!
var ins: [String: NDArray] = [:]
for n in f.descriptor.inputNames { if case .ndArray(let d) = f.descriptor.inputDescriptor(of: n)! { ins[n] = NDArray(descriptor: d) } }
let o = try await f.run(inputs: ins)
print("swift call ok:", Array(o.names))
