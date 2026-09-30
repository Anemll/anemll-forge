//  Copyright (c) 2026, Apple Inc. All rights reserved.
//
//  Use of this source code is governed by a BSD-3-clause license that can be
//  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

// Times compiled Core ML models (.mlmodelc) with plain Core ML on the Neural Engine, GPU and CPU,
// and prints where their ops run. One fixed input per model; the best median of --iters
// predictions over --rounds interleaved rounds.
//
//     swiftc -O time_models.swift -o time_models
//     ./time_models --units ane,gpu --iters 50 --rounds 5 [--fast-prediction] models/*.mlmodelc

import CoreML
import Foundation

func deviceName(_ device: MLComputeDevice?) -> String {
    switch device {
    case .some(.neuralEngine): return "ANE"
    case .some(.gpu): return "GPU"
    case .some(.cpu): return "CPU"
    default: return "-"
    }
}

func fixedInputs(_ model: MLModel) throws -> MLFeatureProvider {
    var features: [String: MLFeatureValue] = [:]
    for (name, description) in model.modelDescription.inputDescriptionsByName {
        guard let constraint = description.multiArrayConstraint else { continue }
        let array = try MLMultiArray(shape: constraint.shape, dataType: constraint.dataType)
        switch array.dataType {
        case .float32:
            let p = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
            for i in 0..<array.count { p[i] = Float((i * 7919) % 2001) / 1000 - 1 }
        case .float16:
            let p = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
            for i in 0..<array.count { p[i] = Float16(Float((i * 7919) % 2001) / 1000 - 1) }
        default:
            break  // zeros
        }
        features[name] = MLFeatureValue(multiArray: array)
    }
    return try MLDictionaryFeatureProvider(dictionary: features)
}

func placement(_ url: URL, _ units: MLComputeUnits) async throws -> String {
    let config = MLModelConfiguration()
    config.computeUnits = units
    let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
    guard case let .program(program) = plan.modelStructure, let main = program.functions["main"] else {
        return "?"
    }
    var counts: [String: Int] = [:]
    for op in main.block.operations where op.operatorName != "const" {
        guard let usage = plan.deviceUsage(for: op) else { continue }
        counts[deviceName(usage.preferred), default: 0] += 1
    }
    return counts.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value)" }.joined(separator: ",")
}

@main
struct TimeModels {
    static func main() async throws {
        var units = ["ane", "gpu"]
        var iters = 50
        var rounds = 5
        var fastPrediction = false
        var paths: [String] = []
        var args = CommandLine.arguments.dropFirst().makeIterator()
        while let arg = args.next() {
            switch arg {
            case "--units": units = (args.next() ?? "").split(separator: ",").map(String.init)
            case "--iters": iters = Int(args.next() ?? "") ?? iters
            case "--rounds": rounds = Int(args.next() ?? "") ?? rounds
            case "--fast-prediction": fastPrediction = true
            default: paths.append(arg)
            }
        }
        let unitMap: [String: MLComputeUnits] = [
            "ane": .cpuAndNeuralEngine, "gpu": .cpuAndGPU, "cpu": .cpuOnly, "all": .all,
        ]
        for unit in units {
            guard let computeUnits = unitMap[unit] else { continue }
            // Load everything first, then time the models interleaved over several rounds and keep
            // each model's best round median, so background load on the machine evens out.
            var models: [(name: String, model: MLModel, input: MLFeatureProvider, placement: String)] = []
            for path in paths {
                let url = URL(fileURLWithPath: path)
                let config = MLModelConfiguration()
                config.computeUnits = computeUnits
                if fastPrediction {
                    // Let Core ML spend more time specializing the model for faster predictions.
                    var hints = MLOptimizationHints()
                    hints.specializationStrategy = .fastPrediction
                    config.optimizationHints = hints
                }
                let model = try MLModel(contentsOf: url, configuration: config)
                let input = try fixedInputs(model)
                for _ in 0..<5 { _ = try await model.prediction(from: input) }
                models.append((url.deletingPathExtension().lastPathComponent, model, input,
                               try await placement(url, computeUnits)))
            }
            var best = [Double](repeating: .infinity, count: models.count)
            var fastest = [Double](repeating: .infinity, count: models.count)
            for _ in 0..<rounds {
                for (i, entry) in models.enumerated() {
                    var times: [Double] = []
                    for _ in 0..<iters {
                        let start = DispatchTime.now().uptimeNanoseconds
                        _ = try await entry.model.prediction(from: entry.input)
                        times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
                    }
                    times.sort()
                    best[i] = min(best[i], times[times.count / 2])
                    fastest[i] = min(fastest[i], times[0])
                }
            }
            for (i, entry) in models.enumerated() {
                print(String(format: "%-40@ %-4@ median %8.3f ms  min %8.3f ms  %@",
                             entry.name, unit, best[i], fastest[i], entry.placement))
            }
        }
    }
}
