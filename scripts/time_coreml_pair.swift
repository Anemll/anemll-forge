// Time matched S=8 and S=16 compiled Core ML models; report the per-layer slope.
// Usage: time_coreml_pair <name> <S8.mlmodelc> <S16.mlmodelc> <channels>
//        time_coreml_pair --single <name> <model.mlmodelc> <channels>
import CoreML
import Foundation

func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    return (sorted[(sorted.count - 1) / 2] + sorted[sorted.count / 2]) / 2
}

func model(_ path: String) throws -> MLModel {
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    return try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: config)
}

func checkOutput(_ output: MLFeatureProvider) throws {
    guard let name = output.featureNames.first,
          let array = output.featureValue(for: name)?.multiArrayValue,
          array.count > 0,
          Double(truncating: array[0]).isFinite else {
        throw NSError(domain: "time_coreml_pair", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Missing or invalid model output"])
    }
}

func makeInput(channels: Int) throws -> MLFeatureProvider {
    let array = try MLMultiArray(shape: [1, channels, 4, 4].map(NSNumber.init(value:)),
                                 dataType: .float16)
    for i in 0..<array.count {
        let centered = (i * 73) % 257 - 128
        array[i] = NSNumber(value: Float(centered) / 256)
    }
    return try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: array)])
}

func timedCalls(_ model: MLModel, _ input: MLFeatureProvider, count: Int) throws -> Double {
    var samples = [Double]()
    samples.reserveCapacity(count)
    for _ in 0..<count {
        let start = DispatchTime.now().uptimeNanoseconds
        let output = try model.prediction(from: input)
        let end = DispatchTime.now().uptimeNanoseconds
        try checkOutput(output)
        samples.append(Double(end - start) / 1_000_000)
    }
    return median(samples)
}

if CommandLine.arguments.count == 5 && CommandLine.arguments[1] == "--single" {
    guard let channels = Int(CommandLine.arguments[4]), channels > 0 else {
        fputs("Expected a positive channel count\n", stderr)
        exit(2)
    }
    do {
        let name = CommandLine.arguments[2]
        let one = try model(CommandLine.arguments[3])
        let input = try makeInput(channels: channels)
        for _ in 0..<10 { try checkOutput(one.prediction(from: input)) }
        let rounds = try (0..<5).map { _ in try timedCalls(one, input, count: 50) }
        print(String(format: "%@\t%.6f\t%@", name, median(rounds),
                     rounds.map { String(format: "%.6f", $0) }.joined(separator: ",")))
        exit(0)
    } catch {
        fputs("Error: \(error)\n", stderr)
        exit(1)
    }
}

guard CommandLine.arguments.count == 5,
      let channels = Int(CommandLine.arguments[4]), channels > 0 else {
    fputs("Usage: time_coreml_pair <name> <S8.mlmodelc> <S16.mlmodelc> <channels>\n", stderr)
    exit(2)
}

do {
    let name = CommandLine.arguments[1]
    let small = try model(CommandLine.arguments[2])
    let large = try model(CommandLine.arguments[3])
    let input = try makeInput(channels: channels)
    for _ in 0..<10 {
        try checkOutput(small.prediction(from: input))
        try checkOutput(large.prediction(from: input))
    }
    var s8 = [Double]()
    var s16 = [Double]()
    for round in 0..<5 {
        if round.isMultiple(of: 2) {
            s8.append(try timedCalls(small, input, count: 50))
            s16.append(try timedCalls(large, input, count: 50))
        } else {
            s16.append(try timedCalls(large, input, count: 50))
            s8.append(try timedCalls(small, input, count: 50))
        }
    }
    let t8 = median(s8)
    let t16 = median(s16)
    let slope = (t16 - t8) / 8
    print(String(format: "%@\t%.6f\t%.6f\t%.6f\t%@\t%@", name, t8, t16, slope,
                 s8.map { String(format: "%.6f", $0) }.joined(separator: ","),
                 s16.map { String(format: "%.6f", $0) }.joined(separator: ",")))
} catch {
    fputs("Error: \(error)\n", stderr)
    exit(1)
}
