# Core ML timing utility

`time_models.swift` is copied unchanged from `coremltools/examples/fp8/`; its Apple copyright and BSD-3-Clause notice and adjacent `LICENSE.txt` are preserved. Source hashes are recorded in `../../provenance/LOCAL_EXPERIMENT_IMPORTS.json`.

From the Forge root on a compatible macOS SDK:

```sh
swiftc -O -parse-as-library tools/coreml/time_models.swift -o /tmp/forge-time-models
/tmp/forge-time-models --units ane,gpu --iters 50 --rounds 5 /path/to/models/*.mlmodelc
```

This uses public Core ML prediction and placement APIs; it does not reproduce the historical private `ane_mil_bench` execution path. It requires compiled model artifacts, not a sibling source checkout. The local Swift source was checked with the current compiler during localization; no model execution was performed.
