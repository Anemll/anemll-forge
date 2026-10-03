# NIAH retrieval pilot, 3 October 2026

This is a **pilot report**, not an official RULER score or a general long-context capability result. The operator reported eight runs between about 6:03 and 7:08 AM Pacific: one FP16-KV and one V8 run at each of four needle-in-a-haystack placements. All eight were reported as exact matches. The target was reported as the mix25 selectable bundle, with commit `6e0a85c` pinned.

The raw run files were unavailable for inspection when this note was prepared. The results below are transcribed from the operator's summary; prompt construction, output strings, request traces, server timers, hardware details and package hashes have not been independently checked here.

| Context / needle depth | Prompt tokens | FP16 prefill tok/s | V8 prefill tok/s | V8 change |
| --- | ---: | ---: | ---: | ---: |
| 8K / 50% | 7,642 | 208.85 | 221.90 | +6.25% |
| 64K / 10% | 64,946 | 102.48 | 133.58 | +30.35% |
| 64K / 50% | 64,917 | 103.02 | 133.05 | +29.15% |
| 64K / 90% | 64,925 | 107.62 | 132.17 | +22.81% |

The three displayed 64K rows average **104.37 → 132.93 tokens/s**, a **27.36%** increase in V8 prefill throughput. The operator's prose summary gave 104.32 tokens/s for FP16; that differs slightly from the mean of the supplied per-case figures. This note uses the figures in the table until the raw records resolve the difference.

These are cold prefills with one trial per case. They show exact retrieval in these reported placements, but do not establish repeatability, a confidence interval, performance across other prompts or depths, or an official benchmark score. Prefill throughput is a server timing measure, separate from output accuracy and end-to-end request latency.
