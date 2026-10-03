# M6 NIAH retrieval pilot, 3 October 2026

This is a **pilot report**, not an official RULER score or a general long-context capability result. On M6, eight runs completed between 6:03 and 7:08 AM PDT: one FP16-KV and one V8 run at each of four needle-in-a-haystack placements. All eight passed strict exact match, value retrieval and value-only formatting; paired replies and speculative acceptance histograms were identical. The paired modes used the same quantized Qwen3.8-27B mix25 selectable target and DFlash2 drafter, with source commit `6e0a85c1ab0fa0d2a3ecdea37585b6563d949516` pinned.

The M6 pilot's `protocol.json`, `verification.json`, `per-case-results.csv` and report were inspected for this note. These artifacts record paired rendered prompt token hashes, one random seed per case, zero cached prompt tokens, greedy decoding with thinking off, a 64-token output cap and a 3 ms draft gap. The queue ran all FP16 cases before all V8 cases, using a fresh server for each case. FP16 here describes the KV cache; the target weights were quantized in both modes. The source artifacts are under `/Users/anemll/Documents/Benchmarks/qwen38-niah-20261003/` **on M6** and are not included in this repository.

| Context / needle depth | Prompt tokens | FP16 → V8 cold prefill (s) | FP16 → V8 prefill (tok/s) | V8 change |
| --- | ---: | ---: | ---: | ---: |
| 8K / 50% | 7,642 | 36.59 → 34.44 | 208.85 → 221.90 | +6.25% |
| 64K / 10% | 64,946 | 633.76 → 486.21 | 102.48 → 133.58 | +30.35% |
| 64K / 50% | 64,917 | 630.14 → 487.90 | 103.02 → 133.05 | +29.15% |
| 64K / 90% | 64,925 | 603.27 → 491.21 | 107.62 → 132.17 | +22.81% |

Pooling prompt tokens divided by total prefill time across the three 64K cases gives **104.32 → 132.93 tokens/s**, a **27.42%** V8 increase. This is a token-and-time-weighted pooled rate, so it differs from the arithmetic mean of the three displayed FP16 rates (104.37 tokens/s). Model loading is excluded from prefill timing.

Each record had finite prefill and final logits. The verification report checked all eight records, four paired prompt hashes, tokenizer counts and insertion positions. Cached placement inspection found compile mode 2 and no GPU regions in the 18 selected packages per record; that does not independently prove physical bonded-cluster execution.

These are cold prefills with one trial per case, one synthetic keyed-value prompt per placement and no interleaved timing order. They show exact retrieval in these cases, but do not establish repeatability, a confidence interval, performance across other prompts or depths, or an official benchmark score. The 13–15-token replies are too short for a steady-state decode conclusion. Short-context FP16 and V8 also use different attention arithmetic, so the 8K pair is not an isolated V-quantization comparison. Neither mode was compared with the original BF16 model in this pilot.
