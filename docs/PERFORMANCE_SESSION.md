# Two-hour serving session: reported performance

Source: owner-supplied session analysis received 2026-09-29, describing `qwen38_server.log`, 201 runs and approximately 9K–47K contexts. The raw log, exact run interval, per-run records, software/configuration hashes, and aggregation code were not supplied with this analysis. These are **reported session summaries**, not measurements rerun for this port.

## Decode observations

The reported verifier processes T=8 candidate rows per forward. Emitted tokens per speculative cycle depend on acceptance; T=8 is not eight guaranteed output tokens. Reported mean/peak/minimum generation rates:

- 0–16K bucket: 21.5 / 31.0 / 15.7 tokens/s. Despite its label, the supplied session's observed context range begins near 9K.
- 16–24K: 20.8 / 44.4 / 13.3 tokens/s.
- 24–32K: 18.8 / 37.4 / 12.4 tokens/s.
- 32–48K: 16.4 / 28.2 / 11.2 tokens/s.
- Overall: reported mean 18.7, peak 44.4, minimum 11.2 tokens/s. The weighting used for the mean is not specified.

The bucket means decline by **23.7%**, from 21.5 to 16.4. Verifier medians at approximate 12K, 21K, 30K and 39K contexts are 133.8, 135.6, 154.7 and 181.3 ms respectively—a **35.5%** increase from the first to last sample. Fastest forwards in those groups were 123.7, 125.8, 133.6 and 149.8 ms. Bucket aggregation and these representative contexts are not necessarily the same grouping.

## Traffic model and units

The detailed source calculation uses **10.61 GB including chunks and head per verifier forward**, plus a modeled KV read:

```text
KV bytes per history token = 16 full-attention layers × 4 KV heads × 256 dimensions
                            × 2 (K and V) × 2 bytes = 65,536 bytes = 64 KiB
modeled verifier bandwidth = (10.61e9 + history_tokens × 65,536) / forward_seconds
weight-only bandwidth     = 10.61e9 / forward_seconds
```

Reported median weight-only rates are 79.3, 78.3, 68.6 and 58.5 GB/s. Including the source's estimated KV traffic, rates are 85.3, 88.4, 81.1 and 72.8 GB/s; fastest-forward estimates are 92.3, 93.0, 91.6 and 85.2 GB/s. The exact context counts are unavailable, so the KV-inclusive arithmetic cannot be independently reproduced from the rounded labels.

The pasted summary later describes “~10.6 GB verify weights + 0.64 GB head,” which conflicts with the earlier **10.61 GB total**. Resolve this against actual artifacts before publication. The earlier figures imply approximately 9.98 GB of chunks plus 0.63 GB head. Do not add the head twice. Use decimal GB for bandwidth and distinguish it from GiB when reporting package sizes.

This is an **effective traffic estimate**, not measured DRAM bandwidth: it assumes one full weight read and one KV read per verifier forward and omits activation traffic, writes, caches and possible rereads. Package size may also differ from runtime traffic. Consequently ~92–93 GB/s is the peak of this model in this session, not a demonstrated hardware bandwidth ceiling.

## Prefill observations

For TP=64, reported cold-prefill rates at 16K, 24K, 32K and 48K are approximately 190, 172, 162 and 151 tokens/s. Using 9.98 GB of chunk weights per call gives weight-only effective rates of about 29.6, 26.8, 25.3 and 23.5 GB/s. The summary separately rounds its upper prefill range to 195 tokens/s; retain the per-context values above when comparing buckets.

These estimates exclude KV/activation traffic and should not be directly compared to KV-inclusive verifier rates. “Cold” also needs a precise definition (new conversation versus cold model/cache). The reported continuation range of ~141 to ~73 tokens/s lacks a per-context breakdown.

## Interpretation and next experiments

The evidence is consistent with a growing attention/KV cost, while nominal weight size stays fixed. It does **not** establish that the decline is entirely due to KV streaming: attention compute, placement, memory reuse, stalls and thermal/load effects could also contribute. Lower modeled prefill bandwidth with larger T is compatible with a more compute-heavy regime, but does not prove a compute bottleneck without utilization evidence or a controlled batch sweep.

Speculative acceptance likely contributes to the large throughput spread. To quantify it, retain accepted tokens per cycle together with verifier, drafter, sampling and host times for each run. Context and acceptance alone should not be asserted to explain all variance.

Next measurements should:

1. Reconcile chunk/head byte counts, actual KV rows versus configured entry capacity, and decimal/binary units.
2. Export per-run and per-cycle records with prompt length, context entry, output tokens, seed, sampling settings, model/export hashes and acceptance statistics.
3. Compare token-weighted throughput (`total emitted / total generation time`) with the arithmetic mean of per-run rates; report median and p90/p99 latency separately from maxima/minima.
4. Sweep verifier/prefill batch sizes with identical weights and inputs while checking placement, memory and numerical parity. Hold context and acceptance workload fixed where possible.
5. Repeat interleaved context measurements to distinguish attention cost from load, temperature and run-order effects; collect hardware counters where supported.

This session is useful as an observed user-workload baseline. It should remain separate from isolated microbenchmarks and controlled quality/throughput comparisons in the historical notebooks.

## Release scope

This historical session describes speculative target-plus-drafter serving. The intended fast release retains the matching Core AI DFlash2 drafter. A plain target-only benchmark cannot reproduce these serving timings. Record the paired artifacts, acceptance, context ladder, warm-up, prefill and serving policies using [SPECULATIVE_DECODING.md](SPECULATIVE_DECODING.md); no new benchmark is claimed here.
