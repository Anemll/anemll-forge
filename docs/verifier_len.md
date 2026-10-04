# Verifier block length: 8, 4 and 3 tokens on the M6 ANE

**Result: keep the 8-row verifier.** On the complete Qwen3.8-27B Core AI target with the faster graph, a 4-row or 3-row verifier is no faster than the 8-row one up to 32K context and only 3 to 6% faster at 64K. With measured draft acceptance, shorter verifiers lose 21 to 36% of decode throughput on real coding-agent traffic and 45 to 60% on the high-acceptance coding fixture.

The speculative verifier evaluates `T` rows per target call: the committed anchor token plus `T - 1` DFlash2 drafts. The release uses `T = 8`. All numbers here are from the Apple M6 (`Mac18,5`, macOS 27.0.1 26A434) on 3 October 2026, with the faster graph from [M6_COMPUTE_ACCELERATION_2026-10-03.md](research/M6_COMPUTE_ACCELERATION_2026-10-03.md) (`GDN_FAST=1`, 2048-wide verify attention tiles, V8 KV cache).

## How the shorter verifiers were built

The verify block is also the lazy-commit DeltaNet block (`P` pending rows), so a `T`-row verifier is a separate build with `P = T`, not a padded 8-row call. [`scripts/m6_verify_len.py`](../scripts/m6_verify_len.py) builds all 16 chunks with `v<T>_<ctx>k` verify entries (V8 KV; 8K, 16K, 32K and 64K) and a `T`-row head, from the same quantized export and numerics as the 8-row target. [tests/test_gdn_fast.py](../tests/test_gdn_fast.py) checks on the host that 8-, 4- and 3-row blocks with partial acceptances reproduce the token-by-token gated delta rule exactly.

Every package compiled fully onto the ANE. Timing: [`scripts/m6_entry_sweep.py`](../scripts/m6_entry_sweep.py) `--chain` runs all 16 chunks and the head as one plan, the same target call the server makes per speculative cycle. Random inputs, medians of 15 calls, idle machine.

## Measured verify forward (16 chunks + head)

The 8-row column is the default V8-only build; the 4- and 3-row builds are V8-only packages with verify entries only.

| Context entry | T = 8 | T = 4 | T = 3 |
| --- | ---: | ---: | ---: |
| 8K | 91.5 ms | 95.3 ms (+4.1%) | 94.5 ms (+3.3%) |
| 16K | 97.4 ms | 98.9 ms (+1.5%) | 97.5 ms (+0.1%) |
| 32K | 108.0 ms | 108.3 ms (+0.2%) | 105.8 ms (-2.1%) |
| 64K | 131.5 ms | 127.2 ms (-3.3%) | 123.1 ms (-6.4%) |

Least-squares fits `time = fixed + slope x context`: T = 8 is 85.8 ms + 0.710 ms/K, T = 4 is 90.1 ms + 0.577 ms/K, T = 3 is 89.7 ms + 0.517 ms/K.

Why: at 8 rows the forward is dominated by decoding LUT-compressed weights (MLP and projections), which costs the same for 3 rows as for 8. The Gated DeltaNet core no longer grows much with the block either: its within-block triangular solve, which took 7 serial row updates at 8 rows in the release graph, is now 3 product steps and one matmul ([why the gain comes from the 8-row block](research/M6_COMPUTE_ACCELERATION_2026-10-03.md#why-the-gain-comes-from-the-8-row-block)). Only attention over the KV history scales with rows, so fewer rows help a little at long context. The 8-row package also carries the prefill entries and differs in packaging, which plausibly explains why it is slightly faster at short context; the conclusion does not depend on it. An earlier comparison against a two-format 8-row package (99.4 / 104.2 / 115.4 / 140.2 ms) put the shorter verifiers 4 to 12% faster per call and reached the same result. For reference, the release target (before the faster graph) needed 113.1 ms at 8K and 190.3 ms at 64K for T = 8.

## Draft acceptance

From the owner's serving session log (the same target with DFlash2, V8 cache, real coding-agent traffic at the 8K to 64K entries): 58 completed requests, 32,504 speculative cycles, 107,829 generated tokens. The cycle-weighted share of 8-row cycles accepting exactly `k` of 7 drafts:

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| share | 30.9% | 20.0% | 13.0% | 9.1% | 6.4% | 4.7% | 3.7% | 12.4% |

Mean non-verify time per cycle in that session was 22.7 ms (drafter 16.7 including the 3 ms gap, sampling 3.8, host 1.8, context bookkeeping 0.3).

## Modeled decode throughput

Verification accepts the longest matching draft prefix, so a `T`-row verifier accepts `min(k, T - 1)` drafts where the 8-row verifier accepted `k`. Expected tokens per cycle are `1 + E[min(k, T - 1)]`: **3.31** for T = 8, **2.55** for T = 4 (-23%) and **2.18** for T = 3 (-34%).

Modeled decode rate = tokens per cycle / (measured verify ms + 22.7 ms), holding the drafter, sampling and host time fixed:

| Context entry | T = 8 | T = 4 | T = 3 |
| --- | ---: | ---: | ---: |
| 8K | 29.0 tok/s | 21.6 (-25.5%) | 18.6 (-35.8%) |
| 16K | 27.6 | 20.9 (-24.0%) | 18.2 (-34.1%) |
| 32K | 25.3 | 19.4 (-23.2%) | 17.0 (-32.9%) |
| 64K | 21.5 | 17.0 (-20.9%) | 15.0 (-30.2%) |

To break even with T = 8 under this acceptance, a 4-row verify would have to be about 28% cheaper than the 8-row one; at best it is 3% cheaper. Allowing 3 ms less sampling time for the shorter verifiers changes the losses by about one percentage point.

High-acceptance regime: the synthetic coding fixture of the full-server benchmark ([M6_COMPUTE_ACCELERATION_2026-10-03.md](research/M6_COMPUTE_ACCELERATION_2026-10-03.md#full-server-with-dflash2-measured)), 549 cycles at 8K to 64K, accepted all seven drafts in 76% of cycles (histogram 1.6 / 6.0 / 2.7 / 2.2 / 2.7 / 5.5 / 3.3 / 76.0% for k = 0 to 7), with 18.6 ms of non-verify time per cycle. Tokens per cycle: 7.08 (T = 8), 3.80 (T = 4), 2.91 (T = 3).

| Context entry | T = 8 | T = 4 | T = 3 |
| --- | ---: | ---: | ---: |
| 8K | 64.3 tok/s | 33.4 (-48.0%) | 25.7 (-60.0%) |
| 16K | 61.0 | 32.4 (-46.9%) | 25.0 (-58.9%) |
| 32K | 55.9 | 30.0 (-46.3%) | 23.4 (-58.2%) |
| 64K | 47.1 | 26.1 (-44.7%) | 20.5 (-56.5%) |

That fixture was measured in the server on the two-format build (55.1 / 57.6 / 54.1 / 45.8 tok/s at 8K / 16K / 32K / 64K). Modeled with that build's verify times, the T = 8 rates come within 3% at 16K to 64K and 8.5% at 8K, where the run's reply diverged and its acceptance was lower (6.4 tokens per cycle).

Scope: the tok/s figures are a model built from measured verify times and measured acceptance, not a server run with shorter verifiers. Running a shorter verifier in the server would also need a matching drafter block width and server changes; this measurement shows it is not worth doing for this target on the M6.

## Reproduce

```sh
CAI=<Core AI conversion venv>/bin/python
EXPORT_DIR=<export> $CAI scripts/m6_verify_len.py --T 4,3 --ctx 8192,16384,32768,65536 --out DIR   # builder defaults
for T in 4 3; do python scripts/m6_entry_sweep.py --build DIR/T$T --format v8 --chunks all --chain --out t$T.json; done
python scripts/m6_entry_sweep.py --build <default target build> --format v8 --chunks all --chain --entries v8 --out t8.json
python scripts/m6_verify_len_report.py --sweep 8=t8.json 4=t4.json 3=t3.json --accept session=<hist.json> --other-ms 22.7
```
