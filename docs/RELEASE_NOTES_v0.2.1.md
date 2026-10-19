# Release 0.2.1: 192K context and memory optimizations

19 October 2026. Model files: [anemll/anemll-forge-qwen3.8-27B-192K](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B-192K).

Release 0.2.1 runs the **release 0.2 weights** (`release_vq3pA_mixh_s600_k1_mat`, unchanged) with a context of up to **196,608 tokens (192K)** on the M6 Neural Engine, three times the 64K of the release 0.2 packages. Four memory optimizations make that fit on a 32 GB Mac. The quantization and the measured quality of the weights are those of [release 0.2](../README.md); what is new is the context length, the cache format, the drafters and the runtime that grows the context.

Release 0.2.1 is for **M6 only**. Up to 64K, the [release 0.2 packages](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B) remain faster (about 15% more prefill at short context) and also run on the M5 family.

## Memory optimizations

### 1. INT8 keys and values (`kv8`)

The 16 full-attention layers store every history position's keys and values as INT8, with FP16 scales per token and KV head. Release 0.2 kept the keys in FP16 (V8). The ANE multiplies each history tile's attention scores by the key scales, so it computes the same scores as from FP16 keys rounded to INT8.

| Cache format | Per history position | At 64K (65,472 rows) | At 192K (196,608 rows) |
| --- | ---: | ---: | ---: |
| FP16 keys and values | 64 KiB | 4.0 GiB | 12.0 GiB |
| V8: FP16 keys, INT8 values (release 0.2) | 48.1 KiB | 3.0 GiB | 9.0 GiB |
| **kv8: INT8 keys and values (release 0.2.1)** | **32.25 KiB** | **2.0 GiB** | **6.0 GiB** |

These are the logical cache sizes. Measured at 192K, the cache with its I/O buffers takes 6.8 GB of process memory. On the previous weights, `kv8` against V8 measured a direct KL of 6.6e-5 nats with the same top token at 99.75% of positions (KL-512, 64 sequences), and the same speed within 2%.

### 2. Drafters that use the target's output head

The DFlash2 drafter no longer carries its own copy of the 248,320-row output head. It scales its hidden state by the target head's norm gain (`head_normw_fp16.npy`) and runs it through the target's head, which is already loaded. Two drafters ship, both GPTQ-quantized from the same ProCreations checkpoint:

| Drafter | Weights | Neural Engine memory | Decode speed | Tokens per verify call |
| --- | --- | ---: | ---: | ---: |
| release 0.2 drafter (LUT4, own head copy) | four-bit LUT | 1.60 GB | 1.000 | 1.000 |
| **`all3` (default)** | every projection three-bit vector LUT | **0.74 GB** | 1.010 | 1.002 |
| `gu2a3` | gate and up two-bit, the rest three-bit | 0.63 GB | 1.008 | 0.989 (prose 6% lower) |

Speed and tokens per verify call relative to the release 0.2 drafter, from paired runs on the 192K target at contexts from 1K to 195K tokens and five content types. `all3` saves **0.86 GB** at the same speed.

### 3. Prefill rows per context entry

A chunk's scratch memory on the Neural Engine is about rows times its largest context (the attention scores), and Core AI reserves it for the chunk's largest function at the first call. 64-row prefill in every entry would have doubled the scratch at 192K (178 to 354 MB per chunk, **+2.8 GB**). Release 0.2.1 prefills 64 rows per call up to 96K and 32 rows at 128K and 192K: the 64-row entries fit in the scratch the 32-row 192K entry needs anyway. This costs **+0.75 GB** and makes prefill 11 to 17% faster below 128K.

### 4. One cache at a time, grown on demand

The runtime allocates only the current context entry's cache. It starts a conversation in the 8K entry, moves to the next entry when the context outgrows it (moving the kept rows), and releases the old cache. A new conversation goes back to the smallest entry that holds it. So memory follows the conversation, not the cap:

| Context | Entry | Process memory | of it KV cache and I/O |
| --- | --- | ---: | ---: |
| 1K to 8K | 8K | 1.5 to 1.9 GB | 0.58 GB |
| 16K | 16K | 2.2 GB | 0.86 GB |
| 32K | 32K | 2.8 GB | 1.40 GB |
| 64K | 64K | 3.8 GB | 2.48 GB |
| 96K | 96K | 4.9 GB | 3.56 GB |
| 128K | 128K | 6.0 GB | 4.64 GB |
| 160K to 195K | 192K | 8.2 GB | 6.81 GB |

While it moves from one entry to the next, both caches exist briefly: the move from 128K to 192K peaks at 12.5 GB of process memory.

### Neural Engine total

The target packages take about **15.8 GB** of Neural Engine memory (weights plus scratch), against about 13.0 GB for release 0.2's 64K packages; the long entries need the extra scratch. With the `all3` drafter the total is about 16.5 GB, about 2 GB more than release 0.2 for three times the context. This does not depend on `CTX`: Core AI loads every function of a chunk and reserves its scratch at the first call.

## Longer context

- **Context entries:** 8K, 16K, 32K, 64K, 96K, 128K and 192K. Release 0.2 had 8K to 64K, with a 48K entry.
- **Tiled history attention above 64K.** The ANE limits one tensor axis to 65,536, so the 64K entry holds 65,472 history rows. Entries above 64K attend over the history in tiles without forming that tensor, and hold their full length.
- **Recall:** six codes planted at depths 0 to 185K were recalled correctly at every context size up to 195K tokens.

Speed on M6 (macOS 27.0.1, 32 GB, `all3` drafter). Prefill in tokens per second for the context entry; decode in generated tokens per second by content type:

| Context (prompt tokens) | Prefill rows | Prefill | Code | Prose | Explain | JSON | Thinking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K (1,000) | 64 | 260 | 28.0 | 17.5 | 29.2 | 62.1 | 40.4 |
| 8K (7,000) | 64 | 260 | 26.7 | 16.6 | 28.1 | 56.8 | 38.8 |
| 16K (15,000) | 64 | 235 | 32.2 | 17.2 | 23.1 | 58.4 | 39.0 |
| 32K (31,500) | 64 | 205 | 24.6 | 14.2 | 22.5 | 58.0 | 40.8 |
| 64K (64,000) | 64 | 160 | 28.8 | 15.6 | 23.7 | 53.1 | 36.7 |
| 96K (97,000) | 64 | 123 | 26.0 | 13.7 | 19.4 | 45.8 | 32.0 |
| 128K (129,500) | 32 | 95 | 23.9 | 11.9 | 18.7 | 41.6 | 29.3 |
| 160K (160,000) | 32 | 62 | 19.5 | 10.1 | 13.8 | 34.5 | 22.4 |
| 192K (195,000) | 32 | 62 | 18.4 | 10.0 | 14.7 | 35.5 | 24.8 |

Decode: 256 generated tokens, greedy except thinking (temperature 1.0). Time to first token for a cold prompt: about 5 min 42 s at 64K and 33 min at 195K. Full method and per-entry data: [the 192K sweep](research/V02_192K_SWEEP_2026-10-18.md).

## How to use

These steps assume the inference environment from the [README](../README.md#1-set-up-the-inference-environment), with Forge source from 19 October 2026 or later (body-only drafters and per-context prefill rows).

**1. Download** (about 15 GB with one drafter):

```sh
export FORGE_192K="$HOME/Models/anemll-forge-qwen3.8-27B-192K"
hf download anemll/anemll-forge-qwen3.8-27B-192K \
  --local-dir "$FORGE_192K" --exclude "drafter/gu2a3/*"
```

Leave out `--exclude` to get both drafters.

**2. Compile for the ANE, once** (about 2 h 11 min on M6; needs about 23 GB more disk for the compile caches):

```sh
python forge.py compile --build "$FORGE_192K/coreai/m6-192k" \
  --draft "$FORGE_192K/drafter/all3/dflash2_lut4_gptq_nohead.aimodel"
```

It prints the plan and a line per package with the time left. Ctrl-C stops it safely: finished packages stay cached, and running it again (or starting the server) compiles the rest. Every macOS update, minor ones included, compiles again. Shipping precompiled `.aimodelc` files would not avoid this: most of the time is the system's Neural Engine compiler, which runs on every Mac.

**3. Start the server** with a context cap (stop a running server first with `bash scripts/qwen38_server.sh stop`):

```sh
MODEL="$FORGE_192K/model" BUILD="$FORGE_192K/coreai/m6-192k" \
DRAFT="$FORGE_192K/drafter/all3/dflash2_lut4_gptq_nohead.aimodel" \
DRAFTER="$FORGE_192K/drafter/all3" \
CTX=128K bash scripts/qwen38_server.sh start
```

The server listens on `http://127.0.0.1:8765/v1` (OpenAI-compatible) and updates Pi's model entry for the cap. For the smaller drafter, point `DRAFT` and `DRAFTER` at `drafter/gu2a3`. Check it with `curl -s http://127.0.0.1:8765/health`.

**Choosing `CTX`:** 64K, 96K, 128K or 192K. The cap bounds how far the cache grows and sets Pi's window; the packages and their Neural Engine memory are the same for every cap.

| CTX | Pi compacts above | Top entry: prefill / decode, mean of five tasks | Process memory at the top |
| --- | ---: | --- | ---: |
| 64K | about 45K | 160 / 31.6 tok/s | 3.8 GB |
| 96K | about 78K | 123 / 27.4 tok/s | 4.9 GB |
| 128K | about 111K | 95 / 25.1 tok/s | 6.0 GB |
| 192K | about 176K | 62 / 20.6 tok/s | 8.2 GB |

On a 32 GB Mac, 128K is the comfortable maximum: a prompt just past 128K runs in the slower 192K entry, and the move into it added 1.3 GB of swap.

## Requirements and limits

- M6 Mac with 32 GB of memory, macOS 27 (measured on 27.0.1). No M5 packages yet: these use FP8 attention, which the M5 ANE compiler rejects.
- Disk: about 15 GB of download, 10.5 GB of Core AI compile cache and about 12 GB of compiled programs in the system's ANE cache.
- Prefill at short context is about 15% below the release 0.2 packages; the cause is being investigated.
- Long-context quality was checked by recall only; KL and perplexity were measured on the release 0.2 packages, up to 64K.

## Source changes

- **Body-only drafters** (`scripts/dflash2_coreai_drafter.py`, `scripts/qwen38_server.py`, `scripts/hf_release.py`): a drafter package without an LM head drafts through the target's head, given `head_normw_fp16.npy` beside its configuration.
- **Prefill rows per context entry** (`coreai/qwen38_coreai_build.py`, `scripts/qwen38_coreai_model.py`, `scripts/qwen38_kv_cache.py`): builder options `TPS_LONG`, `PREFILL_LONG_FROM` and `ATT_BLOCK_PREFILL_LONG`; the manifest records the rows of each entry (`ptp`); the runtime prefills each entry with its own rows, also across entry switches.
- **Server wrapper** (`scripts/qwen38_server.sh`): `CTX` accepts every context of the build, up to 192K.
