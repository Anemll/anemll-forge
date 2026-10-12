# oQe 4-bit vs Anemll ANE: card-protocol KL review

Apples-to-apples next-token KL of Jun Kim's updated oQe 4-bit MLX Qwen3.8-27B against Anemll's ANE u48 / Tier A candidate, **using only the evaluation the oQe card publishes**. This document does not invent measured KL. Fill the results table after an M5 Max run.

## What is being compared

| | oQe 4-bit MLX | Anemll ANE (local Tier A) |
| --- | --- | --- |
| Artifact | [`Jundot/Qwen3.8-27B-oQ4e-mtp`](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp) | `/Users/anemll/Models/anemll-forge/qwen38-27b-tierA` |
| Base model | `Qwen/Qwen3.8-27B` | same |
| Runtime on M5 Max | mlx-lm / oMLX 0.7.1 | Forge Core AI (ANE) |
| Published size | 15.806 GiB complete, **4.887 bits/weight** (LM 14.655 GiB; vision + MTP included) | **9.74 GiB** quantized weights + factors, ~**3.17 bits/weight** over 25.6B weights, embedding excluded |
| Announcement | [Improved oQe, 2026-10-12](https://x.com/jundotkim/status/2109557358028337547) | Forge research candidate `u48` / data600 |

The public Hub bundle [`anemll/anemll-forge-qwen3.8-27B`](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B) is **release 0.2**, not this u48 tree. Size matches; the weights do not.

Tweet (verbatim):

> All my oQe models have been updated with Improved oQe Quantization, available from oMLX 0.7.1. Compared with the previous oQe builds, KL divergence is 11.6–18.1% lower on code and 10.2–13.8% lower on prose, so please give them a try. The full model list is below.

Source: https://x.com/jundotkim/status/2109557358028337547 (2026-10-12).

## Card methodology (verbatim)

Source: https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp  
Raw README: https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp/raw/main/README.md

The card does **not** inline the evaluation text. It names the method, the corpora, the windowing, and two category scores:

> ## KL Divergence from the Original Model
>
> Mean next-token KL divergence from the original BF16 model on the oMLX benchmark corpora (lower is better). Each corpus is scored on 16 windows of 2,048 tokens spread across the text (32,768 tokens).
>
> | Corpus | Previous oQ4e (2026-08-17) | Updated oQ4e | Change |
> |---|---|---|---|
> | Code (Python) | 0.0143 | 0.0117 | -18.1% |
> | Novel (English) | 0.0696 | 0.0600 | -13.8% |

That is the whole published protocol. Categories the card reports: **Code (Python)** and **Novel (English)** only. No other splits.

What the card states, and what it leaves open:

| Stated | Not stated |
| --- | --- |
| Reference = original BF16 | Top-K / tail vs full vocabulary |
| Mean next-token KL | Natural vs log2 units (we use nats, `ln`) |
| oMLX benchmark corpora | The raw text (not pasted on the card) |
| 16 windows × 2,048 tokens, spread across the text | Exact start offsets |
| 32,768 tokens per corpus | Scored positions vs consumed tokens (2048 vs 2047 next-token steps) |
| Code vs Novel | Chat template, BOS, skip-first-N |

## Texts we use (the card's named corpora)

The labels **Code (Python)** and **Novel (English)** are oMLX's bundled throughput corpora, not chat prompts:

| Card label | oMLX file | What it is |
| --- | --- | --- |
| Code (Python) | `omlx/admin/bench_corpora/code_python.txt` | Apache-2.0 snapshot of oMLX production Python |
| Novel (English) | `omlx/admin/bench_corpora/novel_en.txt` | Project Gutenberg #2701 *Moby-Dick*, starting at `Call me Ishmael.` |

Pinned fetch: `https://raw.githubusercontent.com/jundot/omlx/34e9e985b753509e0172d31b99331419436e575f/omlx/admin/bench_corpora/` with SHA-256 from oMLX `manifest.json`. Novel English uses oMLX's runtime start marker. Attribution (verbatim from oMLX `ATTRIBUTION.md`):

> - `code_python.txt` is a snapshot of production Python source from oMLX, licensed under Apache-2.0.
> - `novel_en.txt` is Project Gutenberg eBook #2701, *Moby-Dick; or, The Whale* by Herman Melville, with Project Gutenberg boilerplate excluded at runtime.

The harness does not vendor the ~1.5 MiB files. `prepare` downloads them and checks the hashes.

**Windowing.** "Spread across the text" is implemented as 16 evenly spaced start offsets so the first window begins at token 0 and the last ends at the corpus end. Qwen3.6 token counts on these files are ~341k (code) and ~312k (novel), so the 16 × 2048 windows do not overlap. The card does not publish the offset formula; this is the reconstruction.

Each window is 2,048 tokens of raw corpus text (no chat template). Teacher-forced next-token KL scores the 2,047 positions that have a target. Report both **32,768 input tokens** (as the card writes) and **32,752 scored positions** per corpus.

## How both models are scored (same everything)

1. Tokenize each corpus **once** with the BF16 `Qwen/Qwen3.8-27B` tokenizer.
2. Slice the same 16 + 16 windows.
3. Run BF16 once; cache full-vocabulary next-token log-softmax at every scored position.
4. Teacher-force oQe (mlx-lm) on those token ids; KL vs the cached BF16 logits.
5. Teacher-force Anemll ANE on the **same** ids with the Forge Core AI T=8 loop (`eval/oqe_compare/ane_card_eval.py`, same `call` / `accept` path as `scripts/m6_kl512_eval.py`); KL vs the **same** cached BF16 logits.

Metric: \(D_{\mathrm{KL}}(p_{\mathrm{BF16}} \| p_{\mathrm{quant}})\) in nats, full vocabulary, positions equally weighted, mean per corpus. The card says "mean next-token KL" and does not mention a top-K tail, so the comparison column is full-vocab KL, not Forge KL-512.

## Results (empty until the M5 Max run)

Do not copy smoke-test numbers into this table. Cells stay blank until the 27B run writes `summary.json`.

| Corpus | Input tokens | Scored positions | oQe published (updated) | oQe this run | Anemll ANE this run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code (Python) | 32,768 | 32,752 | 0.0117 | | |
| Novel (English) | 32,768 | 32,752 | 0.0600 | | |

oQe published previous (2026-08-17), for context only: Code 0.0143, Novel 0.0696.

Size (published; measure-on-disk is filled by the script when the files are present):

| | Published GiB | Published bits/weight | Measured GiB (this machine) |
| --- | ---: | ---: | ---: |
| oQe 4-bit complete checkpoint | 15.806 | 4.887 | |
| oQe language backbone | 14.655 | — | |
| Anemll u48 weights + factors (no embedding) | 9.74 | ~3.17 | |

## Exact run command (M5 Max)

Needs Apple Silicon, mlx-lm or oMLX ≥ 0.7.1, Forge Core AI, and ~60 GiB free unified memory for BF16. Default paths: `/Users/anemll/Models/Qwen3.8-27B`, `/Users/anemll/Models/anemll-forge/qwen38-27b-tierA`, and `Jundot/Qwen3.8-27B-oQ4e-mtp` (or `/Users/anemll/Models/Qwen3.8-27B-oQ4e-mtp` if that directory exists).

```sh
# from the anemll-forge checkout
eval/oqe_compare/run_m5max.sh
```

Equivalent:

```sh
python3 eval/oqe_compare/run_compare.py --preset m5max \
  --out /Users/anemll/Models/anemll-forge/oqe_compare
```

Stages (resumable): `--stage prepare|reference|eval-oqe|eval-ane|summarize`.

CPU checks that do **not** produce comparison numbers:

```sh
python3 eval/oqe_compare/run_compare.py --dry-run --preset m5max --out /tmp/oqe-dry
python3 eval/oqe_compare/run_compare.py --smoke --out /tmp/oqe-smoke
python3 -m unittest tests.test_oqe_compare -v
```

After a real run, `summary.json` → `comparison[]` holds the two category means. Copy those into the table above. Leave a cell blank if that backend did not finish.

## Apples-to-apples caveats

- **Bits/weight and size are not matched.** oQe is a mixed 4/5-bit MLX checkpoint at 4.887 bpw / 15.8 GiB including vision + MTP. Anemll u48 is ~3.17 bpw / 9.74 GiB on the language weights. A lower KL at more bits is not a smaller-model win.
- **Context is 2,048-token raw windows**, not chat. The card does not use a system prompt or thinking template.
- **Code vs prose** are the only card categories. Do not average them into one headline without showing both.
- **Window placement** is our reading of "spread across the text". If Jun used the first 32,768 tokens only, the numbers will not reproduce his 0.0117 / 0.0600 even for oQe.
- **Full-vocab vs unspecified card KL.** If oMLX used a truncated softmax, our oQe-this-run column can differ from 0.0117 / 0.0600 even with the same text.
- **Scored length.** 16 × 2048 consumed tokens; 16 × 2047 next-token terms. Off-by-one relative to a 32,768-score reading.
- **Runtimes differ** (MLX GPU vs compiled ANE). The BF16 teacher is PyTorch. Residual engine error is in both candidate columns against the same teacher.
- **Tokenizer** is the BF16 checkpoint's. oQe ships a copy; ids must match.
- **Reference cache dtype** defaults to float16 log-softmax (~16 GiB at this vocab). Use `--ref-dtype f32` if disk allows.

## Background only: Forge Tier A KL-512 (different methodology)

**Not a comparison column. Do not put these next to 0.0117 / 0.0600.**

Forge's own quality gate (`scripts/m6_kl512_eval.py`, `scripts/qwen38_kl.py`) is a different evaluation:

- 64 BF16-generated chats (thinking on), longest sequence 857 tokens
- Teacher top-512 tokens plus one tail bucket
- Shared 45-token system prompt excluded on the "dev" number unless marked full
- `u48` / data600 published in `docs/QUANTIZATION.md`: **dev 0.0730 / holdout 0.0779**, 9.74 GiB

That gate answers "how close is this export to BF16 on our chat trace?" It does not use the oMLX Code/Novel windows, 2,048-token context, or (as published) full-vocabulary KL. Mixing 0.0730 with 0.0117 would be a methodology error.

This harness reuses the Forge **runtime** (Core AI T=8 teacher forcing) only to obtain ANE logits on the **card** windows. It does not load the 64-chat trace and does not report the 0.0730 / 0.0779 gate as a result.
