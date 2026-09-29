# Qwen attribution and model redistribution

The planned Hugging Face destination is **`anemll/anemll-forge-qwen3.8-27B`** under [ANEMLL](https://huggingface.co/anemll). This is ANEMLL's independent quantized, text-only Core AI conversion of **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**, developed by the Qwen Team. Original model copyright: **Alibaba Cloud**. ANEMLL provides the quantization, conversion, Apple Neural Engine runtime and experiment documentation; it does not claim authorship of the original model or official Qwen/Apple endorsement.

## Verified source and license

The local source checkpoint's Hugging Face download metadata records revision **`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`**. Ten local source files, including the license, config, tokenizer and chat template, match that pinned upstream revision's actual Git-blob/LFS hashes. All 18 weight-shard download etags match upstream SHA256 metadata; the large shard bytes were not rehashed in this review. Export/embedding lineage must still be verified when staging the final bundle. [QWEN_SOURCE.json](../release/huggingface/QWEN_SOURCE.json) records this evidence and its limits.

The pinned [upstream LICENSE](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE) is **Apache License 2.0** and includes `Copyright 2026 Alibaba Cloud`. The copy in [release/huggingface/LICENSE](../release/huggingface/LICENSE) preserves its exact bytes. No standalone upstream `NOTICE` was found in the reviewed revision's repository listing. Our [NOTICE](../release/huggingface/NOTICE) is an added ANEMLL attribution record, not a substitute for an upstream notice.

## Distribution requirements

Apply [Apache 2.0 §4](https://www.apache.org/licenses/LICENSE-2.0#redistribution) to the Qwen-derived artifacts:

- Give recipients the full upstream license. A Hub metadata tag or external license link alone is not the included license copy.
- Mark modified files prominently as changed. For our conversion, identify quantized weights, low-rank corrections where present, the float16 embedding extraction and Core AI packaging. Use notices in changed text files and supported model/package metadata. The release inventory's per-file `modification_notice` and [MODIFICATIONS.md](../release/huggingface/MODIFICATIONS.md) document binary changes without corrupting model files; do not assume a general README sentence replaces every file-level notice requirement.
- Retain applicable copyright, patent, trademark and attribution notices in distributed source. Do not replace Alibaba Cloud's copyright with ANEMLL's.
- Preserve relevant upstream `NOTICE` attribution if one is included in the actual source distribution. Recheck if the source revision or included dependencies change.

[Apache 2.0 §6](https://www.apache.org/licenses/LICENSE-2.0#trademarks) provides no general trademark license. Use “Qwen” to identify the upstream model and describe this as an independent ANEMLL conversion. The name `anemll-forge-qwen3.8-27B` identifies the derivative without presenting it as an official Qwen release.

The upstream [citation section](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md#citation) invites academic citation. We link it in the model card as attribution; it is not an additional mandatory Apache license condition.

## Concrete release files

Copy [release/huggingface/](../release/huggingface/) into the bundle root before generating its inventory. It contains the draft model card, exact Qwen license, NOTICE, modifications record and pinned-source evidence. The download helper includes these documents with every runtime download and verifies their hashes.

The model card uses `base_model: Qwen/Qwen3.8-27B`, `base_model_relation: quantized` and `license: apache-2.0`. It identifies the Core AI/Swift runtime and text-only scope. The upstream vision/video features, native context claims, Transformers loading examples and upstream benchmark scores are not claims about this conversion.

This model-artifact license does not select a license for independently authored ANEMLL source code or clear other bundled components. Before publication, finish the source-code attribution/license review, SDK/toolchain redistribution checks, calibration/data provenance review, actual modified-file notices and artifact lineage/validation recorded in [RELEASE.md](RELEASE.md). No public upload or visibility change is performed by this preparation.

## Separate DFlash2 provenance

The release also derives a drafter from [ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/tree/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b), pinned at `4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b`. The source checkpoint SHA-256 was checked against that upstream revision during private preparation. Its Apache 2.0 LICENSE and upstream NOTICE must be preserved separately under `drafter/`, alongside `DFLASH2_SOURCE.json`. Extracted selector codebooks remain derived source material; they need the same provenance and notices. This verification does not reconstruct the lineage of the compiled Core AI body or settle licensing of independently adapted source code.

The historical [DFlash2 notebook](../DFLASH2_ANE_PLAN.md) also identifies reference code `dflash-07ebd93`. Preserve and audit that separate code attribution before publication. Do not infer that the Qwen target's license establishes rights for every drafter or code component. See [the release pairing guide](SPECULATIVE_DECODING.md).
