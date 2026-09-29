# Private release preparation

Keep the GitHub repository private until the owner explicitly approves a visibility change. This source port does not publish a release or model assets.

## First port scope

Imported inference, quantization, conversion, reference and experiment source, Swift bridge source, and edited historical documentation. Added an explicit-path launcher and a current Core ML workflow. Defaults now use the documented tanh MLP SiLU fix and loopback serving. The calibration session importer requires an explicit glob; no session contents were imported. The bridge build script now propagates compiler failures and uses the selected Xcode. Removed a stale sklearn module stub from the Core AI builder.

The source repository is left unchanged. Its dirty/untracked working tree is recorded by source hashes in the provenance manifest. No binaries, models, arrays, raw logs, or source Git history were imported.

The [Hugging Face bundle workflow](HUGGING_FACE.md) now includes release inventory generation, revision-pinned downloads and an integrity/inference smoke-test command. Inference can use prepared config/tokenizer/embedding assets without original checkpoint shards. The tooling passed fixture tests; uploaded-artifact downloads and full-model hardware validation remain outstanding.

## Required before public release

- Confirm the canonical Qwen checkpoint name, upstream URL/revision, architecture, license and weight redistribution terms. `Qwen3.8-27B` is the research name supplied by the owner and used in the source; the implementation uses `qwen3_5`.
- Select a source-code license and audit provenance/attribution for code adapted from transformers, DFlash, other repositories and private SDK examples. No license is invented by this port.
- Check the upstream OptiQ sensitivity dataset's attribution and redistribution terms before public distribution; the newly imported sensitivity notes identify its source.
- Pin the public dependency sources, including any coremltools patches and Core AI SDK constraints; produce a clean install/build test.
- Recover and review the exact best mixed-bit plan, calibration recipe, export metadata and reference trace. Publish only data with suitable rights and explicit consent; replace private sessions with reproducible public or synthetic fixtures.
- Recover missing external benchmark helpers and validation harnesses, or replace them with standalone equivalents.
- Audit direct research entry points for machine paths, output overwrites, cache operations, SDK/private API use and unsupported modes. `forge.py` covers the documented first path; the entire notebook collection is not yet portable.
- Rebuild a single chunk, check ANE placement and finite/norm/L2 parity, then rebuild the full target. Validate long generation, context transitions, sampling and memory stability on the claimed hardware/OS.
- Confirm artifact formats and hashes, tokenizer/template behavior, upstream notices, test data provenance, resource requirements and documented failure messages.
- Convert historical claims into reproducible benchmark records with commands, environment, sample size and linked artifacts. Keep projections, CPU-only checks and hypotheses clearly labeled.

## Editorial policy

Keep both a concise current guide and the experiment history. Correct machine-specific commands and privacy-sensitive references; preserve failed approaches and superseded recommendations with status and dates. Cite evidence for architecture claims and describe compiler limitations in the versions where they were observed. Do not silently rewrite old results as newly measured or present a private artifact as publicly available.
