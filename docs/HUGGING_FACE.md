# Core AI release bundle on Hugging Face

The intended fast M6 release uses **Core AI inference with the matching DFlash2 speculative drafter**. Upload target chunk/head packages, matching config/tokenizer/embedding assets, and the drafter package, metadata, configuration, compact selector codebooks and provenance/license files. **Core ML model packages are not required for this release.** The repository retains Core ML conversion and experiment code for reproduction and learning.

HF destination: **`anemll/anemll-forge-qwen3.8-27B`** under [ANEMLL](https://huggingface.co/anemll). The model card describes a research project for the M6 Apple Neural Engine. Quality evaluation is currently KL; measured V8 prefill/decode results and their limits are preserved in the [KV-cache quantization research trace](research/KV_CACHE_QUANTIZATION_2026-10-02.md). The download helper defaults to this repository.

For inference, start with the download and smoke-test steps below or the [README quickstart](../README.md). The staging sections are for maintainers preparing a new bundle; test a download from the exact uploaded revision.

Run commands from the ANEMLL Forge checkout. Choose the bundle path and replace the uploaded-bundle revision placeholder with its full commit hash. The pinned upstream checkpoint revision and uploaded bundle revision identify different repositories. Read [Qwen attribution and redistribution requirements](ATTRIBUTION.md).

## 1. Stage the assets

Create a self-contained directory outside the source checkout:

```text
bundle/
  model/
    config.json
    tokenizer.json
    tokenizer_config.json
    embed_tokens_fp16.npy
    ...other tokenizer/config files, if needed
  coreai/
    manifest.json
    ...all referenced chunk and head packages
  drafter/
    dflash2_lut4_gptq.aimodel/
    dflash2_lut4_gptq.json
    config.json
    selector.safetensors
    LICENSE
    NOTICE
    DFLASH2_SOURCE.json
  export/                         # optional quantized conversion weights
    ...complete quantized export
  README.md                       # model card
  LICENSE                         # exact upstream Qwen license
  NOTICE                          # ANEMLL-added Qwen attribution
  MODIFICATIONS.md                # prominent conversion/change notices
  QWEN_SOURCE.json                 # pinned upstream source evidence
  config.json                     # exact model/config.json copy for Hub discovery
```

Use config, tokenizer and embeddings from the **exact pinned checkpoint used to produce the runtime assets**. Stage the original `config.json`, including its `text_config`. The current checker expects 64 layers, hidden size 5120 and an explicit vocabulary size. `embed_tokens_fp16.npy` must be the matching embedding matrix, C-contiguous float16 with shape `(vocab_size, 5120)`. Renaming unrelated embeddings or substituting a newer tokenizer is not compatible.

The `model/` directory accepts only the supported tokenizer/config files and the embedding array; do not copy original weight shards there. Optional files include generation/special-token configuration, chat templates, vocabulary and merges. Copy the five prepared files from `release/huggingface/` into the bundle root; they are required, hashed and downloaded with the model. Copy actual file contents, not cache symlinks.

Preserve Core AI's `manifest.json`, its `ctxs`, and every referenced chunk/head asset with the same relative names. Include source `.aimodel` packages so the target OS can compile them; a precompiled `.aimodelc` alone may be incompatible with another OS/toolchain. Include any explicitly referenced compiled packages too. Existing sibling `.aimodelc` packages are inventoried because the runtime can prefer them. Added runtime files absent from the inventory are rejected. The helper requires T=8 and ordered chunk ranges covering all 64 layers. Do not rename files without updating their runtime manifest.

The published bundle uses these packages since revision `1192a9c83d1b3e7ad76602ed6ec6d05e6852cad2` (4 October 2026). A target converted with the current defaults is V8-only (`kv_cache.format: v8`, single-format entries) and uses the faster graph recorded in each chunk's `numerics` (`GDN_FAST`, `ATT_BLOCK`, `ATT_BLOCK_PREFILL`). Its packages are about 0.3 MB per chunk larger than the release graph and contain the same weights, but every chunk package and the manifest change, so an update replaces all 16 chunk packages; the head, model assets and drafter can stay. Downloaders compile the new packages once on first start (about 23 minutes on M6). See [M6 compute acceleration](research/M6_COMPUTE_ACCELERATION_2026-10-03.md).

The previous selectable KV-cache update (revision `cd7dfc605ccad091b961f7788939c30d01c3793e`) replaced the 16 target chunks and manifest, declares both FP16/V8 layouts with V8 as default, and retains the existing head, model assets and tested drafter. Preserve the complete `entries` and `entries_by_kv` maps for both formats; selected-function audits choose a format without deleting the other aliases. Its context ladder is 8K/16K/32K/48K/64K. Update the Forge runtime together with these packages and regenerate `release.json`; older FP16-only revisions remain supported. The manifest must not carry a maintainer's private absolute export path. See [V8 model compatibility and conversion](KV_CACHE_V8.md).

The required `drafter/` contains the matching Core AI DFlash2 artifact and sidecar, checkpoint config and compact predecessor/successor codebooks. Preserve the separate upstream LICENSE/NOTICE and pinned source evidence. See [the exact compatibility contract](SPECULATIVE_DECODING.md); neither original BF16 drafter weights nor a Core ML drafter package is required for serving.

The optional `export/` is for rebuilding/conversion, not required to run a prepared bundle. Include only reviewed export assets. The manifest generator inventories everything under runtime and export directories; it is not a private-data filter. Exclude calibration sessions, logs, unrelated checkpoints and temporary experiments.

## 2. Create and check the release inventory

```sh
python forge.py release-manifest \
  --bundle /path/to/bundle \
  --model-id Qwen/Qwen3.8-27B \
  --model-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

python forge.py quick-test \
  --bundle /path/to/bundle --runtime coreai --check-only
```

`release-manifest` writes `release.json` only after layout validation. It records upstream identity, runtime contexts, component paths, file sizes and SHA-256 hashes, including the required root model-card/license/notice/source records. It also creates an exact root `config.json` copy and records its size/hash in optional `hub_config` metadata. An existing root config must already match; mismatches and symlinks are rejected. This optional field keeps older releases and inference clients compatible. Derived binary entries carry a `modification_notice` referencing `MODIFICATIONS.md`. Authors must also mark actually changed text files and supported model-package metadata appropriately. The supplied upstream revision does not itself prove artifact lineage; regenerate the inventory after any asset changes.

`--check-only` checks selected-component hashes, manifests and the embedding header without loading the model. It still reads all selected assets for hashing, so large bundles take time. Passing it proves consistency, not correct generation or ANE placement.

You then create/select the HF **model** repository and upload the prepared bundle yourself, preserving this layout. Record the resulting full commit hash. Resolve exact upstream identity, redistribution terms, code license, and supported public toolchain in [RELEASE.md](RELEASE.md) before publication; this helper does not clear those gates.

## 3. Download the uploaded revision

Browse the [complete model files](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B/tree/main), [Core AI target](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B/tree/main/coreai), [DFlash2 drafter](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B/tree/main/drafter), and [tokenizer/config/embeddings](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B/tree/main/model). Download the matching pair with the helper rather than selecting target files alone.

```sh
python -m pip install huggingface_hub

python forge.py download \
  --repo anemll/anemll-forge-qwen3.8-27B \
  --revision FULL_BUNDLE_COMMIT_HASH \
  --runtime coreai \
  --output /path/to/downloaded-bundle
```

If repository access requires authentication, use `hf auth login` with your own account. `--repo` defaults to `anemll/anemll-forge-qwen3.8-27B` and can be overridden. The helper resolves a branch/tag to a commit before downloading, selects `model/`, `coreai/`, `drafter/` and the required release documents, and verifies their inventory. Core AI is the download and quick-test default. A full commit hash makes subsequent downloads reproducible. Use a new output directory for a different release.

Add `--include-export` only when conversion weights are needed and the release includes them. The download step checks export hashes too when requested. For revision pinning, filtered downloads and local-directory behavior, see the [official Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download).

## Hub download counts

Hugging Face counts server-side requests to selected query files rather than adding every chunk download. The published card uses YAML front matter with `library_name: anemll-forge`, `pipeline_tag: text-generation`, Apache-2.0 licensing and upstream attribution. A separate YAML file is unnecessary. The root `config.json` supplies the documented default query file; nested `model/config.json` and `drafter/config.json` do not replace this root discovery file. The current helper downloads and verifies the root copy through the normal pinned snapshot when `hub_config` is declared. It does not force cache bypasses or make counting-only requests. Earlier releases without the field remain supported.

The reported count is a query-file request metric, including GET/HEAD requests, rather than a count of completed weight transfers or unique users. Direct chunk-only transfers can be absent from this metric. Setting a custom library name does not register a custom counting rule; an official Forge integration could use `release.json` as its single query file through Hugging Face's library-registration process. Follow the [download-count rules](https://huggingface.co/docs/hub/models-download-stats) and [library-integration guide](https://huggingface.co/docs/hub/models-adding-libraries). Check the live counter after normal downloads; these metadata changes do not establish a historical backfill.

## 4. Run a short macOS smoke test

First repeat the portable integrity check on the downloaded copy:

```sh
python forge.py quick-test \
  --bundle /path/to/downloaded-bundle --runtime coreai --check-only
```

For actual inference, activate the inference environment from the [README setup](../README.md#1-set-up-the-inference-environment) and [ENVIRONMENT.md](ENVIRONMENT.md), then build the bridge with the matching Xcode/SDK for Core AI:

```sh
bash coreai/swift_bridge/build.sh

python forge.py quick-test \
  --bundle /path/to/downloaded-bundle --runtime coreai \
  --prompt "The capital of France is" --tokens 16 \
  --report /path/to/smoke-report.json
```

The smallest advertised context is selected by default; `--ctx` accepts a context listed for Core AI. Keep `--report` outside the bundle directory. The default smoke path runs actual greedy speculative generation with the matching drafter, target verification and acceptance. It checks finite vocabulary-shaped logits and visible generated text and records runtime details. A `--plain` smoke is an explicitly labeled target-only diagnostic; it does not validate the release pair. It does not assert an expected answer, benchmark performance, certify placement or establish long-context quality.

Actual inference requires macOS and compatible model hardware/toolchain. It loads the full target and drafter even for 16 tokens, and first-load compilation can take minutes. Run it when adequate memory and the ANE are available. Current Python imports still include PyTorch, safetensors, coremltools and research helpers; tokenizers is also required. Core AI additionally needs the native Swift bridge and compatible system framework. The current coremltools dependency does not require distributing Core ML model packages. Installing `huggingface_hub` only enables download. The starting inference pins are in [requirements-inference.txt](../requirements-inference.txt); full clean-environment validation remains outstanding.

## 5. Serve the complete release

```sh
python forge.py serve --runtime coreai \
  --model /path/to/downloaded-bundle/model \
  --build /path/to/downloaded-bundle/coreai --ctx 16384
```

The launcher resolves the matching sibling `drafter/` by default; missing assets are an error. Explicit `--draft` and `--drafter` paths support other layouts. `download --plain`, `quick-test --plain` and `serve --plain` deliberately select the target-only diagnostic path; do not use their results as validation or timing of the complete release. Regenerate `release.json` after adding or replacing drafter assets, then test the exact uploaded revision. The earlier target-only upload is not the final speculative bundle.

For the fixed-shape entry ladder, KV-buffer resizing, cache retention and the separate drafter ring, see [how ANE context expansion works](SPECULATIVE_DECODING.md#how-ane-context-expansion-is-implemented).
