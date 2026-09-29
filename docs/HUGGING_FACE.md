# Core AI release bundle on Hugging Face

The M6 release uses **Core AI inference**. Upload the Core AI chunk/head packages, matching config/tokenizer assets and prepared embedding table. **Core ML model packages are not required for this release.** The repository retains Core ML conversion and experiment code for reproduction and learning.

Stage and validate a bundle locally, upload it yourself, then test a download from the exact uploaded revision. These commands do not create a Hub repository, upload files, or change repository visibility. Keep release preparation private until publication is explicitly approved.

Run commands from the ANEMLL Forge checkout. Choose the bundle path and HF model repository ID, and replace both revision placeholders with full commit hashes. The upstream checkpoint revision and uploaded bundle revision identify different repositories.

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
  export/                         # optional quantized conversion weights
    ...complete quantized export
  README.md                       # model card and reproduction provenance
  ...applicable licenses/notices
```

Use config, tokenizer and embeddings from the **exact pinned checkpoint used to produce the runtime assets**. Stage the original `config.json`, including its `text_config`. The current checker expects 64 layers, hidden size 5120 and an explicit vocabulary size. `embed_tokens_fp16.npy` must be the matching embedding matrix, C-contiguous float16 with shape `(vocab_size, 5120)`. Renaming unrelated embeddings or substituting a newer tokenizer is not compatible.

The `model/` directory accepts only the supported tokenizer/config files and the embedding array; do not copy original weight shards there. Optional files include generation/special-token configuration, chat templates, vocabulary and merges. Put licenses and the model card at the bundle root. Copy actual file contents, not cache symlinks.

Preserve Core AI's `manifest.json`, its `ctxs`, and every referenced chunk/head asset with the same relative names. Include source `.aimodel` packages so the target OS can compile them; a precompiled `.aimodelc` alone may be incompatible with another OS/toolchain. Include any explicitly referenced compiled packages too. Existing sibling `.aimodelc` packages are inventoried because the runtime can prefer them. Added runtime files absent from the inventory are rejected. The helper requires T=8 and ordered chunk ranges covering all 64 layers. Do not rename files without updating their runtime manifest.

The optional `export/` is for rebuilding/conversion, not required to run a prepared bundle. Include only reviewed export assets. The manifest generator inventories everything under runtime and export directories; it is not a private-data filter. Exclude calibration sessions, logs, unrelated checkpoints and temporary experiments.

## 2. Create and check the release inventory

```sh
python forge.py release-manifest \
  --bundle /path/to/bundle \
  --model-id UPSTREAM_OWNER/EXACT_CHECKPOINT \
  --model-revision FULL_UPSTREAM_COMMIT_HASH

python forge.py quick-test \
  --bundle /path/to/bundle --runtime coreai --check-only
```

`release-manifest` writes `release.json` only after layout validation. It records upstream identity, runtime contexts, component paths, file sizes and SHA-256 hashes. It records your supplied upstream revision; it does not prove the files came from that checkpoint. Regenerate it after any asset changes. Root-level model-card/license files are outside this asset inventory and need their own review.

`--check-only` checks selected-component hashes, manifests and the embedding header without loading the model. It still reads all selected assets for hashing, so large bundles take time. Passing it proves consistency, not correct generation or ANE placement.

You then create/select the HF **model** repository and upload the prepared bundle yourself, preserving this layout. Record the resulting full commit hash. Resolve exact upstream identity, redistribution terms, code license, and supported public toolchain in [RELEASE.md](RELEASE.md) before publication; this helper does not clear those gates.

## 3. Download the uploaded revision

```sh
python -m pip install huggingface_hub

python forge.py download \
  --repo YOUR_HF_ACCOUNT/YOUR_MODEL_REPO \
  --revision FULL_BUNDLE_COMMIT_HASH \
  --runtime coreai \
  --output /path/to/downloaded-bundle
```

Authenticate with your own HF credentials when downloading a private or gated repository. The repository ID is always a parameter; no destination is assumed. The helper resolves a branch/tag to a commit before downloading, selects `model/` plus `coreai/`, and verifies their inventory. Core AI is the download and quick-test default. A full commit hash makes subsequent downloads reproducible. Use a new output directory for a different release.

Add `--include-export` only when conversion weights are needed and the release includes them. The download step checks export hashes too when requested. For revision pinning, filtered downloads and local-directory behavior, see the [official Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download).

## 4. Run a short macOS smoke test

First repeat the portable integrity check on the downloaded copy:

```sh
python forge.py quick-test \
  --bundle /path/to/downloaded-bundle --runtime coreai --check-only
```

For actual inference, activate the existing compatible research environment described in [ENVIRONMENT.md](ENVIRONMENT.md), then build the bridge with the matching Xcode/SDK for Core AI:

```sh
bash coreai/swift_bridge/build.sh

python forge.py quick-test \
  --bundle /path/to/downloaded-bundle --runtime coreai \
  --prompt "The capital of France is" --tokens 16 \
  --report /path/to/smoke-report.json
```

The smallest advertised context is selected by default; `--ctx` accepts a context listed for Core AI. Keep `--report` outside the bundle directory. The smoke path runs short greedy generation without a drafter. It checks finite vocabulary-shaped logits and visible generated text, and reports load/generation time. It does not assert an expected answer, benchmark performance, certify placement or establish long-context quality.

Actual inference requires macOS and compatible model hardware/toolchain. It loads the full target even for 16 tokens, and first-load compilation can take minutes. Run it when adequate memory and the ANE are available. Current Python imports still include PyTorch, safetensors, coremltools and research helpers; tokenizers is also required. Core AI additionally needs the native Swift bridge and compatible system framework. The current coremltools dependency does not require distributing Core ML model packages. Installing `huggingface_hub` only enables download, and a clean public dependency recipe remains a release gate.
