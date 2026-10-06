# Converting this export to Core AI packages

These steps rebuild the Core AI target published in [anemll/anemll-forge-qwen3.8-27B](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B) (16 chunk packages, the output head and a manifest) from this repository, with [ANEMLL Forge](https://github.com/Anemll/anemll-forge). They were run on an M6 with macOS 27. About 25 minutes of export and, overlapped with it, the first ANE compile.

## 1. Environment

Clone ANEMLL Forge and create a **conversion** environment, separate from the inference one (the builder patches `coreai-opt` internals, so keep the pinned versions):

```sh
git clone https://github.com/Anemll/anemll-forge.git && cd anemll-forge
uv venv --python 3.13 --seed .venv-convert     # or: python3.13 -m venv .venv-convert
.venv-convert/bin/python -m pip install -r requirements-conversion.txt
```

## 2. Download this repository

```sh
export Q="$HOME/Models/anemll-quantized-qwen3.8-27b-for-CoreAI"
hf download anemll/anemll-quantized-qwen3.8-27b-for-CoreAI --local-dir "$Q"       # about 18 GB
```

Check the weights the converter will read against the published digests (2,097 arrays; a few minutes):

```sh
MODEL="$Q/model" EXPORT_DIR="$Q/export/mix25in_mixr_lr64mix" \
  .venv-convert/bin/python scripts/qwen38_weights_digest.py --check "$Q/weights_digest.json"
```

## 3. Export

The published packages: 8-bit attention with M6 and M5 function sets in the same packages, transposed key cache, V8 cache, five context entries:

```sh
export OUT="$HOME/coreai-builds"                  # writes $OUT/mix25in_mixr_lr64mix_kvv8
MODEL="$Q/model" EXPORT_DIR="$Q/export/mix25in_mixr_lr64mix" OUT="$OUT" \
SILU=tanh MLP_SILU=tanh GDN_SQ=16 GDN_SV=64 MLP_DS=1 QCONV_INT8=0 \
ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_INT8MM=s8,s8b,sm8,pvf8 ATT_INT8MM_M5=s8,s8b KV_KEYS_T=1 \
  .venv-convert/bin/python coreai/qwen38_coreai_build.py all --kv-cache-dtype v8 \
  --ctx 8192,16384,32768,49152,65536 --pctx 8192,16384,32768,49152,65536
```

To compile for this Mac's ANE while the export runs, start this first, in the **inference** environment (the compile cache is per Python):

```sh
python forge.py compile --follow --build "$OUT/mix25in_mixr_lr64mix_kvv8" &
```

Variants: drop `ATT_INT8MM_M5` for an M6-only build; drop the `ATT_*` and `KV_KEYS_T` settings for the plain V8 graph; `--ctx` / `--pctx` choose the context entries. The settings are recorded in each chunk's `numerics` in `manifest.json`, and the server prints them at startup.

## 4. Run it

The runtime also needs the tokenizer and FP16 embedding table (`model/`) and the paired DFlash2 drafter (`drafter/`) of the inference bundle. Download that bundle and point `BUILD` at the new packages:

```sh
export FORGE_BUNDLE="$HOME/Models/anemll-forge-qwen3.8-27B"
python forge.py download --repo anemll/anemll-forge-qwen3.8-27B --revision main --runtime coreai --output "$FORGE_BUNDLE"
BUILD="$OUT/mix25in_mixr_lr64mix_kvv8" CTX=64K scripts/qwen38_server.sh start
```

On an M5 the runtime derives its M5 function set once on first start (it needs `coreai-core`, included in `requirements-inference.txt`).

## What to expect

- The weights inside your packages equal the published ones (step 2's check). The package bytes differ slightly from the published files and from build to build: the Core AI converter's serialization is not byte-deterministic.
- Disk: 18 GB for this repository, about 10 GB per full build, and 11 to 14 GB of compile cache per build in `~/Library/Caches/coreai-cache`.
- Memory: the export runs one chunk at a time (several GB); keep other large jobs off the machine while it runs.
