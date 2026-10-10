# Converting this export to Core AI packages

These steps rebuild the Core AI target and its DFlash2 drafter published in [anemll/anemll-forge-qwen3.8-27B](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B) (16 chunk packages, the output head and a manifest) from this repository, with [ANEMLL Forge](https://github.com/Anemll/anemll-forge). They were run on an M6 with macOS 27. About 45 minutes of export and, overlapped with it, the first ANE compile. Release 0.2's export needs the ANEMLL Forge source from release 0.2 on (its converter packages the 64-entry vector lookup tables of the MLP and applies the token-mixer rotations recorded in the export).

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
hf download anemll/anemll-quantized-qwen3.8-27b-for-CoreAI --local-dir "$Q" --exclude "export/mix25in_mixr_lr64mix/*"   # about 17 GB
export E=release_vq3pA_mixh_s600_k1_mat DIGEST=weights_digest_0.2.json   # release 0.2, the current packages
# export E=mix25in_mixr_lr64mix DIGEST=weights_digest.json              # the first release (drop the --exclude above)
```

Check the weights the converter will read against the published digests (a few minutes):

```sh
MODEL="$Q/model" EXPORT_DIR="$Q/export/$E" \
  .venv-convert/bin/python scripts/qwen38_weights_digest.py --check "$Q/$DIGEST"
```

## 3. Export

The published packages: 8-bit attention with M6 and M5 function sets in the same packages, transposed key cache, V8 cache, five context entries:

```sh
export OUT="$HOME/coreai-builds"                  # writes $OUT/${E}_kvv8
MODEL="$Q/model" EXPORT_DIR="$Q/export/$E" OUT="$OUT" \
SILU=tanh MLP_SILU=tanh GDN_SQ=16 GDN_SV=64 MLP_DS=1 QCONV_INT8=0 \
ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_INT8MM=s8,s8b,sm8,pvf8 ATT_INT8MM_M5=s8,s8b KV_KEYS_T=1 \
  .venv-convert/bin/python coreai/qwen38_coreai_build.py all --kv-cache-dtype v8 \
  --ctx 8192,16384,32768,49152,65536 --pctx 8192,16384,32768,49152,65536
```

To compile for this Mac's ANE while the export runs, start this first, in the **inference** environment (the compile cache is per Python):

```sh
python forge.py compile --follow --build "$OUT/${E}_kvv8" &
```

**Build without FP8 (M5 and M6).** A package without FP8 runs on both chips. On M5-family Macs it compiles directly, with no first-start derivation, no second copy of the chunks and no `coreai-core` at run time; on M6 it gives up the FP8 speedup but is the base for combinations not yet validated with the FP8 forms (kv8, contexts above 64K). Use `ATT_INT8MM=s8,s8b` (INT8 scores, FP16 softmax and PV) and leave out `ATT_INT8MM_M5`, with its own `OUT`:

```sh
MODEL="$Q/model" EXPORT_DIR="$Q/export/$E" OUT="$HOME/coreai-builds-nofp8" \
SILU=tanh MLP_SILU=tanh GDN_SQ=16 GDN_SV=64 MLP_DS=1 QCONV_INT8=0 \
ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_INT8MM=s8,s8b KV_KEYS_T=1 \
  .venv-convert/bin/python coreai/qwen38_coreai_build.py all --kv-cache-dtype v8 \
  --ctx 8192,16384,32768,49152,65536 --pctx 8192,16384,32768,49152,65536
```

Measured on an M5 Max against the previous V8 packages: prefill +1.2% at 8K to +7.4% at 64K, decode +3.7 to +8.2%.

**More context on Macs with more memory.** `--ctx` / `--pctx` take any list of entries; the runtime grows through them as a conversation does. Entries above 64K are research builds so far (an 80K-only package used 25.7 GB of wired memory on a 32 GB M6), and every entry adds compile time and wired memory, so check memory on your Mac first.

**INT8 keys and values.** `--kv-cache-dtype kv8` stores keys as INT8 too (about half the cache of FP16, against a quarter saved by V8). It has been measured as a research build; the 8-bit attention forms and the transposed key cache were validated with V8, so check quality (the long-context evals) before relying on that combination.

Other variants: drop `ATT_INT8MM_M5` for an M6-only build; drop the `ATT_*` and `KV_KEYS_T` settings for the plain V8 graph; `--ctx` / `--pctx` choose the context entries. The settings are recorded in each chunk's `numerics` in `manifest.json`, and the server prints them at startup.

## 4. Drafter (optional)

The published drafter (`dflash2_lut4_gptq.aimodel`) from this repository's `drafter/` and the target's LM head; check its inputs first (165 arrays):

```sh
DRAFTER="$Q/drafter" DRAFT_EXPORT="$Q/drafter" HEAD_EXPORT="$Q/export/$E/lm_head.safetensors" MODEL="$Q/model" \
  .venv-convert/bin/python scripts/qwen38_weights_digest.py --drafter --check "$Q/drafter/$DIGEST"
DRAFTER="$Q/drafter" DRAFT_EXPORT="$Q/drafter" HEAD_EXPORT="$Q/export/$E/lm_head.safetensors" MODEL="$Q/model" \
  OUT="$OUT/drafter" .venv-convert/bin/python coreai/dflash2_coreai_build.py     # -> $OUT/drafter/dflash2_lut4_gptq.aimodel
```

The build reads only the drafter's small tensors and config (`drafter/small.safetensors`) and the mask token's embedding row (`model/`), not the full checkpoints. Put the new package and the inference bundle's `drafter/config.json` and `selector.safetensors` side by side to use it (step 5, `DRAFT` / `DRAFTER`).

## 5. Run it

The runtime also needs the tokenizer and FP16 embedding table (`model/`) of the inference bundle, and the paired drafter (the bundle's `drafter/`, or yours from step 4). Download that bundle and point `BUILD` at the new packages:

```sh
export FORGE_BUNDLE="$HOME/Models/anemll-forge-qwen3.8-27B"
python forge.py download --repo anemll/anemll-forge-qwen3.8-27B --revision main --runtime coreai --output "$FORGE_BUNDLE"
BUILD="$OUT/${E}_kvv8" CTX=64K scripts/qwen38_server.sh start
```

On an M5 the runtime derives its M5 function set once on first start (it needs `coreai-core`, included in `requirements-inference.txt`).

## What to expect

- The weights inside your packages equal the published ones (step 2's check). The package bytes differ slightly from the published files and from build to build: the Core AI converter's serialization is not byte-deterministic.
- Disk: about 17 GB per export in this repository, about 11 GB per full build, and 11 to 14 GB of compile cache per build in `~/Library/Caches/coreai-cache`.
- Memory: the export runs one chunk at a time (several GB); keep other large jobs off the machine while it runs.
