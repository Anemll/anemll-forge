# M5 Pro with 24 GB Unified Memory: opt-in direct-attention profile

This research workaround is restricted to **Apple M5 Pro with exactly 24 GB
Unified Memory**. It is not a default for the M5 family and must not activate
on M6, M5/M5 Max, or an M5 Pro with another memory capacity.

Unmarked upstream bundles retain their normal context ladder, cache selection
and runtime behavior. Prefer those on other hardware. No automatic fallback,
hardware-based context selection, requantization or compiler-mode change is
introduced by this profile.

## Gates and scope

- The extension and validation CLIs require explicit `--m5pro-24gb` opt-in.
- The gate checks macOS, the exact chip brand and `hw.memsize == 24 * 1024**3`.
  It checks physical Unified Memory, not available/free RAM.
- Missing detection, unknown profiles and hardware mismatches fail closed.
  The opt-in flag does not bypass these checks.
- Derived manifests carry `hardware_profile: m5pro-24gb-direct-fp16-v1`
  (a structured object, including chip and memory requirements).
- The launcher, both runtime backends and package probe reject incompatible
  marked builds before allocating a model. Context selection preserves the
  marker and validates a marked source.
- This profile supports FP16 KV cache, decode through 31K (31744 tokens), and
  batched prefill only through 24K. The experimental 31K Pi configuration is
  hardware-checked before profile writes. Standard 32K/64K Pi settings are
  unchanged.

Simply copying the build to a 32 GB M6 does not make it eligible. Removing
its hardware marker is not a supported migration procedure. The generic
context-selection tool remains available for explicitly selected, unmarked
upstream graphs; it does not enable this workaround.

## Reproducible experiment

Measured configuration: M5 Pro, 24 GB Unified Memory, macOS 27.0 (26A428),
Xcode 27.0, Core AI authoring SDK `coreai-core==1.0.0b2`, Python 3.11.
These measurements do not establish compatibility with other OS/compiler
versions or other chips.

The source was the legacy FP16-cache paired bundle at model revision
`2a2108f491f0e455506d6ec79a82eb55476cdbef`. The updated selectable-cache
reference was pinned to `cd7dfc605ccad091b961f7788939c30d01c3793e`.
Keep target, output head, tokenizer/embeddings and DFlash2 drafter paired;
do not apply these tools to selectable/V8-cache graphs.

Observed on this configuration:

| Formulation | Observation |
| --- | --- |
| Published blocked 24K/32K graphs | ANE compiler failure |
| Updated upstream 32K V8 graph | Topological-sort failure / GPU fallback |
| Direct FP16 32K graph pair | Compilation failure / GPU fallback |
| Direct FP16 24K and 31K | Successful ANE placement in the tested builds |

The working/failing shapes bracket a compiler boundary for this formulation.
They do **not** prove an internal ANE dimension limit or explain Apple's
topological-sort error. Do not describe the workaround as a compiler fix.

## Prepare a validated 8K/16K source

Choose fresh output directories outside the source build. Model assets and
generated reports should live outside the source checkout. The ignored
`.forge-models/` directory is available for private local experiments only;
do not commit or attach its contents.

Example paths below are placeholders; substitute your own locations.
Use an authoring environment with the matching Core AI SDK:

```sh
export FORGE_BUNDLE="$HOME/Models/anemll-forge-qwen3.8-27B"
export SOURCE_COREAI="$HOME/Models/qwen38-fp16-8k16k"
export DERIVED_COREAI="$HOME/Models/qwen38-m5pro-24gb-direct"
export AUTH_PY=".venv-authoring/bin/python"

"$AUTH_PY" coreai/select_context.py \
  --source "$FORGE_BUNDLE/coreai" --output "$SOURCE_COREAI" \
  --ctx 8192,16384
```

Probe every retained target package, head and paired drafter in isolated
processes, then run `coreai/inspect_coreai_cache.py --strict`.
A successful load, `error: null`, or an ANE compilation message alone does
not establish placement. Always require a successful strict cache audit.
Until using a fail-closed probe version, do not rely on probe exit status
alone to detect GPU fallback.

## Experimental 24K direct attention instead of the published blocked graph

The extension clones the existing direct-attention 16K graphs and changes
context shapes plus two slice-boundary constants. It does not change the
weights. The intended attention operation is unchanged, but equivalence to
the published blocked formulation needs broader numerical/quality checks.

For 24K only, use `--ctx 24576`. For the lower-memory 31K ladder:

```sh
"$AUTH_PY" coreai/extend_direct_24k.py \
  --source "$SOURCE_COREAI" --output "$DERIVED_COREAI" \
  --ctx 24576,31744 --prefill-ctx 24576 --m5pro-24gb
```

Original and transformed graph round trips are checked exactly. Resource
verification first compares the raw bytecode section. If the SDK reorders
buffers, it compares all named resources' complete encodings, including
alignment, against a separately loaded source asset. Missing or changed
resources fail verification. The manifest is published only after every
chunk passes; incomplete output directories must not be served.

## 31K memory investigation

The full 31K batched-prefill candidate compiled but caused severe paging
during a long-prompt test on this memory configuration. It is not the
recommended 24 GB profile.

The lean ladder retains decode at 8K/16K/24K/31K and batched prefill only at
8K/16K/24K. Prompt text above 24K uses eight-token verification blocks.
This reduces peak native scratch allocation but makes large-prompt
processing above 24K slower.

After isolated probes and strict cache auditing, validate the lean build:

```sh
.venv/bin/python coreai/validate_direct_24k.py \
  --bundle "$FORGE_BUNDLE" --build "$DERIVED_COREAI" \
  --ctx 31744 --m5pro-24gb
```

Synthetic smoke/parity results for the lean build:

- All 18 packages / 115 entry points fully ANE, zero GPU regions.
- Exact cached-prefix and position preservation during 24K-to-31K resize;
  continuation logits cosine 1.0 and identical top token.
- A 24870-token calibration prompt crossed all three growth boundaries
  and answered the synthetic capital-of-France question correctly.
- Total prompt plus short generation: 291.25 seconds. Resize costs:
  150 ms (8K-to-16K), 361 ms (16K-to-24K), 765 ms (24K-to-31K).
- System-wide wired memory at completion: 21.84 GiB. Memory remains tight.

These are bounded execution/parity checks, not a capability benchmark,
a full-window 31744-token test, or proof of general long-context quality.
Performance depends on prompt length, cache reuse and speculative acceptance.

## Run explicitly

```sh
.venv/bin/python forge.py serve --runtime coreai \
  --model "$FORGE_BUNDLE/model" --build "$DERIVED_COREAI" \
  --ctx 31744 --host 127.0.0.1 --port 8765 --kv-cache-dtype fp16 \
  --draft "$FORGE_BUNDLE/drafter/dflash2_lut4_gptq.aimodel" \
  --drafter "$FORGE_BUNDLE/drafter"
```

There is no new default server profile. Existing validated builds are not
overwritten. Use a separate Pi profile based on `examples/pi/`, then run:

```sh
.venv/bin/python scripts/qwen38_pi_config.py --ctx 31744 \
  --pi-dir "$PI_PROFILE" --build m5pro-24gb-direct
PI_CODING_AGENT_DIR="$PI_PROFILE" \
  pi --offline --provider ane-qwen38 --model qwen38-27b-ane --thinking off
```

Choose `PI_PROFILE` explicitly; do not overwrite another provider's profile.
The Pi helper backs up changes and rejects this context on other hardware.

## Sharing and contribution boundaries

Submit only reviewed source, portable documentation and tests. Do not submit
model bundles, cache contents, generated reports, logs, Pi profiles,
credentials, token/feature traces or private session notes. Token IDs and
features are not anonymized data. The original local notes are excluded
from source contributions.

Keep generic probe hardening separate from the hardware-specific research
changes. Discuss the experimental formulation with maintainers before
submitting it. Include synthetic reproduction commands and clearly state
the missing M6/other-memory and broad quality validation.
