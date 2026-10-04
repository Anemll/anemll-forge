# DFlash2: the fast Core AI release path

The intended M6 release pairs the **Qwen3.8-27B Core AI target, Swift bridge and matching Core AI DFlash2 drafter**. Release downloads, serving and performance benchmarks use this pair. Plain target-only inference is available with `--plain` for numerical debugging and matched comparisons; it is not the release speed path. Historical measurements remain observations from their recorded setup, not new measurements of the assembled bundle.

## What the verifier does

After prompt prefill, the target selects an anchor token. DFlash2 proposes seven following tokens. One target call evaluates `[anchor, draft1, ..., draft7]` with `T=8`. The server checks proposals in order, commits the anchor and accepted prefix, and chooses a target token at the first rejection. If all seven pass, the last verifier row supplies a bonus token. Accepted stop tokens terminate the sequence. The target's committed intermediate features also update the drafter's context ring.

Eight evaluated rows therefore do not guarantee eight emitted tokens. Useful tokens per cycle depend on acceptance; end-to-end time also includes the drafter, target verification, host selection, context updates, prompt prefill and context changes.

For greedy decoding the acceptance rule checks each proposal against the target argmax. Equivalence to plain greedy decoding depends on the same target weights, numerical behavior, state and generation policy: a batched graph can differ numerically from a single-token path. For sampling the implementation treats the proposed path as deterministic, accepts a proposal with its target probability and samples the target distribution excluding it on rejection. This preserves the specified target distribution in the algorithmic construction; it does not promise identical random draws or token sequences from the same seed across implementations. Validate the actual runtime, including penalties, stops and context transitions. A fluent response or high acceptance alone does not establish quality.

Drafts are proposed deterministically, so a sampled draft is accepted with the target probability of that token; at temperature 0.7 to 1.0 this accepts fewer drafts than greedy decoding. The [DFlash2 sampling plan](research/DFLASH2_SAMPLING_PLAN.md) describes exact speculative sampling with the drafter's own distribution and the tests needed before changing this rule. Verifier cost does not depend on temperature, and an 8-row verifier stays the fastest end to end ([verifier block length](verifier_len.md)).

The source is [qwen38_server.py](../scripts/qwen38_server.py), particularly `draft_cycle`; [qwen38_spec_unit_test.py](../scripts/qwen38_spec_unit_test.py) is the historical CPU sampling experiment. It is not a hardware validation or a routine fast test.

## Exact release pairing

The prepared target is `mix25in_mixr_lr64mix`. The locally identified deployed drafter is `dflash2_lut4_gptq.aimodel`, with sidecar metadata referencing that target export and its `lm_head.safetensors`. It retains the earlier `drafter_lut4_gptq_q7_cal` calibration and uses mask scale `0.7`. A later recalibration experiment is not evidence that its output was deployed.

The matching contract is:

- Target hidden size **5120**, vocabulary **248320**, and tap order **[5, 19, 33, 47, 61]**. Five target features concatenate into **25600** values per position.
- Drafter **five layers**, **32 attention heads**, **8 KV heads**, head dimension **128**, selector rank **256** and candidate top-k **16**.
- Draft width **8**, new-context capacity **8**, prefill-context update width **64**, sliding window/ring capacity **2048**. The ring window is not the target's maximum context length.
- Target verify width **8**, prefill width **64**, matching embedding/tokenizer/head identity. Prepared target context entries are 8K, 16K, 24K, 32K, 48K and 64K; the largest usable KV capacity is **65472** rows.

Match identities and shapes, not filenames alone. A stale head previously reduced acceptance while appearing to agree with a reference using the same stale head. Current metadata/source checks help detect mismatches; they do not independently prove every compiled weight's lineage. Rebuild or revalidate the pair if the target head, quantization, taps or tokenizer changes.

## Self-contained assets

The [bundle workflow](HUGGING_FACE.md) places these beside `model/` and `coreai/`:

```text
drafter/
  dflash2_lut4_gptq.aimodel/
  dflash2_lut4_gptq.json
  config.json
  selector.safetensors
  LICENSE
  NOTICE
  DFLASH2_SOURCE.json
```

`selector.safetensors` contains `candidate_selector.predecessor_codebook` and `candidate_selector.successor_codebook`. The runtime also accepts those keys in a legacy `model.safetensors`; the release uses the compact file, so original BF16 drafter shards are unnecessary for serving. The target's prepared FP16 embedding table supplies anchor embeddings. The Core AI package contains the converted drafter body/head. Do not substitute the old Core ML RTN package or a debug package without a head.

The separate upstream checkpoint is [ProCreations/Ternary-Bonsai-2-27B-DFlash2 at 4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/tree/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b). Preserve its Apache 2.0 LICENSE and upstream NOTICE with the converted/extracted assets. [Attribution](ATTRIBUTION.md) distinguishes this evidence from Qwen target provenance and remaining source-code licensing work. The [DFlash2 notebook](../DFLASH2_ANE_PLAN.md), [session lessons](SESSION_LESSONS.md) and [M3U archive](../pipelines/m3u/README.md) retain the research history.

## Serve and diagnose

Build the Swift bridge in a compatible environment, then use the downloaded bundle:

```sh
bash coreai/swift_bridge/build.sh
python forge.py serve --runtime coreai \
  --model /path/to/bundle/model --build /path/to/bundle/coreai --ctx 16384
```

The launcher resolves the sibling `drafter/` package and configuration by default and fails if required assets are absent. [SERVER.md](SERVER.md) lists every startup option, wrapper variable and request default. For independently organized assets, set `--draft /path/to/dflash2_lut4_gptq.aimodel` and `--drafter /path/to/config-and-selector-directory`. `--plain` explicitly disables speculation for a diagnostic run. Keep diagnostic reports separate from release performance results.

The Core AI path uses the Swift bridge, bonded compile mode `2`, and a default `DRAFT_GAP_MS=3` minimum interval after verification before the next draft submission. The drafter fixes PyTorch's host thread count to one to avoid contention observed on the research machine. These are measured mitigations for that stack, not permanent hardware requirements or universal speed guarantees.

## How ANE context expansion is implemented

The prepared Core AI graph uses **fixed-shape context entry points**, rather than changing one ANE graph's tensor shapes at every token. Each four-layer chunk contains `v8_8k`, `v8_16k`, … for eight-row verification and `p64_8k`, `p64_16k`, … for 64-row prefill. The output head stays `h8`. Context entries in a chunk share its model weights; selecting a larger entry does not load another 27B weight copy. See the [builder](../coreai/qwen38_coreai_build.py) and [runtime](../scripts/qwen38_coreai_model.py).

The server restricts the growth ladder to advertised entries at or below `--ctx`. With `--ctx 16384`, a fresh conversation starts at 8K and can grow to 16K. It does not start every request at 16K. The current loader still obtains handles and output buffers for **all manifest entries**, including larger entries outside that ladder, so this flag is a usable-context cap, not a guarantee of strictly 16K-only loading.

On each prefill or verifier call, `fit(pos + n)` checks whether the committed position plus the incoming rows fit the active entry. If they do not, it selects the smallest allowed entry whose usable capacity fits. `resize` then:

1. Allocates new K/V IOSurface buffers for the full-attention layers and a mask with the new history length. The default export uses FP16 K/V; the experimental V8 export uses FP16 K, INT8 V and FP16 token/head scales.
2. Copies the cached prefix into the new buffers, including V codes and their scales for V8. It retains rows up to `min(hi, new_capacity)`, where `hi` tracks cached rows needed by snapshots as well as the current committed position.
3. Preserves the current position, pending commit count and DeltaNet recurrent/convolution state; resizing does not replay the prompt or reset these states.
4. Clears cached Swift binding plans because they reference the old KV buffers. The next call binds the chosen `v8_*` or `p64_*` functions to the new buffers and runs the chunks/head through the bridge.

Visibility is controlled by the current position and mask; retained rows beyond that position do not automatically become visible history. Verification commits only the accepted prefix through `accept(k)`. The KV allocation/copy can temporarily hold old and new buffers together, and the first use of a new entry can incur additional runtime specialization. The runtime records `[ctx]` transitions with position, copied-row count and elapsed milliseconds. These costs belong in latency and peak-memory measurements.

Ignoring padding and runtime scratch space, full-attention KV storage scales as `2 × attention_layers × KV_heads × history_rows × head_dim × 2 bytes` for FP16 K/V. Weight storage and DeltaNet state do not scale by that same context factor. Total loaded memory also includes programs, per-entry buffers, the drafter and host assets; KV arithmetic alone is not a full-memory estimate.

The [experimental V8 option](KV_CACHE_V8.md) lowers the logical K/V/scale payload from 64 KiB to 48.125 KiB per history position for this target. It requires a matching Core AI export and preserves the same DFlash2, valid-row prefill writes and speculative accepted-prefix commits. The stock download remains FP16; changing a runtime flag does not convert its model inputs.

The largest prepared entry is labeled 64K but has **65,472 usable history rows**. The builder reserves the largest block width (64) so `[history | block]` stays within its observed 65,536-element compilation boundary. This is a constraint of the recorded graph/compiler path, not proof of a permanent universal ANE architecture limit. Speculative HTTP requests additionally reserve eight verifier-write positions before granting an output budget. If no allowed entry fits, direct runtime calls fail and the server rejects an oversized prompt or limits generation; it does not silently expand beyond the configured ladder.

`reset(shrink=True)` returns to the smallest entry. Restoring a shorter cached snapshot can also shrink the buffers; rows dropped by shrinking make longer snapshots unrestorable, requiring a fresh prefill. Context expansion itself does not summarize the conversation. Pi or Terminus compaction constructs a new prompt and is a separate operation.

**DFlash2 has a separate 2,048-row ring.** Committed target features update that ring at absolute positions. Slot `position % 2048` is reused, and masks select rows within the drafter's sliding window. Increasing the target from 8K to 16K does not enlarge this ring or turn it into a 16K drafter KV cache. Target context size and draft acceptance must be measured separately.

## Context, warm-up and serving policies

`--ctx 16384` caps a growing 8K→16K ladder; it does not pin every call to the 16K entry. Speculation reserves eight positions for verifier writes, so the server may reduce the requested output cap to fit. Record the actual prompt length, granted cap and entry transitions.

The first start of a build on a macOS build compiles every target and drafter package for the ANE once and prints `[ANE compile]` progress (packages left, estimate, time left, safe-stop and faster-option hints); `python forge.py compile --build <dir>` does this without serving. Startup loads packages but does not execute a full inference warm-up. A short first request exercises the small-context prefill, verifier, head and drafter; a later larger-context entry can still be cold. Report compilation/load time, cold first-request time and warmed serving time separately. Count prompt prefill and summarization calls in task wall time, not just generation throughput.

## Eviction, memory pressure and stalled calls

The research found several distinct failure modes. Calling all of them "model eviction" hides which workaround applies:

- **Leaked KV buffers during resizing:** on September 29, the server using snapshot `89db7ac` was killed for low swap after 160 transitions between 8K and 16K. The discarded KV allocations total about 120 GiB; Activity Monitor showed a 122.27 GB Python process footprint and 95.41 GB of system swap. Two ownership defects were isolated: the Python bridge cached `self.np = np.asarray(self)`, creating a `Buffer → ndarray → Buffer` cycle, and native IOSurface operations left autoreleased objects alive on the Python calling thread. Fixing the Python cycle alone still grew the live probe from 3.66 to 8.50 GB before its guard stopped it. A separate no-model test released eight 32 MiB buffers: without draining the allocation path footprint rose about 270 MB; with a pool it rose about 1 MB. Further isolation required pools around both `cai_buffer_create` and `cai_buffer_address`; pooling only creation or only release was insufficient. `.np` now returns a zero-copy view without caching it on the buffer, and those two Swift entry points drain scoped `autoreleasepool`s. The returned buffer handle remains retained, and external views continue to keep storage alive. Pull both sources and **rebuild the native bridge** with `bash coreai/swift_bridge/build.sh`; replacing the Python wrapper alone is insufficient. Four [ownership regression tests](../tests/test_bridge_ownership.py) cover final release and repeated replacement; sustained model-level transition validation is a separate check. A single-entry ladder avoids resize allocations as an interim workaround, but `--ctx 16384` still allows an 8K→16K ladder and does not pin the entry. Preserve failed attempts and exclude infrastructure interruptions from quality scores.

- **Historical Core ML function residency:** separately loaded context/prefill functions retained separate weight copies despite sharing files on disk. Loading two large sets exceeded the research machine's memory budget and alternating calls stalled; eviction was described as probable. The release uses Core AI entries sharing a chunk's weight copy. That addresses the duplication found in that experiment, but does not guarantee residency under every workload. See [the Core ML investigation](../FEEDBACK_ANE_MULTIFUNCTION_MEMORY.md).
- **Compiled cache unavailable at load:** under a nearly full disk, previously compiled Core AI packages failed `load_function` with `Foundation._GenericObjCError error 0`. Moving the affected cache aside and recompiling restored loading. Purgeable ANE program-cache eviction was the suspected explanation, not a directly measured event. Keep adequate free disk. The target runtime retries a failed source `.aimodel` load once after purging caches keyed by that package's `main.hash`; it does not purge on every request. This retry does not cover every failure, compiled-package overrides or the drafter loader. See the [original observation](../COREAI_PORT_NOTES.md#measured-m6-macos-27-2026-09-26) and [`pick_package`/`_load`](../scripts/qwen38_coreai_model.py).
- **Excess resident program memory:** the default MPSGraph compilation retained bonded and nonbonded variants. Mode `2` reduced program memory with matched outputs in the recorded tests. The target loader aligns stale cached specializations before loading because setting the environment variable alone cannot convert an existing specialization and can otherwise cause an assertion. This is an undocumented, version-sensitive mitigation; do not apply the analogous setting to Core ML packages based on this result.
- **Per-call output allocation:** the historical Core AI Python binding accumulated IOSurface allocations and eventually failed. Use the Swift bridge's persistent input/output buffers for sustained serving. Its finite stability tests do not establish that every future framework version is free of leaks.
- **Dispatch contention:** multithreaded host top-k work delayed target dispatch; the Core AI drafter fixes PyTorch to one thread. Drafter submissions immediately after target verification also showed intermittent 300–700 ms stalls. `DRAFT_GAP_MS=3` removed those drafter stalls in the small recorded A/B, while rare verifier stalls remained. Neither observation proves that programs were evicted.

**Repeated multi-second verifier calls need a new diagnosis.** Low free memory, compression or swap activity makes residency pressure plausible, but those counters alone do not prove repeated ANE program eviction. Separate load/recompile time from warmed `draft`, `verify`, host selection and context-update timing. Capture system memory and disk headroom, concurrent workloads, compile mode/cache metadata and ANE placement. When available, correlate `aned` program-load events with slow calls; a process's disk/page-in counters do not account for all device or daemon activity.

First compare a fixed short prompt on the same pinned target/drafter pair with competing model servers and benchmark VMs stopped. Change one condition at a time and retain failed-run records. If a smaller-context **compiled package** is tested, verify its target/head identity and numerical parity: `--ctx 16384` on the 12-entry package limits usable context but does not remove its largest-entry scratch allocation. A measured chunk needed 151 MB more scratch for the 8K–64K package than the 16K/24K package; the full-target 2.4 GB figure is a projection from that chunk. Filtering Python handles alone is not an established fix. Recompile only an affected cache for a verified load or placement failure; indiscriminate cache deletion during live serving is not a latency workaround. See [program/scratch measurements](../COREAI_PORT_NOTES.md#context-ladder-8k-64k-12-entry-points-per-chunk-2026-09-28) and [session lessons](SESSION_LESSONS.md#runtime-memory-different-problems-need-different-mitigations).

With both fixes, a bounded live Core AI + GPTQ DFlash2 probe passed **32 context transitions / 16 roundtrips**: all old buffers released, 72 KV rows preserved exactly, target logits identical across 8K/16K and finite drafter outputs. Process `ri_phys_footprint` changed from **3,127,332,248 to 3,131,608,496 bytes** (about **4.3 MB**); the sampled transient peak was **4,205,875,328 bytes**. This counter excludes other processes and ANE system allocations. The [lifetime result record](results/coreai_buffer_lifetime_2026-09-29.json) preserves the failed Python-only probe, native isolation, rebuilt-source/binary hashes and successful bounded check. It does not establish multi-hour serving stability or validate a freshly downloaded release bundle.

### Cached GPU placement can look like an ANE stall

A September 29 release-preparation diagnostic isolated a repeated 5.7-second verifier stall to `chunk_L00-03`. Its cached specialization placed all 12 entries on GPU; the other 15 target chunks, head and drafter remained fast. A Core AI worker waited for Metal command completion. Stopping the local benchmark VM did not remove the stall. This was a placement/cache failure, not demonstrated ANE model eviction.

The bad cache's creation time matched a failed sandbox launch; the original reason that specialization chose GPU remains unresolved. Preserving only that chunk's cached specialization and recompiling the unchanged source in an environment with permitted hardware access restored ANE placement. Synthetic 8K/16K probes then measured that chunk's prefill at 22.7–24.7 ms and verification at 6.0–7.7 ms; full target verifier plans took 106–141 ms. These are bounded diagnostic timings, not release throughput, acceptance or quality benchmarks.

To investigate this failure:

1. Attribute latency to exact packages and functions. The Swift bridge's `Plan.run(times=True)` returns native wall milliseconds per binding while retaining the existing combined plan.
2. Inspect every loaded entry's cached specialization. The successful entries had `mps.fullyPlacedOnANE`, `mps.noGPUActivity` and ANE region callees; the failed entries had GPU region symbols and `mps.disableNDX`. Neither preferred ANE compute, compile mode `2`, nor the top-level `GPU adapter present: NO` field established ANE placement. Cache-mode alignment checks the mode, not the selected device. These internal fields can change with the OS and compiler; interpret them alongside function timing and driver evidence.
3. Correlate driver records with those function timings. A passive capture during inference is:

   ```sh
   /usr/bin/log stream --level debug \
     --predicate 'sender == "AppleH16ANEInterface"'
   ```

   `Using Bonded NID` followed by ANE0 and ANE1 completion of the same `removeRequestByUUID` UUID proves bonded execution for that request. `fNumANEs: 2` alone describes available units. Bonded records do not prove that every target chunk reached ANE; a GPU chunk is absent from that request stream.
4. For a verified bad specialization, stop the owned model process and preserve that package's exact cache entry, identified by OS build, executable and source `main.hash`. Re-specialize it with the intended settings and check each entry's placement before repeating the bounded timing probe. Retain the old cache and failed-run evidence. Do not wipe unrelated caches or clear caches during live serving.

The read-only [cache audit utility](../coreai/inspect_coreai_cache.py) checks these entry attributes without loading or compiling models:

```sh
python coreai/inspect_coreai_cache.py \
  --model-dir /path/to/bundle/coreai \
  --drafter /path/to/bundle/drafter/dflash2_lut4_gptq.aimodel \
  --executable /path/to/inference/python --strict
```

Use the serving process's OS build and executable identity. For an archived cache, set `--os-build` and `--cache-root`. The JSON report marks fully ANE, GPU regions, or unknown evidence; `--strict` exits `1` for GPU or unknown/missing placement, and malformed inputs exit `2`. Full ANE status requires recognized ANE region symbols for each expected entry as well as the manifest attributes. Run it after a permitted specialization has created the cache; a missing cache is not a placement failure. An existing target `.aimodelc` override produces `unknown`: this utility inspects source caches and cannot establish placement of a selected compiled override. If several specializations exist, the audit conservatively checks all of them and cannot determine which one a live process chose. It does not establish numerical correctness or live hardware execution.

## Stops and generation policies

EOS, user stop strings and output caps apply to returned tokens. A stop string or cap can interrupt a speculative group whose accepted prefix has already been committed. The next request may restore a snapshot or re-prefill instead of extending the current cache. Include these boundaries, new conversations, context growth and repeated turns in release validation.

DRY penalties, loop guards and forced thinking closure affect generation independently of speculation. For quality benchmarks, record their settings and use the declared matched policy. The historical service used DRY off, loop guard six and reasoning budgets; a benchmark with guards disabled is a different serving policy. A 4K cap with thinking enabled can produce no final answer. Do not claim the drafter eliminates that limitation.

## Release validation

Require a pinned download containing both components, verified hashes/provenance, and a speculative smoke result that reports the drafter path and cycles. Add matched plain-versus-speculative greedy tests, sampled-distribution checks, stop/cap/cache/context tests, actual task scoring and sustained memory/placement measurements. Report accepted drafts per cycle, useful output tokens per verifier call, phase timings and complete loaded memory. Integrity checks and mocked tests alone do not establish ANE placement, speed or release quality.
