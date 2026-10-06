# Server startup and options

This page lists every way to start the Qwen3.8-27B ANE server, every startup option with its default, and the per-request generation defaults. The README has the short version; [SPECULATIVE_DECODING.md](SPECULATIVE_DECODING.md) explains the verifier, context ladder and serving policies behind them.

## Three ways to start

| Path | Use it for | Runs |
| --- | --- | --- |
| `python forge.py serve ...` | A foreground session; stop with Ctrl-C | Validates the checkpoint, build manifest, context, KV format and target/drafter pair, then runs `scripts/qwen38_server.py` |
| `scripts/qwen38_server.sh start` | A background server with a PID file, a log, status checks and the Pi profile sync | The same `forge.py serve` command, configured through environment variables |
| `scripts/qwen38_server.py ...` directly | Changing server-wide generation defaults (below) | No launcher validation; needs the launcher's environment |

All three serve the same OpenAI-compatible API: `POST /v1/chat/completions` (streaming, tools), `GET /v1/models` and `GET /health`. One request runs at a time. There is no authentication; the default listen address is loopback.

## Startup options

`forge.py serve` flags and the wrapper's environment variables set the same things:

| Option | `forge.py serve` | Wrapper variable | Default | Notes |
| --- | --- | --- | --- | --- |
| Bundle | | `FORGE_BUNDLE` | `~/Models/anemll-forge-qwen3.8-27B` | Wrapper only; sets the two paths below |
| Checkpoint (tokenizer, chat template, embedding) | `--model` (required) | `MODEL` | `$FORGE_BUNDLE/model` | |
| Core AI target build | `--build` (required) | `BUILD` | `$FORGE_BUNDLE/coreai` | Directory with `manifest.json` |
| Context cap | `--ctx` | `CTX` | `16384` | Must be a context entry in the build's manifest; the wrapper also accepts `8K` to `64K`. The runtime starts at the smallest entry and grows up to this cap. The release bundle has 8K, 16K, 24K, 32K, 48K and 64K; a conversion has the entries passed to `--ctx` when it was built |
| KV cache format | `--kv-cache-dtype` | `KV_CACHE_DTYPE` | `auto` | `auto` follows the manifest's default; `fp16` or `v8` require a build that has that format and are rejected before loading or stopping a server otherwise |
| Speculative drafter | `--draft PATH` or `--plain` | `DRAFT` | on: `drafter/dflash2_lut4_gptq.aimodel` next to the build | `--plain` (`DRAFT=off`) is a slower target-only diagnostic |
| Drafter config and selector | `--drafter` | `DRAFTER` | The drafter package's directory | |
| Listen address | `--host` | `BIND_HOST` | `127.0.0.1` | `0.0.0.0` exposes the unauthenticated API to the network |
| Port | `--port` | `PORT` | `8765` | Separate instances need distinct ports |
| Runtime | `--runtime` | | `coreai` | `coreml --plain` is the historical Core ML diagnostic; the wrapper always uses Core AI |
| Print without running | `--dry-run` | `check` subcommand | | `--dry-run` prints the server command and environment as JSON |
| Python | (the one running `forge.py`) | `PY` | `.venv/bin/python` in the checkout, else `python3` | Must be the inference environment; the ANE compile cache is keyed by its executable name |
| Log and PID file | | `LOG`, `PIDFILE`, `ANEMLL_FORGE_STATE` | `~/.anemll-forge/server.log`, `server.pid` | `restart` keeps the previous log as `<log>.prev` |
| Pi profile sync | | `PI_SYNC`, `PI_DIR`, `PI_BUILD` | `1`, `~/.pi/agent`, the build's directory name | Runs after the server is serving; `PI_SYNC=0` skips it |
| Startup watch | | `START_WAIT_S` | `7200` | How long `start` and `restart` follow the startup |

The wrapper does not accept other server flags (`EXTRA_ARGS` is rejected) so the managed port and context cannot be overridden by accident.

### Wrapper subcommands

```sh
scripts/qwen38_server.sh start      # validate, start in the background, follow the startup
scripts/qwen38_server.sh restart    # validate the new settings first, then stop and start
scripts/qwen38_server.sh stop       # stop only the server recorded in the PID file
scripts/qwen38_server.sh status     # PID and /health
scripts/qwen38_server.sh check      # validate checkpoint, build, context, KV format, drafter and bridge only
scripts/qwen38_server.sh log        # follow the server log
```

Settings are per command: `CTX=64K scripts/qwen38_server.sh restart` restarts with a 64K cap, and a later plain `restart` returns to the 16K default.

## What startup does and prints

1. The banner: listen address, model directory and its context entries, KV format, drafter, thinking default, budgets, sampling and repetition defaults.
2. `target graph: GDN_FAST=1 ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096 | build <path>` for a build with the faster graph, or `(not recorded in manifest: release defaults)` for packages built before it (published revisions up to `cd7dfc6`) ([Faster ANE graph](../README.md#faster-ane-graph-conversion-default)).
3. `KV cache: FP16 K / INT8 V + FP16 token/head scales` (V8) or `FP16 V`.
4. On the first start of a build on a given macOS build, `[ANE compile]` lines: packages still to compile, an estimate (about 20 to 25 minutes for the full target on M6), per-package progress with time left, and build options that compile faster. Stopping is safe at any point; finished packages stay cached and the next start resumes. `python forge.py compile --build <dir>` compiles without serving.
5. Loaded target chunks, the head and the drafter, then `serving OpenAI API on http://<host>:<port>/v1`.

The wrapper prints the compile and progress lines while it watches and ends with the target graph and KV cache lines. Ctrl-C only stops watching; the server keeps starting in the background.

### First-start compile: time and readout

A new model (a new conversion, a new download, or the same build after a macOS update) is compiled for the ANE once. Measured on the M6 on 3 October 2026 for the default build (V8 only, faster graph, 8K to 64K): the 16 target chunks took **22m48s**, 1m19s to 1m26s each with one at 2m33s. The head and drafter were already cached from the release bundle in that run; on a fresh Mac they compile too (estimated at about 30 s and 60 s, not measured separately). The next start loaded all 18 packages from the cache and was serving 2 s after launch.

The readout from that compile, shortened (`...` marks omitted lines):

```text
[22:37:16] target graph: GDN_FAST=1 ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096 | build <build>
[22:37:16] [ANE compile] 16 of 18 packages are not compiled yet for macOS 26A434 / python: compiling them once now (later starts load from the cache in seconds)
[22:37:16] [ANE compile] estimated ~27m33s in total for GDN_FAST=1 ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096, 1 KV format, 5 contexts; progress and time left are printed below
[22:37:16] [ANE compile] safe to stop at any time (Ctrl-C or qwen38_server.sh stop): each package is cached as soon as it finishes, and the next start resumes with the rest
[22:37:16] [ANE compile] quick test: convert with fewer contexts, e.g. --ctx 8192,16384 --pctx 8192,16384
[22:37:16] [ANE compile] compiling chunk_L00-03.aimodel (1/16): expected ~1m43s, ~27m33s left in total
[22:37:46] [ANE compile] chunk_L00-03.aimodel (1/16): 30s so far (expected ~1m43s); ~27m03s left in total
[22:38:38] [ANE compile] compiled chunk_L00-03.aimodel in 1m22s (1/16 done); ~20m27s left in total
...
[22:59:46] [ANE compile] chunk_L60-63.aimodel (16/16): 1m00s so far (expected ~1m26s); ~26s left in total
[23:00:04] [ANE compile] compiled chunk_L60-63.aimodel in 1m19s (16/16 done); ~0s left in total
[23:00:04] [ANE compile] done in 22m48s: 18 packages compiled and cached for macOS 26A434 / python; the server now loads them in seconds
```

How to read it:

- **Plan:** how many packages still need compiling, for which macOS build and Python executable name (the cache key), and the estimate for this build's graph, KV formats and contexts.
- **Hints:** stopping is safe, and options that compile faster: fewer contexts, or `python forge.py compile` ahead of serving.
- **Progress:** a `compiling` line per package, a heartbeat every 30 s (`ANE_COMPILE_HEARTBEAT_S`) with the time so far, and a `compiled ... in` line with the time left.
- **Time left** starts from a model of the build settings (here 27m33s, about 20% high) and is rescaled by the measured package times, so it is close after the first package. A slow package (2m33s here) raises it until the next ones finish.
- **Later starts** print `all 18 packages already compiled for this Mac (macOS 26A434): loading from cache`.

`GET /health` reports the loaded state: `kv_cache_dtype`, `kv_cache_formats`, `target_graph`, `context` (the cap), `active_context_entry`, `position`, and the last prefill and decode statistics.

## Generation defaults (per request)

Clients set these in each `POST /v1/chat/completions` request. The server-wide default in the last column applies when the request leaves the field out.

| Request field | Default | Server flag |
| --- | --- | --- |
| `chat_template_kwargs.enable_thinking` | `false` | `--think` makes `true` the default |
| Thinking for context-summary requests (Pi's compaction) | as requested | `--summary-no-think` (launcher: `SUMMARY_NO_THINK=1`) turns it off, with the non-thinking sampling defaults |
| `temperature` | 1.0 with thinking, 0.7 without; `0` decodes greedily | |
| `top_p` | 0.95 with thinking, 0.8 without | |
| `top_k` | 20 (`0` also means 20) | |
| `presence_penalty` | 0 | `--presence` |
| `dry_multiplier` | 0 (off) | `--dry`; base `--dry-base 1.75`, tolerated run `--dry-allowed 8` tokens |
| `max_tokens` or `max_completion_tokens` | 4096, reduced to fit the context cap | `--max-tokens` |
| `reasoning_effort` (or in `chat_template_kwargs`) | `medium` | `--think-budget low=2048,medium=6144,xhigh=12288` |
| `thinking_budget` | From the effort; `0` means no limit | |
| `seed`, `stop`, `stream`, `stream_options.include_usage`, `tools`, `tool_choice` | OpenAI semantics | |

Notes:

- The sampling defaults follow the Qwen3.8 model card. Speculative decoding keeps the target's sampling distribution. Presence and DRY penalties apply only when sampling, not to greedy decoding.
- The card suggests `presence_penalty` 1.5 for non-thinking chat. It lowers every token that already appeared, which is usually unhelpful for code; see [DFLASH2_SAMPLING_PLAN.md](research/DFLASH2_SAMPLING_PLAN.md).
- Thinking budget: `minimal` and `low` map to `low`, `medium` to `medium`, anything else to `xhigh`. When the budget is reached the server closes the thinking with Qwen's budget phrase. The budget is capped at `max(max_tokens - 4096, max_tokens / 2)`, so with the default `max_tokens` of 4096 thinking gets at most 2048 tokens. A 4096 cap with thinking can end without a final answer.
- The output cap shrinks to fit: a verify writes 8 rows past the committed position, so `max_tokens` is at most the context cap minus the prompt minus 8.
- The loop guard (`--loop-guard 6`, `0` turns it off) stops generation when the output tail repeats six times with the same period. It is server-wide only.

### Changing a server-wide default

`forge.py serve` and the wrapper keep the release defaults above. To change one for every request, run the server script directly with the launcher's validated command and environment:

```sh
python forge.py serve --runtime coreai --model "$FORGE_BUNDLE/model" --build "$FORGE_BUNDLE/coreai" \
  --ctx 16384 --dry-run
```

This prints the `argv` and `environment` that `forge.py serve` would use. Run that `argv` with those environment variables and append the server flags, for example `--think --max-tokens 8192`. `python scripts/qwen38_server.py --help` lists every flag.

## Runtime environment variables

These are measured mitigations and research switches, not settings most users need:

| Variable | Default | Effect |
| --- | --- | --- |
| `DRAFT_GAP_MS` | `3` | Minimum interval after a verify before the next draft is submitted |
| `COREAI_DRAFTER_COMPUTE` | `ane` (set by the launcher) | Drafter compute device |
| `MPSGRAPH_ANE_BONDED_COMPILE_MODE` | set by SoC policy (`1` on the M5 family, H17; `2` on M6+, H18) | ANE compile mode; an explicit value overrides the policy. Compiled packages are cached per mode. See [ANE compile mode policy](ANE_COMPILE_MODE_POLICY.md) |
| `ANE_COMPILE_HEARTBEAT_S` | `30` | Seconds between `[ANE compile]` progress lines during a long compile |
