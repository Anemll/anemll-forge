# Pi coding sessions with the Core AI + DFlash2 release

This example was checked against **Pi 0.87.1** on 2026-09-29. It connects Pi's coding tools to the local **Core AI target + matching DFlash2 drafter**, using model ID `qwen38-27b-ane` at `http://127.0.0.1:8765/v1`. It does not launch a second model or use Pi's llama.cpp integration. See [the release pairing guide](SPECULATIVE_DECODING.md) first.

The profile declares **16,384 total context tokens**, including instructions, tool schemas/results, previous messages, preserved reasoning and the current response. It is not a 16K output allowance. The prepared examples are [models.json](../examples/pi/models.json) and [settings.json](../examples/pi/settings.json). They contain a deliberately dummy local API key, no credentials, and no personal paths.

## Start the paired server

Build the Swift bridge in the compatible environment and start the downloaded release:

```sh
bash coreai/swift_bridge/build.sh
python forge.py serve --runtime coreai \
  --model /path/to/bundle/model --build /path/to/bundle/coreai --ctx 16384
```

The matching sibling `drafter/` is required by default. Do not pass `--plain` for coding sessions intended to use release performance. The launcher binds loopback and the server serializes requests. Wait for the serving message; initial compilation and the first request can be slow. `--ctx 16384` permits the runtime to grow through 8K and 16K entries.

This server's defaults are presence penalty **0**, DRY **0**, loop guard **6** and the documented reasoning-closure policy. The Pi example explicitly sends temperature **0.6**, top-p **0.95**, top-k **20**, presence penalty **0** and DRY **0**. These are a declared practical starting policy, not measured optimal coding settings. The loop guard is server-wide, not a supported Pi request override. It can terminate legitimate repeated output; inspect the server log when a response stops unexpectedly. To disable it for a declared policy comparison, use the underlying server's `--loop-guard 0` option when starting that server. Do not silently mix those results with the default policy.

## Install Pi and apply the configuration patches

The Forge Pi changes are the supplied provider/model and context-management configuration. Pi **0.87.1** already supports the checked request/replay behavior; no Pi source-code patch is required for this version. The four runtime modules used by the mock validator were byte-identical to the published npm packages. Use Node.js **22.19 or newer** and install into a dedicated directory:

```sh
export FORGE_PI_ROOT="$HOME/.local/share/anemll-forge/pi"
npm install --prefix "$FORGE_PI_ROOT" @earendil-works/pi-coding-agent@0.87.1
export PATH="$FORGE_PI_ROOT/node_modules/.bin:$PATH"
export PI_MODULES="$FORGE_PI_ROOT/node_modules"
node examples/pi/validate_pi.mjs
```

The validator uses mocked transport, not the model server. Upgrade older Pi versions to this checked version rather than applying undocumented patches to their installed JavaScript. Then apply the configuration to an isolated profile below.

## Use an isolated Pi profile first

Run the following from the Forge checkout. Choose a new empty profile directory; this avoids overwriting your normal Pi providers, settings or sessions:

```sh
mkdir -p /path/to/new-forge-pi-profile
cp examples/pi/models.json examples/pi/settings.json /path/to/new-forge-pi-profile/
export PI_CODING_AGENT_DIR=/path/to/new-forge-pi-profile
export PI_LIVE_THROUGHPUT_DIR="$PI_CODING_AGENT_DIR"
pi --offline --list-models qwen38
```

The listing should show provider `ane-qwen38`, model `qwen38-27b-ane`, roughly **16.4K context / 4.1K max output**, thinking support and no images. Then enter the project you intend Pi to work on and start a fresh session:

```sh
cd /path/to/your/coding-project
pi --offline --provider ane-qwen38 --model qwen38-27b-ane --thinking off
```

`--offline` disables Pi startup network operations; it does not disable HTTP calls to the local model or network operations performed by coding tools. The profile changes Pi configuration/session storage, not the files its tools can edit: the working directory is still your coding project. Start with one small, verifiable change and inspect its diff/tests.

For an existing profile, merge the single provider and exact per-model compaction override into your files; do not replace whole files containing other providers or credentials. Pi's `/model` reloads model configuration; restart after changing compaction settings. A resumed session can restore its earlier model/thinking selection, so check those settings or start fresh. [Official Pi model configuration](https://pi.dev/docs/latest/models)

## Larger server contexts (32K / 64K)

The server's `--ctx` can be raised to the prepared 32K/48K/64K entries; the runtime still starts small and grows the ladder. The supplied profile is deliberately 16K. To match a larger server context, sync the profile budget:

```sh
python scripts/qwen38_pi_config.py --ctx 65536 --pi-dir "$PI_CODING_AGENT_DIR" --build coreai
```

This updates `contextWindow`, `maxTokens = clamp(ctx/4, 2048, 16384)` and the per-model compaction override (`reserveTokens = maxTokens + 4096`, `keepRecentTokens = min(ctx/8, 8192)`). Reopen `/model` to reload models.json; restart Pi for compaction. `scripts/qwen38_server.sh` runs the same sync automatically after a successful start unless `PI_SYNC=0`; set `PI_DIR` to target a profile other than `~/.pi/agent`.

The helper accepts only the prepared 8K/16K/24K/32K/48K/64K windows. Both profile files must contain valid JSON objects; missing or invalid settings are rejected before either file changes. Updates preserve other providers and settings, retain original/previous backups, and replace each file atomically with rollback on write failure. If the wrapper returns while cold loading continues in the background, run this helper after the server becomes ready.

A client/server window mismatch adds no capability: a 16K Pi profile against a 64K server still compacts at 16K. At 64K the usable capacity is 65,472 rows and the paired target + drafter wire roughly 25 GB, so validate the server first and run one client at a time. These budgets mirror Pi's client-side clamp; they do not change the server's eight reserved verifier positions.

## Thinking without consuming the whole response

The example keeps `reasoning: true` so Pi offers both `off` and thinking modes. It maps Pi's thinking selection into `chat_template_kwargs.enable_thinking` and `reasoning_effort`, with `preserve_thinking: true`. It deliberately uses `thinkingFormat: "chat-template"`; the top-level Qwen thinking format is not the server's interface.

For ordinary edits start with `--thinking off`. For a task needing more deliberation, use `--thinking low` or select low through `/thinking`. With the full **4096-token response ceiling**, this Pi version sends:

- **Off:** `enable_thinking: false`, no explicit thinking budget.
- **Low:** `enable_thinking: true`, effort `low`, `thinking_budget: 2048`.
- **Medium/high:** mapped effort `medium`/`xhigh`, but the budget is capped at **3072**, leaving nominal room for at least 1024 answer tokens within the same response ceiling.

Pi reduces those budgets when its context estimator reduces the response ceiling. `thinkingTokenBudgetField: "thinking_budget"` is important: it makes Pi send the budget in the field Forge actually reads. A generic `thinking_token_budget` would not control this server.

When the explicit budget is reached, Forge injects a reasoning-closure phrase and `</think>` into model context before continuing. This is a generation policy, not transparent speculative decoding. The injected tokens also consume context. It improves the chance of an answer/tool call but cannot guarantee one. `thinking_budget: 0` disables this closure; use it only for a deliberately unbounded-thinking comparison within the shared total cap, where empty final answers remain possible. Do not put a fixed `max_tokens` in `samplingParams`: that would override Pi's context-aware clamp.

## Compaction and tool continuity

Pi 0.87.1 clamps response capacity to the smaller of the model cap and `contextWindow - estimated_prompt - 4096`, floored at one. The example uses `maxTokens: 4096`, compaction reserve **8192** and recent-history retention **2048** for this exact provider/model. Automatic compaction therefore starts around **8K estimated context**, before Pi's safety margin crowds out the response. The default global 16K reserve/20K retained history is inappropriate for this 16K model. Branch-summary reserve is also set to 8192 in the isolated profile. [Official settings](https://pi.dev/docs/latest/settings) and [compaction reference](https://pi.dev/docs/latest/compaction)

Estimates are not exact tokenizer accounting. Tool output can overshoot the threshold, and the server reserves eight verifier positions and may lower the requested output cap further. Keep file reads and command output focused; compact before a large new investigation and preserve a short written task plan. If compaction fails, retain the session and start a fresh task with an explicit summary rather than inflating advertised context beyond the server's real capacity.

The server renders the original Qwen tool-aware chat template, converts its XML-style tool output to OpenAI `tool_calls`, and returns thinking separately as `reasoning_content`. Pi 0.87.1 recognizes that field and replays it on assistant messages; `preserve_thinking: true` keeps it in the template. Do not flatten reasoning into final text, remove tool-call IDs or discard tool results in a proxy. The template, tools and retained reasoning all count toward context. The config sends a `system` role, avoiding unsupported `developer` role assumptions.

Use **one Pi session at a time** against this single-request server. Do not run Terminal-Bench, smoke tests or another coding client concurrently: their requests contend for the ANE and can disturb cache reuse and timing. Pi tool execution and compaction calls are part of coding-session wall time; standalone decode throughput is not an estimate of whole-task time.

## Optional: a live decode-TPS/TTFT status footer

[`examples/pi/extensions/live-throughput-status.ts`](../examples/pi/extensions/live-throughput-status.ts) adds a status-line footer showing live decode tok/s, TTFT, final token counts and (with a Prometheus-exposing backend) server-side prefill throughput. This server has no Prometheus `/metrics` endpoint, so leave `metricsUrls` empty for `ane-qwen38`; the footer still shows client-observed decode TPS, TTFT and prompt size. For the isolated profile, copy the extension into `$PI_CODING_AGENT_DIR/extensions/` and keep `PI_LIVE_THROUGHPUT_DIR` set to that profile so its calibration metadata stays there. Run `/reload` after installation. See [`TPS_PI_EXTENSION.md`](../examples/pi/extensions/TPS_PI_EXTENSION.md) for configuration and tests.

## Checks performed and remaining limits

The installed Pi parser accepted the example schema. An isolated offline `--list-models` run resolved the expected provider, context, output cap and capabilities. A mocked transport exercised off/low/medium/high request payloads, bounded thinking, explicit sampling, streamed reasoning/tool-call parsing, replayed reasoning/tool results and the context clamp. It made no model-server calls. See [the portable mock validator](../examples/pi/validate_pi.mjs); set `PI_MODULES` to your installed Pi `node_modules` directory and run it with Node.

The configuration checks do not establish live coding quality or throughput. Recheck compatibility on other Pi versions and perform a real paired-server tool round trip before relying on a long session.
