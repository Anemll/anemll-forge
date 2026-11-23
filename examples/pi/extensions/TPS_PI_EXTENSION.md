# Live decode TPS, TTFT & prefill throughput for Pi (v2)

A model-neutral throughput footer for
[Pi](https://github.com/earendil-works/pi), validated against
`@earendil-works/pi-coding-agent` 0.85.1.

It observes Pi's standard assistant-stream events, so it works with local and
hosted models. For real server-side **prefill** throughput it can optionally
read Prometheus metrics (e.g. vLLM `/metrics`); prefix caching is handled
correctly, which client-side `usage.input` cannot.

Source: [gist.github.com/Anemll/f95a14877862f289e19b12586850eded](https://gist.github.com/Anemll/f95a14877862f289e19b12586850eded). This copy is vendored here so the ANEMLL Forge Pi example (see [PI_CODING.md](../../../docs/PI_CODING.md)) stays reproducible without depending on an external gist at session time; update both together when the gist changes.

## What the footer shows

Live, while streaming (estimated until the provider reports token counts):

```text
⚡ 33.8 tok/s TTFT: 0.25s   ~7 tok
```

Final, with exact usage, optional server prefill, prompt size, and elapsed:

```text
⚡ 62.8 tok/s TTFT: 0.25s   60 tok   800 t/s   Prompt: 1.0k tok   took 1s
```

| Segment | Meaning |
|---|---|
| `62.8 tok/s` | Decode rate (client-observed; first chunk excluded) |
| `TTFT: 0.25s` | Time from request send to first output delta |
| `~TTFT` | TTFT is approximate (request hook did not fire) |
| `60 tok` / `~7 tok` | Output tokens (exact vs. estimated) |
| `800 t/s` | **Server-side** prefill throughput (only if `metricsUrls` set) |
| `gen: 12.4 tok/s` | Optional: generation throughput incl. queue/prefill/TTFT |
| `Prompt: 1.0k tok` | Input + cache-write tokens |
| `took 1s` | Whole-request wall clock (`agent_start → agent_settled`) |

## Metric definitions

- **Decode TPS** — rate-first. Exact when a provider reports cumulative
  `usage.output` on intermediate chunks (Gemini-style / cumulative-usage
  OpenAI servers); otherwise a per-model, per-content (thinking vs. text/tool)
  `chars/token` ratio learned via EWMA and persisted across `/reload`. Both
  live and final rates **exclude the first chunk** from the numerator, since
  that chunk defines the window start (fencepost consistency). This is what
  keeps batched/buffered SSE streams from being overstated.
- **TTFT** — `before_provider_request` → first streamed delta.
- **Prefill TPS** — server-side only. Diffs Prometheus counters
  `vllm:prompt_tokens_by_source_total{source="local_compute"}` and
  `vllm:time_to_first_token_seconds_sum` across one request:
  `Δlocal_compute ÷ ΔTTFT_sum`. Prefix-cache hits are ignored, so this is the
  honest per-request prefill rate.
- **Generation throughput** (optional) — `output ÷ (request → message_end)`;
  includes queue + prefill + TTFT + generation, excludes tool execution.
- **`took X`** — `agent_start → agent_settled`, the whole user request
  including tool time. Scaled: `30s`, `1m 30s`, `2h 5m`, `1d 3h`.
- **Prompt** — uncached input + cache-write tokens. A `tokens/TTFT` ratio is
  intentionally *not* shown: TTFT contains network, queue, scheduling, and
  first-token decode, so it is not prefill throughput.

## Install

Copy the bundled [`live-throughput-status.ts`](./live-throughput-status.ts)
into `~/.pi/agent/extensions/`:

```bash
cp examples/pi/extensions/live-throughput-status.ts ~/.pi/agent/extensions/
```

Or fetch the latest revision directly from the source gist:

```bash
curl -fsSL https://gist.githubusercontent.com/Anemll/f95a14877862f289e19b12586850eded/raw/live-throughput-status.ts \
  -o ~/.pi/agent/extensions/live-throughput-status.ts
```

Then run `/reload` in Pi. If an older `qwen38-mtp-tps-status.ts`-style v1
extension is already installed, rename it out of `extensions/` (Pi only
auto-discovers `.ts` files there) rather than running both — they duplicate
the same status-footer job.

## Tests

[`tests/live-throughput-status.test.ts`](./tests/live-throughput-status.test.ts) covers the
calibration, fencepost, lifecycle, prefill-parsing, generation-metric, sliding-window and server-exact
(`/health`) paths (30 assertions). Run it with Node 22.18+ (built-in TypeScript stripping) or Bun:

```bash
cd examples/pi/extensions/tests
node live-throughput-status.test.ts      # or: bun run live-throughput-status.test.ts
# 30 passed
```

## Configuration

Optional `~/.pi/agent/live-throughput-config.json`:

```json
{
  "metricsUrls": {
    "gx10-vllm": "http://192.168.1.68:8888/metrics"
  },
  "showGenerationTps": false
}
```

| Field | Default | Description |
|---|---|---|
| `charsPerTokenSeed` | `4` | Initial chars/token estimate |
| `ewmaAlpha` | `0.3` | Calibration learning rate |
| `ratioMin` / `ratioMax` | `1` / `16` | Sanity bounds for learned ratios |
| `updateIntervalMs` | `200` | Minimum live repaint interval |
| `minLiveWindowMs` | `200` | Minimum window before showing a live rate |
| `idleClearMs` | `0` | Auto-clear status after idle (`0` = off) |
| `showElapsed` | `true` | Append `took X` at `agent_settled` |
| `clearOnTurnEnd` | `true` | Clear footer while idle between turns |
| `showGenerationTps` | `false` | Show the generation-throughput segment |
| `metricsUrls` | `{}` | Provider id → Prometheus `/metrics` URL |
| `metricsTimeoutMs` | `500` | Metrics fetch timeout |
| `statusModes` | `["tui","rpc"]` | Modes where the status is shown |

Calibration is persisted to `~/.pi/agent/live-throughput-state.json`.

### ANEMLL Forge servers: server-exact readout (22 Nov 2026)

ANEMLL Forge servers (`ane-qwen38`, see [PI_CODING.md](../../../docs/PI_CODING.md), and the V8-96K hybrid server)
have no Prometheus `/metrics`, but their `GET /health` reports the running decode (`decode_live`: tokens and
seconds on the server clock) and, after each request, its decode and prefill stats. Map the provider in
`healthUrls` and the footer uses them:

```json
{
  "healthUrls": { "ane-qwen38": "http://127.0.0.1:8765/health" },
  "healthPollMs": 1000
}
```

The live rate is then the server's own (no `~`), and the final line shows the server's decode tok/s, tokens per
verify call and prefill tok/s (prompt tokens actually computed over prefill seconds). The final stats are used only
when the server's decode token count equals `usage.output`; otherwise the client-side numbers are shown.
Speculative and tree decoding deliver tokens in bursts per verify cycle, so this is the only exact readout for these
servers. `examples/pi/install_pi_v8_96k.py` writes this entry.

| Field | Default | Description |
|---|---|---|
| `healthUrls` | `{}` | Provider id -> ANEMLL Forge `GET /health` URL (server-exact mode) |
| `healthPollMs` | `1000` | `/health` poll interval while streaming |

### Live rate display

`/tps` switches the live rate between a sliding window (default, `20.1 tok/s (5s)`), the running average since the
first chunk (`16.4 tok/s avg`) and both; `/tps window 10`, `/tps both`, `/tps status`. The choice is kept in
`live-throughput-state.json`; `PI_TPS_MODE=window|avg|both` and `PI_TPS_WINDOW_S=5` override it at launch.

## Provider matrix

| Provider kind | Live decode | Final decode | Prefill |
|---|---|---|---|
| Standard OpenAI-completions | estimated | exact | only if `metricsUrls` |
| Anthropic messages | estimated | exact | — |
| OpenAI Responses | estimated | exact | — |
| Gemini / cumulative-usage OAI | exact | exact | only if `metricsUrls` |
| vLLM (with `/metrics`) | estimated/exact | exact | **server-side** |

For vLLM, the required series are:

```text
vllm:prompt_tokens_by_source_total{source="local_compute"}
vllm:prompt_tokens_by_source_total{source="local_cache_hit"}   # ignored
vllm:time_to_first_token_seconds_sum
vllm:time_to_first_token_seconds_count
```

## Limitations

- Client-observed decode TPS measures the stream window, not model decode
  time. Gateways that batch SSE chunks can still distort the live value; the
  first-chunk exclusion mitigates it. True decode requires server timing.
- Server-side **decode** TPS (vLLM `vllm:inter_token_latency_seconds_*`) is not
  wired in; only prefill is.
- Compact mode is not included in this revision.
- `metricsUrls` adds one short GET before each request (up to
  `metricsTimeoutMs`); keep the endpoint on the LAN.

## Changelog

### v2 (this revision)
- Rate-first footer; removed the `Decode:` label and the misleading
  `Input/TTFT` rate.
- Live **and** final decode numerators exclude the first chunk (fencepost fix).
- Adaptive per-model, per-content `chars/token` calibration via EWMA, persisted
  across `/reload`.
- Exact `usage.output` path for providers that stream cumulative usage.
- Server-side **prefill** TPS from Prometheus metrics (prefix-cache aware).
- Optional **generation** throughput segment (`showGenerationTps`).
- Whole-request wall clock (`took X`) with s/m/h/d scaling.
- Lifecycle cleanup on `session_shutdown` and `turn_end` (`clearOnTurnEnd`).
- Robust abort/error/zero-output handling; approximate TTFT flag.
- Large-number formatting (`12.3k`, `1.2M`); configurable `statusModes`.

### v1 (`74d80a9`)
- Original: `chars/4` live estimate, `output-1` final numerator, `Input/TTFT`
  prompt-rate estimate, fixed `Decode:` footer.
