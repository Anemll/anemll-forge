# Jeff decision server

`forge.py jeff-serve` loads one compiled Jeff Core AI package and answers decision requests. It speaks the same route as upstream `jeff-serve`: `POST /v1/systemone`. The process also serves a browser demo at `/`.

This is not the 27B chat server. `forge.py serve` still requires 64 layers and hidden size 5120. Jeff commands (`jeff-convert`, `jeff-smoke`, `jeff-serve`) are the only ones that bypass that gate.

## One command

Weights stay on the machine. The Core AI interpreter loads the packages. The interpreter that launches `forge.py` must have `transformers`, because that is what tokenizes Jeff's chat template (`TOKENIZER_PYTHON`).

```sh
export COREAI_PYTHON=/Users/anemll/SourceRelease/GITHUB/ML_playground/CoreAI-experiments/experiments/coreai_gemm_bench/.venv/bin/python

python forge.py jeff-serve \
  --model /Users/anemll/Models/jeff/jeff-base-v1.3 \
  --build /Users/anemll/Models/jeff-coreai/coreai \
  --host 127.0.0.1 --port 8787
```

Then open http://127.0.0.1:8787/.

| Page | What it does |
| --- | --- |
| `/` | Snake: every move is one Jeff decision, with per-option probabilities and tokenize / prefill / head / total time. Below it, a routing panel classifies a message into refunds, shipping, technical, or other. |
| `GET /health` | `ready`, model name, checkpoint, `max_options`, modalities `["text"]`, backend, whether a prefix cache is installed |
| `GET /v1/models` | `jeff`, `jeff-latest`, and the checkpoint name (`jeff-qwen3.5-0.8b` for this base) |
| `POST /v1/systemone` | One decision. `POST /v1/decide` is the same handler |

The package is loaded once. One request runs at a time. `JEFF_QUEUE_MS` (default `0`) is how long a second request waits before `529` and `Retry-After: 1`. `JEFF_API_KEY`, when set, requires `Authorization: Bearer …` on `/v1/systemone`, `/v1/decide`, and `/v1/models`. `/health` and the demo page stay open. `JEFF_MAX_TOKENS` can lower the cap; it cannot exceed the build's KV rows (2,048 on the `p256_2k` package). Upstream Jeff's default of 8,192 does not fit this build.

The server is text only. A request with `images` is `422`.

## Request and response

The body matches upstream Jeff (`jeff.client`, `docs/v1.3-request-format.md` in the Jeff repo):

```json
{
  "model": "jeff-latest",
  "state": {"rules": "stable text", "latest": "the part that changes"},
  "questions": {
    "move": {
      "type": "choice",
      "instructions": "Pick the snake's next move.",
      "criteria": {"up": "one cell up", "down": "one cell down", "left": "one cell left", "right": "one cell right"}
    }
  }
}
```

`state` may be text, a JSON object, or a list. `questions` may also be `noul` or `score`. `orders: 2` answers each question again with its options reversed and averages the two distributions.

A short form is accepted when `questions` is omitted:

```json
{"state": "The disk is full.", "options": ["page", "wait", "ignore"], "instructions": "What next?"}
```

A list of strings becomes option keys `o1`, `o2`, … (the same keys `jeff.client.choose` uses for a list). An object is the criteria map, key to description (`null` means the key alone).

The response is Jeff's decision object plus timings:

```json
{
  "model": "jeff-qwen3.5-0.8b",
  "answers": {
    "move": {"type": "choice", "choice": "up", "probabilities": {"up": 0.4, "down": 0.2, "left": 0.2, "right": 0.2}, "confidence": 0.2}
  },
  "usage": {"input_tokens": 180, "output_tokens": 0, "orders": 1},
  "timings": {
    "tokenize_ms": 1.2,
    "calls_ms": [64.0],
    "prefill_ms": 64.0,
    "head_ms": 0.3,
    "total_ms": 66.0,
    "questions": [{"id": "move", "tokenize_ms": 1.2, "calls_ms": [64.0], "prefill_ms": 64.0, "head_ms": 0.3, "tokens": 180, "prefix_tokens": 0}]
  }
}
```

`calls_ms` is one number per 256-row prefill call. `prefill_ms` is their sum. `head_ms` is the readout. `total_ms` is the whole request inside the model lock (tokenize, prefill, head, and the small Python around them). `prefix_tokens` is how many leading tokens were resumed from a cache; it is 0 until a cache is installed.

Aliases `jeff` and `jeff-latest` (and the legacy name `jeff-qwen3.8-27b`) all select this checkpoint. There is no adapter loading.

An existing Jeff client points at this server with no other change, for text requests:

```python
from jeff import Client
jeff = Client("http://127.0.0.1:8787", model="jeff-latest")
jeff.choose("The disk is full.", {"page": "Page someone.", "wait": "Wait."})
```

## Prefix cache hook

Every request currently starts from an empty Gated DeltaNet state and position 0. A later live-last cache plugs in at two places:

1. **Runtime.** `JeffCoreAI.capture_state()` copies position, token ids, the last hidden row, and each chunk's GDN/conv state and KV. `JeffCoreAI.prefill(token_ids, prefix=snapshot)` and `decide(..., prefix=snapshot)` restore that snapshot and prefill only the suffix. The shape check lives in `coreai/jeff_prefix.py` (`resume_at`). A snapshot that already covers the prompt runs the readout on the cached hidden row and does not call the backbone.
2. **Server.** Set `app.prefix_cache` to an object with `lookup(token_ids) -> snapshot | None` and `store(token_ids, snapshot)`. After each question the server passes `lookup`'s snapshot into `decide` and then stores `capture_state()`. With `prefix_cache is None` (the default) the server does not copy KV.

The demo's snake state is an object whose last field is `latest` (the board). Question, options, and `rules` stay in front of that, which is the split a live-last cache would reuse. The cache policy itself is not in this branch.

## What stays on the ANE

`jeff-serve` loads the same `p256_2k` packages as `jeff-smoke`. Placement is a property of the cached specialization, checked with:

```sh
python coreai/inspect_coreai_cache.py \
  --model-dir /Users/anemll/Models/jeff-coreai/coreai \
  --executable python --strict
```

`--executable python` matches the Core AI SDK venv (the cache key is the interpreter name). The head and every chunk entry should be `fully_ane`.

## Tests

Request and response tests use a fake engine, so they do not load Core AI:

```sh
python -m unittest tests.test_jeff_serve tests.test_jeff_coreai tests.test_launcher tests.test_checkpoint
```

`tests.test_jeff_serve.UpstreamClientTests` runs the real `jeff.client` against that fake engine when `JEFF_SRC` (default `/Users/anemll/Models/jeff/jeff-src/src`) is present.

## Measured latency

Filled in after a local run on the M5 Max against the demo's snake prompt and a routing prompt. See the pull request for the screen capture.
