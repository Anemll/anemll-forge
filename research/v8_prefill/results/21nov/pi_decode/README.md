# Pi decode speed: today (tree8 server) vs 19 Nov (served chain)

`serve_t/pi_decode_compare.py` over every server log with sampled (Pi) requests; numbers only, no prompt text.
Token-weighted tok/s and tok/verify, cycle-weighted ms/cycle, banded by the prompt length at decode.

| context | 19 Nov 12:42 chain: tok/s, tok/verify, ms/cycle | 21 Nov 07:31-10:10 tree8 server: tok/s, tok/verify, ms/cycle |
|---|---|---|
| <8K | 51.1, 5.74, 111.9 (3 req) | 48.3, 5.42, 112.0 (5 req) |
| 8-16K | 25.1, 2.83, 112.5 (6 req) | 34.4, 3.91, 113.5 (9 req) |
| 16-32K | 23.4, 2.81, 120.5 (14 req) | 31.6, 3.80, 120.3 (22 req) |
| 32-64K | none | 27.1, 3.68, 135.9 (12 req) |
| all | 27.6, 3.22, 116.7 (23 req, 19310 tok) | 33.5, 4.10, 122.4 (48 req, 53675 tok) |

- Tree8 never runs on Pi requests: `tree_verify.py:281` sends every sampled cycle (temperature > 0; Pi uses 1.0)
  to the served chain. Today's log has tree summary lines only for 3 greedy test requests.
- The cost per cycle is the same within 1% at matched context (draft 13.0-13.1 ms, verify 96-106 ms): the tree
  build's chain path costs what the 19 Nov chain did.
- The 8-32K gain (+35-37%) is all tok/verify: today's text was easier for the drafter. Content, not the server.
- 20 Nov had 3 Pi requests (1261 tok: 21.8 tok/s, 2.38 tok/verify, 109.0 ms/cycle, <8K), too few to compare.

Files: `pi_decode.log` (every log, by band), `pi_decode_today_vs_19nov.log`, `pi_decode.json` (per request).
