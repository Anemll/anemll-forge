# Forge KV-wrap leak, 22 Nov 2026

The served forge was killed by forge_watchdog.sh at 16:30:16 (footprint 14336 MB > 12288 MB): 4.0 GB at launch
(13:49), +1 to 2 GB on each context shrink then grow. Last step in the log: a cold 56K context-summary request, `[ctx]
98304 -> 8192`, then `8192 -> 65536` and "64 ANE KV segments registered", footprint 12288 -> 14336 MB in 2 s.

Cause: both GPU engines wrap the ANE's KV segment buffers zero-copy (makeBuffer bytesNoCopy) and keep every wrap
in a global list that never drops anything (gpu_v8_f8q.swift v8_wrap, dec_gpu_tree.swift dec_wrap). After a reset
frees a context's KV segments, the old Metal buffers keep their pages resident: footprint(1) shows them as "Owned
physical footprint (unmapped)". The prefill engine's per-layer segment list also kept stale entries past the new
context.

Fix: v8_unwrap / v8_kv8_clear (libgpu_v8_f8q_uw.dylib) and dec_unwrap (libdec_gpu_tree_uw.dylib). v8_engine.py
`_register_segments` clears each attention layer's segments, registers the current ones, then drops the previous
wraps. dec_split.py `_sync_kv` drops the previous wraps after re-registering. With the served dylibs (no new symbols)
the Python code behaves as before. Served launcher defaults are unchanged.

Test (memleak_ab.sh + memleak_alt.py, fresh launch per arm, served config v8_dec8ts, greedy, thinking off, cold
requests via unique prompt nonces, max_tokens 48; footprint 3 s after each request; server stopped after each arm):

| run | arm | start | after each request (MB) | unmapped after last |
|---|---|---|---|---|
| r1 2K / 56K x3 | old (served dylibs) | 4546 | 4633 6278 6299 8156 8162 9983 9986 | 5418 |
| r1 2K / 56K x3 | new (_uw dylibs)    | 4546 | 4633 6160 4367 6198 4396 6215 4412 | 0 |
| r2 2K / 70K x2 | old | 4546 | 4634 7259 7272 10240 10240 | 2838 |
| r2 2K / 70K x2 | new | 4546 | 4633 7259 7274 7470 7474 | 49 |

r1 old: +1806 MB unmapped per 56K cycle (one 64K KV set). r2 exercises the 96K entry and the decode split engine
(15 split verify steps per 70K request in both arms).

Smoke: 2K set (tasks 1, 8) output checksum 29aef2ec9db18a3f in all four launches. Parity: every reply text identical
old vs new (r1 7/7, r2 5/5), including the 64K and 96K entry decodes.

Residual (not a leak, bounded): after a 96K request, a later small request keeps the 96K KV set resident (r2 new:
7274 MB at 8K instead of about 4.4 GB) until the next 96K visit replaces it: dec_split `_sync_kv` (its `_kvobjs` and
the decode engine's wraps) only runs on the 96K entry. At most one 96K set (about 2.8 GB) is held.
