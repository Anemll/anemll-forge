# Splash 1.6.1 against forge, whole server, whole machine (18 Nov 2026, evening)

Harness: `scripts/m6_compare_bench.py`, as in the 4 October run:
- One cold prompt per context, filling the entry to capacity minus 512, with a nonce. Its time to the first token is
  the prefill.
- Then three identical greedy requests: thinking off, 256 tokens; decode is their median.
- Client-side stream timing.
- Power from mactop `total_power` (whole machine, SMC) once a second, with a 30 s idle baseline per server.
- Each server alone on the machine.

**Splash** (`splash161_*`):
- **Build:** the checkout `incoai_splash` at d8af1054 ("App 1.6.1"), built with `make -j4` (33 s). Its Python
  environment is `make install-environment`, on the Python 3.13 from the 1.3.2 bundle, since none was on PATH.
- **Launch:** `pipe/splash161/serve.sh`, defaults plus `--offline`.
  - Model: `unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS` at revision 4ca72078, the only one cached, so the Oct 4 model.
  - Draft: the cached DFlash2 snapshot 015e7956, given as a local directory. Offline mode cannot resolve the draft
    repository's branch (first attempt, `nov18_cmp_chain.try1.log`).
  - Vision on (default).
  - Its own calibration: "Neural Engine FFN split at share 0.74 for chunks of 512 rows or more: 31.7 ms per 2048-row
    FFN layer against 63.2 on the GPU alone".

**Forge** (`forge_f8x_ka.*`): the served V8 build on 8765 as it now runs.
- FP8 x prefill attention (`libgpu_v8_f8.dylib` 50198bdf4f8b6b8b).
- Decode split verify with the GPU clock keep-alive (`libdec_gpu_clk.dylib` 2b64e979a91a414f, 60 ms hold).
- Drafter all3.

Both banners are in the logs.

| context | prompt tokens | prefill tok/s, Splash / forge | W | J per prompt token (net of idle) | decode tok/s, Splash / forge | W | J per generated token (net) |
|---|---|---|---|---|---|---|---|
| 8K | 7,673 | 485 / 688 (**+42%**) | 44.3 / 47.1 | 0.091 / 0.068 (0.084 / 0.064) | 56.3 / 60.6 (+8%) | 42.1 / 23.5 | 0.751 / 0.388 (0.686 / 0.334) |
| 16K | 15,865 | 462 / 643 (**+39%**) | 47.0 / 48.5 | 0.102 / 0.075 (0.094 / 0.070) | 55.1 / 56.8 (+3%) | 43.4 / 23.9 | 0.789 / 0.421 (0.722 / 0.363) |
| 32K | 32,249 | 419 / 566 (**+35%**) | 44.7 / 47.5 | 0.107 / 0.084 (0.098 / 0.078) | 51.1 / 57.5 (+13%) | 42.9 / 23.9 | 0.839 / 0.419 (0.768 / 0.362) |
| 64K | 64,953 | 365 / 461 (**+26%**) | 45.6 / 46.4 | 0.125 / 0.101 (0.115 / 0.094) | 49.1 / 49.8 (+1%) | 42.1 / 23.7 | 0.858 / 0.477 (0.784 / 0.411) |

- **Idle:** Splash 3.65 W, forge 3.27 W. The October records idled at 8.4-9.1 W, so absolute watts are not comparable
  with October; within this table both runs share today's state.
- **Prefill:** forge is 26-42% faster at about the same whole-machine power (46-48 W against 44-47 W). Per prompt
  token it uses 19-26% less energy gross and 18-24% less net of idle.
- **Decode:** forge is +1 to +13% faster at 23.5-23.9 W against 42-43 W, about 45-50% less energy per generated
  token. Decode speed is one fixture with very high draft acceptance for both drafters, so the decode rates are
  close to each other by construction.
- **Determinism:** the three replies at each context were identical for both servers.
- **Splash 1.6.1 against 1.2.0 (Oct 4):** prefill 485 / 462 / 419 / 365 against 311 / 286 / 272 / 231 tok/s
  (+56-58%, the ANE FFN split); decode about the same (54.2 / 55.7 / 52.0 / 48.7).
- **Not matched:**
  - The models differ: GGUF UD-IQ3_XXS against forge's VQ 3-bit build.
  - Splash loads vision by default; forge is text only.
  - Splash keeps INT8 K and V; forge's GPU prefill uses INT8 K with an e4m3 V copy, and the ANE cache is INT8 K and V.
