# Brief: profile the latest forge on this M6 (23 Nov 2026)

"Latest forge" = the served V8.2 tree8 + sliced configuration with the KV-wrap leak fix (_uw dylibs):
build v8_dec8ts (8 packages x 81 entries), tree8 for greedy and sampled requests, sliced entries 24/40/48/56K,
F8=x, QKE=tg, INT8 KV, drafter all3 (DFlash2), ctx 98304, port 8765, model id qwen38-27b-ane.

## Paths

| what | path |
|---|---|
| repo (branch v8-gpu-ane-prefill, HEAD 7e9627c) | /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8 |
| research dir (V) | $REPO/research/v8_prefill |
| served launcher | /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/v8_server_dec8_qk.sh (same file as $V/pipe/v8_server_dec8_qk.sh) |
| leak-fix dylibs | /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/engine/libgpu_v8_f8q_uw.dylib, libdec_gpu_tree_uw.dylib |
| build | /Volumes/SSD4TB/anemll-forge-research/v8_dec8ts |
| GPU tables | /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_dec8ts/manifest_best.json |
| python (name must stay `python`) | /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python |
| server log | /Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log (rotated per launch) |
| memory watchdog log | /Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/watchdog_v8_server.log |
| bench | $V/bench_tree/tree_bench.py (prompts smoke_2k_v1.json, realistic_v1.json, long_agent_v1.json, u1_code_agent.json) |
| drafter timing launch (example of a full profile window) | $V/serve_t/draft_prof.sh |
| leak / footprint A/B | $V/serve_t/memleak_ab.sh, memleak_alt.py |
| power sampler | /Users/anemll/homebrew/bin/mactop (tree_bench samples it unless --no-power) |
| reviewer lock | $V/results/22nov_review/windows/runner.lock |
| results go to | $V/results/23nov/ (one folder per run + README index) |

## Before you start (all must hold)

1. No reviewer window: the runner.lock above is not held and no `window.zsh` process is running.
2. Pi idle: last request in v8_server.log is 10+ min old and no connection to 8765 for 60 s
   (`lsof -nP -iTCP:8765`).
3. Nothing else on the Neural Engine (one model at a time). Do not touch other ports or services.
4. Internal disk free >= 15 GiB (`df -h /`); stop and ping below that.
5. Note what is on 8765 now (pid, launch env via `ps eww`) so you can put it back exactly.

## Launch the latest forge

```
cd /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
V8_SRV_DYLIB=libgpu_v8_f8q_uw.dylib DEC_DYLIB=libdec_gpu_tree_uw.dylib \
  nohup zsh v8_server_dec8_qk.sh > /Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server_tree.launch.out 2>&1 &
```

It arms forge_watchdog (kills above 12288 MB footprint). Add profiling env in front (e.g. `DRAFT_PROF=1`) only;
keep every other variable as the launcher sets it. Wait for `listen 127.0.0.1:8765`, then check the banner in
v8_server.log, all of these must appear, otherwise stop and report:

- `v8: GPU engine libgpu_v8_f8q_uw.dylib`
- `dec8: BANNER decode GPU dylib .../libdec_gpu_tree_uw.dylib`
- `tree: mode=tree8 ... sampled=on` and the slice line for 24/40/48/56
- `speculative` ON with drafter all3 (a launch at 08:41 today showed `speculative OFF (no drafter)`, 38 entries per
  chunk and 11 tok/s decode: that is NOT the latest forge)
- `model dir /Volumes/SSD4TB/anemll-forge-research/v8_dec8ts`, 81 entries per chunk

## Smoke (before and after profiling)

```
cd $V/bench_tree
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python tree_bench.py \
  --prompts smoke_2k_v1.json --label smoke --out $V/results/23nov/<run>/smoke.json --no-power
```

Output checksum must be `29aef2ec9db18a3f`.

## Profiling runs (pick what you need)

- Prefill / decode by context: `tree_bench.py --prompts realistic_v1.json --rungs <list> --label <l> --out ...`
  (power via mactop; `--temp 1.0 --think 1 --seed 7100` for Pi-like sampled requests).
- Long agent session: `--prompts long_agent_v1.json` (greedy, then sampled seed 7100).
- Drafter / cycle split: relaunch with `DRAFT_PROF=1`; the server logs `drafter prof: body / head / ctx / propose
  host` per request (see serve_t/draft_prof.sh for the full window).
- Footprint over time: `footprint <pid>` / the watchdog log; leak check = alternate 2K and 56K requests
  (serve_t/memleak_alt.py); expect flat after the first 96K visit (one 96K KV set stays held until the next 96K
  visit, known).
- GPU/ANE traces: Instruments / IOReport only with forge as the sole Neural Engine user; never run standalone ANE+Metal
  probes while forge is up (they have hung the ANE before).

## Rules

- Compile only via forge.py (nothing should need a compile; all 17 packages are cached for macOS 26C5047e).
- Do not edit served launchers, defaults or dylibs; profiling env goes on the command line only.
- Do not delete caches. Do not reboot. No sudo.
- If Pi sends a request during your window (max_tokens 32768), abort at once and restore.
- Never invent numbers: raw output to $V/results/23nov/<run>/, plus a README with the command lines.
- **When done:** stop your server (`kill <pid>`, wait until 8765 is free), relaunch exactly what was there before
  (by default the launch above without the profiling env), check the banner, run the smoke (29aef2ec9db18a3f), report.
