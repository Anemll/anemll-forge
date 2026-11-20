# GPU clock for short bursts: the decode split's attention calls (18 Nov 2026)

M6 (Mac18,5, 12-core GPU, 502-1735 MHz), macOS 27.3 (26C5035r). The question: the split verify alternates ~5 ms ANE
calls with 16 short GPU attention calls, and the GPU runs them at a low clock (the server's telemetry: 874 MHz
active-weighted, power controller `PWRCTRL` in `DEADLINE`). Can the GPU run short bursts at full clock with no ramp,
and at what cost?

**Answer: yes. Keep the GPU continuously busy with one simdgroup on a second command queue, as short serial
dispatches (20 us each, 250 per command buffer), while the decode runs.**
- **GPU-only:** each gapped attention call then runs at the steady-state kernel time. At 64K it drops from 3.21 to
  1.06 ms GPU and from 3.8 to 1.17 ms wall; at 95K from 4.47 to 1.54 ms GPU and from 5.0 to 1.64 ms wall.
- **Side effects:** no slowdown of the attention kernel, and identical output.
- **Cost:** about 0.8 W GPU and 0.6-1 W CPU while held. The attention itself costs about 2.4x more energy per call
  at 1735 MHz than at 502 MHz.
- **What does not work:** pre-warms, periodic micro-dispatches, low-duty bursts, event-gated pre-commit, QoS,
  latencyCritical, and the private deadline / priority calls. The controller needs 10-20 ms of near-continuous busy
  time before it raises the clock.

End to end on the server, with an external keep-alive process and 4 on / off pairs at 66-70K tokens:
- **Split verify step:** 138.2 to 122.4 ms (-15.8 ms, -11%).
  - GPU attention: 30.8 to 20.8 ms per step.
  - ANE calls and handoffs: 107.4 to 101.5 ms. The ANE got no slower.
- **Decode cycle:** 152.5 to 137 ms.
- **Power:** +2.1 W combined.
- **~92K tokens (one usable on-turn):** split verify step ~143-149 to 133.9 ms.

## Method

- `gpu_clk.swift` (`pipe/unit/power/`): the served decode kernel (`engine/dec_kernels.metal`: `dec_prep`,
  `dec_attn_r4`, `dec_combine_gate`), encoded exactly as `engine/dec_gpu.swift` does it.
  - **Data:** one synthetic INT8 layer, segments 8K+8K+16K+32K+64K, at 64K (65,536) or 95K (95,232) keys.
  - **Regimes:** one process runs every regime; each regime is a set of options (see the file header).
  - **Per call:** GPU ms (`gpuEnd - gpuStart`), wall ms (commit to `waitUntilCompleted` return, which is what the
    server waits for), and start latency.
  - **Gaps:** `mach_wait_until` CPU sleeps stand in for the ANE calls.
- `gpu_clk_run.py`: runs it with IOReport (`iorep_sample`, 100 ms) and powermetrics (`pm_service.sh`, 100 ms), then
  summarizes per regime.
  - **Rounds:** every regime runs in 4-5 interleaved rounds (the order rotates each round), 2.5 s measured per phase
    after 0.4 s discarded, with 1.5-2 s idle between phases.
  - **Reported:** the median of the per-round medians, with the min-max of those per-round medians in brackets.
- **Power:**
  - GPU W = IOReport `Energy Model / GPU Energy`.
  - CPU W and "combined" (CPU + GPU + ANE) come from powermetrics.
  - DRAM power is not visible to either on this machine. The battery's `PowerTelemetryData` (`SystemLoad`) updates
    every ~10 s, too coarse for these phases.
- **Isolation:** the forge server (port 8765) stayed up and idle during the GPU-only runs; no ANE model was loaded.

## Results, GPU-only (64K keys unless noted; 5 ms CPU gap between calls unless noted)

### Reference and baseline

| regime | rounds | GPU ms / call | wall ms / call | GPU MHz | PWRCTRL (active time) | GPU W | CPU W | combined W |
|---|---|---|---|---|---|---|---|---|
| steady (50 copies per command buffer) | 4 | 1.057 [1.054-1.059] | - | 1735 | PERF 84 DEADLINE 13 | 18.6 | 2.1 | 20.7 |
| back to back, 1 per command buffer | 5 | 1.058 [1.057-1.063] | 1.235 [1.232-1.237] | 1735 | DEADLINE 67 PERF 33 | 16.1 | 1.4 | 13.7 |
| gap 5 ms, nothing else (r2) | 5 | 3.209 [3.180-3.242] | 3.776 [3.735-3.825] | 502 | PRFBOOST 80 PERF 20 | 0.84 | 0.53 | 1.17 |
| gap 5 ms (r3) | 4 | 3.205 [3.195-3.212] | 3.770 [3.766-3.787] | 502 | PRFBOOST 75 PERF 25 | 0.84 | 0.56 | 1.39 |
| gap 1.5 ms | 5 | 3.200 [3.160-3.237] | 3.748 [3.665-3.813] | 502 | PRFBOOST 60 DEADLINE 40 | 1.5 | 0.57 | 1.71 |
| server pattern: gaps 5 / 9 ms + 14 ms every 16th call | 5 | 3.208 [3.197-3.248] | 3.799 [3.717-3.845] | 502 | PRFBOOST 80 DEADLINE 20 | 0.63 | 0.50 | 0.97 |
| 95K, gap 5 ms | 5 | 4.473 [4.426-4.514] | 5.045 [4.925-5.115] | 502 | PRFBOOST 60 PERF 40 | 1.08 | 0.56 | 1.38 |
| 95K, back to back | 5 | 1.532 [1.531-1.538] | 1.708 [1.707-1.711] | 1735 | DEADLINE 84 PERF 16 | 16.8 | 1.3 | 14.1 |

- **Clock:** with any gap the GPU stays at P1 (502 MHz) for the whole run. The kernel then takes 3x longer, and about
  0.55 ms more per call goes to start-up and wake-up (wall minus GPU).
- **DRAM clock state:** in r1 one round ran all the gapped regimes at 2.58 ms instead of 3.17, still at 502 MHz. In
  that round the DRAM clock state `DCS_F10` was at 46% instead of 0%, and GPU memory latency was 63 cycles instead of
  ~210. So at low clock the kernel also depends on the DRAM clock.

### Keep-alive: one simdgroup spinning on a second command queue (same process)

`ka=cont:US:1`: one dispatch of US us per command buffer. `ka=contm:N:US:1`: N serial dispatches of US us per command
buffer. Both keep two command buffers in flight.

| regime | rounds | GPU ms / call | wall ms / call | GPU MHz | PWRCTRL | GPU W | CPU W | combined W |
|---|---|---|---|---|---|---|---|---|
| cont 200 us | 5 | 1.193 [1.192-1.196] | 1.275 [1.274-1.278] | 1735 | DEADLINE 100 | 3.36 | 1.99 | 4.50 |
| cont 1000 us | 5 | 1.250 [1.243-1.264] | 1.345 [1.337-1.355] | 1735 | DEADLINE 96 | 3.35 | 1.41 | 3.96 |
| cont 50 us (r1) | 4 | 1.099 [1.099-1.105] | 1.168 [1.165-1.196] | 1735 | DEADLINE 98 | 3.45 | 4.71 | 8.20 |
| contm 10 x 100 us | 5 | 1.157 [1.155-1.162] | 1.255 [1.253-1.259] | 1735 | DEADLINE 98 | 3.36 | 1.43 | 4.00 |
| contm 25 x 200 us | 5 | 1.198 [1.195-1.205] | 1.304 [1.296-1.313] | 1735 | DEADLINE 81 PERF 19 | 3.36 | 1.26 | 3.85 |
| contm 50 x 20 us | 4 | 1.065 [1.064-1.065] | 1.163 [1.162-1.165] | 1735 | DEADLINE 99 | 3.45 | 1.65 | 5.09 |
| contm 200 x 5 us | 4 | 1.065 [1.064-1.065] | 1.164 [1.163-1.165] | 1735 | DEADLINE 96 | 3.45 | 1.70 | 5.16 |
| **contm 250 x 20 us** | 4 | **1.058 [1.058-1.061]** | **1.170 [1.165-1.173]** | 1735 | DEADLINE 78 PERF 21 | 3.34 | 1.37 | 4.71 |
| contm 1000 x 20 us | 4 | 1.060 [1.059-1.061] | 1.177 [1.174-1.178] | 1735 | DEADLINE 53 PERF 46 | 3.35 | 1.17 | 4.51 |
| gap 1.5 ms, contm 25 x 200 us | 5 | 1.195 [1.194-1.201] | 1.298 [1.296-1.307] | 1735 | DEADLINE 85 | 6.86 | 1.25 | 6.44 |
| server pattern, contm 25 x 200 us | 5 | 1.200 [1.195-1.211] | 1.306 [1.300-1.315] | 1735 | DEADLINE 72 PERF 27 | 2.58 | 1.24 | 3.21 |
| 95K, contm 25 x 200 us | 5 | 1.738 [1.733-1.748] | 1.843 [1.839-1.855] | 1735 | DEADLINE 83 | 4.24 | 1.23 | 4.46 |
| 95K, contm 50 x 20 us | 4 | 1.537 [1.535-1.537] | 1.634 [1.632-1.635] | 1735 | DEADLINE 99 | 4.31 | 1.59 | 5.91 |
| **95K, contm 250 x 20 us** | 4 | **1.537 [1.535-1.538]** | **1.643 [1.642-1.644]** | 1735 | DEADLINE 71 PERF 28 | 4.30 | 1.29 | 5.57 |
| 95K, server pattern, contm 25 x 200 us | 5 | 1.736 [1.733-1.738] | 1.841 [1.837-1.843] | 1735 | DEADLINE 82 | 3.26 | 1.23 | 3.72 |

**Effect on concurrent GPU work.** The keep-alive running beside back-to-back attention calls, all at full clock:

| keep-alive | GPU ms / call, back to back |
|---|---|
| none | 1.057 [1.057-1.058] |
| contm 25 x 200 us | 1.190 [1.188-1.200] (+12.5%) |
| contm 10 x 100 us | 1.162 [1.162-1.163] (+10%) |
| contm 50 x 20 us | 1.058 [1.058-1.059] (0%) |
| contm 200 x 5 us | 1.058 [1.058-1.060] (0%) |
| 95K, contm 50 x 20 us | 1.530 [1.528-1.531] vs 1.532 alone (0%) |

- **Dispatch length:** a keep-alive dispatch of 100 us or longer slows the attention by 9-18%. It occupies a slot that
  the attention's last wave waits for. Dispatches of 20 us or less cost nothing measurable.
- **Wall time:** with the GPU awake, commit to return drops from ~0.55 ms over the GPU time to ~0.1 ms. That is why
  the gapped wall (1.16 ms) is below plain back to back (1.235 ms).

**Keep-alive cost alone** (no attention calls; idle CPU 0.5 W, GPU 0.0 W):

| keep-alive | command buffers / s | GPU W | CPU W | combined W |
|---|---|---|---|---|
| contm 50 x 20 us | 1000 | 0.79-0.85 | 1.62-1.64 | 2.40-2.50 |
| contm 25 x 200 us | 200 | 0.82 | 1.14 | 1.77 |
| contm 250 x 20 us | 200 | 0.79 | 1.39 | 2.18 |
| contm 1000 x 20 us | 50 | 0.79 | 2.51 (encode cost of 1000 dispatches; 1.17 in the gapped phase) | 3.29 |
| cont 200 us (r1) | 5000 | 0.81 | 3.50 | 4.34 |

- **GPU:** holding P13 with one simdgroup costs ~0.8 W.
- **CPU:** the cost comes from command buffer submissions and dispatch encoding, about +0.6-1 W at 200 command
  buffers per second (powermetrics CPU W varies by about ±0.3 W between phases).
- **QoS:** the keep-alive thread's QoS (user-interactive, utility, background) made no difference (1.63 / 1.63 / 1.64
  W).
- **DRAM:** the DRAM clock state goes to `DCS_F10` 100% (from 0% in the gapped baseline). Its power is not measured
  here.

**Energy per attention call**, counting the keep-alive's share of the gap:

| | GPU mJ / call | combined mJ / call |
|---|---|---|
| 64K, 502 MHz, no keep-alive | 8.3 | 11.6-13.8 |
| 64K, 1735 MHz, back to back | 19.9 | 17-22 |
| 64K, keep-alive (contm 250 x 20 us) | 24.6 | 34.7 |
| 95K, no keep-alive | 12.0 | 15.3 |
| 95K, keep-alive | 33.5 | 43.4 |

The clock costs voltage: a call at 1735 MHz takes ~2.4x the GPU energy of a call at 502 MHz.

### What does not work (r1, 4 rounds unless noted)

| regime | GPU ms / call | wall ms | GPU MHz | PWRCTRL | note |
|---|---|---|---|---|---|
| tiny dispatch every 1 ms (1 thread) | 2.909 [2.567-2.917] | 3.117 | 502 | PERF 50 PRFBOOST 50 | GPU "active" 100% but stays at P1 |
| tiny dispatch every 0.5 ms | 2.611 [2.581-2.623] | 2.789 | 502 | PERF 50 PRFBOOST 50 | |
| 100 us spin every 1 ms | 3.026 [2.691-3.050] | 3.221 | 502 | PERF 50 PRFBOOST 50 | the spin stretches at 502 MHz |
| 500 us spin every 2 ms | 2.225 [2.063-2.452] | 2.404 | 780 | PERF 99 | |
| full-GPU ALU burst 2 ms at 10% duty | 2.271 [1.981-2.287] | 2.755 | 758 | PERF 98 | p90 wall 4.15 ms |
| full-GPU burst 0.5 ms at 5% duty | 3.163 [2.571-3.179] | 3.709 | 502 | PERF 100 | |
| pre-warm: 1 simdgroup busy for the last 1 ms of each gap | 3.176 [2.740-3.241] | 3.383 | 502 | PERF 75 DEADLINE 25 | |
| pre-warm: 1 simdgroup busy for the last 2 ms | 3.056 [2.710-3.106] | 3.224 | 563 | PERF 100 | |
| pre-warm: full-GPU 300 us burst 0.5 ms ahead | 3.707 [3.555-3.720] | 3.919 | 502 | PERF 100 | worse |
| attention pre-committed, gated on an `MTLSharedEvent` | 3.149 [2.570-3.175] | 3.677 (from the signal) | 502 | PERF 100 | |
| submitting thread QoS user-interactive | 3.109 [2.580-3.171] | 3.604 | 502 | PERF 100 | |
| `ProcessInfo` activity `.latencyCritical` | 2.850 [2.572-3.158] | 3.292 | 502 | PERF 100 | within the baseline's range that run (2.581-3.184) |
| PRIVATE `-[MTLCommandBuffer commitWithDeadline:]`, now + 500 us (r3) | 3.204 [3.178-3.239] | 3.775 | 502 | PRFBOOST 75 | |
| PRIVATE `-[MTLCommandBuffer setDeadline:]`, now + 500 us (r3) | 3.197 [3.180-3.230] | 3.760 | 502 | PRFBOOST 75 | |
| PRIVATE `-[MTLCommandQueue setGPUPriority:0]` (r3) | 3.225 [3.194-3.249] | 3.779 | 502 | PRFBOOST 75 | |

Ramp (r3, 4 rounds): after 150 ms of GPU idle, the keep-alive starts L ms before one attention call.

| L | GPU ms | wall ms | GPU MHz (window) |
|---|---|---|---|
| 1 ms | 3.328 [3.028-3.350] | 3.563 | 502 |
| 3 ms | 3.210 [3.157-3.307] | 3.418 | 505 |
| 5 ms | 2.988 [2.872-3.106] | 3.201 | 588 |
| 10 ms | 1.940 [1.883-1.982] | 2.091 | 764 |
| 20 ms | 1.213 [1.208-1.227] | 1.341 | 1148 (window mostly idle) |

The controller needs 10-20 ms of continuous busy time to reach the top state. That is why a per-call pre-warm (the 16
Nov kick, and the 1-2 ms leads above) cannot fix the clock. Any duty well below 100% lets it fall back, even with the
GPU never power-gated (the tiny dispatches every 0.5-1 ms).

### Interfaces checked, not built on

- **Performance-state assertion (private):** `-[MTLDevice newPerformanceStateAssertion:error:]`,
  `currentPerformanceState`, `supportsPerformanceStateAssertion`. This is what Instruments' "Performance State"
  setting uses. Here `supportsPerformanceStateAssertion` returns NO for a plain process.
- **Deadlines (private):** `-[MTLCommandBuffer commitWithDeadline:]`, `setDeadline:`, `deadlineProfile`;
  `supportsDeadlineProfile` returns YES. With mach-absolute deadlines there was no effect (units or namespace unknown).
- **Queue priority and QoS (private):** `-[MTLCommandQueue setGPUPriority:]` / `setBackgroundGPUPriority:` (default
  priority 1, background 2): no effect. `commitsWithQoS` = NO, so the submitting thread's QoS does not reach the GPU.
- **Public API:** nothing beyond `maxCommandBufferCount` / `logState` on `MTLCommandQueueDescriptor`, and nothing on
  `MTL4CommandQueueDescriptor`. Residency sets showed no change (16 Nov).
- **OS level:**
  - `pmset` offers only `lowpowermode` (0).
  - Game Mode needs a fullscreen app in the games category, so it does not fit a server. Not tried, since no settings
    were changed.
  - Apple's developer forum (thread 768160) reports the same heuristic for real-time audio on the GPU: clocks rise only
    while a command buffer is saturating, and work alternating with idle does not raise them. DTS advises against a
    spinning keep-alive as not future-proof. Anukari ships one anyway ("waste makes haste": one warp spinning, a
    dedicated thread, short spin kernels serialized).

## Integration (`pipe/engine/dec_gpu_clk.swift` -> `libdec_gpu_clk.dylib`; served files untouched)

A copy of `dec_gpu.swift` with a keep-alive added:
- **Queue and thread:** a second `MTLCommandQueue`, and a utility-QoS thread that keeps two command buffers in flight.
  Each holds 250 serial dispatches of a one-simdgroup FMA spin of 20 us.
- **`DEC_KA_HOLD_MS`:**
  - 0 = off, the served behaviour.
  - > 0: every `dec_run` / `dec_submit` / `dec_submit_part` / `dec_arm` extends the keep-alive to that many ms after
    the call.
  - 60 covers the draft between verify steps (13.5 ms) and the head, so the keep-alive runs continuously through a
    request's decode and stops 60 ms after its last split step.
- **API:** `dec_ka_hold(ms)` starts or extends it explicitly, `dec_ka_stop()` ends it, and `dec_ka_info(out[3])`
  returns command buffers, GPU busy s and s held.
- **Other env:** `DEC_KA_US` (20), `DEC_KA_N` (250), `DEC_KA_TG` (1), `DEC_KA_QOS` (ut).
- **To try it in the server:** load `libdec_gpu_clk.dylib` instead of `libdec_gpu.dylib` in `pipe/dec_split.py` (one
  path) with `DEC_KA_HOLD_MS=60`.
  - The first verify step of a request still meets a cold GPU: the ramp takes 10-20 ms.
  - Calling `dec_ka_hold(60)` when a request's decode starts (before the first draft) hides that ramp as well.

Check `dec_clk_check.py` (r5) runs through this dylib from ctypes, the way the server calls it:
- **Pattern:** the server's gaps (5 / 9 ms, + 14 ms draft every 16 calls), keep-alive on = `dec_ka_hold(60)` before
  each `dec_run`.
- **Rounds:** 4 interleaved rounds of 4 s.

| | GPU ms / call | wall ms / call | per 16-call step, GPU ms | per step, wall ms | GPU MHz | GPU W |
|---|---|---|---|---|---|---|
| 64K off | 3.219 [3.204-3.238] | 3.950 [3.925-3.977] | 51.4 [51.0-51.6] | 63.1 [62.2-63.7] | 502 | 0.61 |
| 64K on | 1.064 [1.064-1.066] | 1.239 [1.234-1.240] | 17.1 [17.1-17.2] | 19.7 [19.7-19.7] | 1735 | 2.50 |
| 95K off | 4.427 [4.421-4.476] | 5.155 [5.137-5.249] | 70.7 [70.6-71.7] | 82.5 [82.4-84.0] | 502 | 0.82 |
| 95K on | 1.541 [1.539-1.543] | 1.708 [1.705-1.714] | 24.7 [24.7-24.8] | 27.3 [27.3-27.4] | 1735 | 3.19 |

The output checksum is identical on and off. The keep-alive held 31.9 s, with the GPU busy 31.8 s of it.

## End to end on the server

`ka_e2e.py` on the running forge server (port 8765, `v8_server_dec8.sh` build, 96K entry, split verify):
- **Conversation:** one cold request of 66,053 prompt tokens, then 8 follow-up turns in the same conversation (300
  new tokens each, 256 max tokens, greedy).
- **Keep-alive:** alternated on / off with the external `gpu_keepalive 600 20 250 1 ut`, a separate process started 1
  s before the request and stopped with SIGTERM after it. The server was not touched.
- **Measured:** the server log's decode line and dec8 line per request; IOReport and powermetrics over each decode
  window.

e2e1, 66-70K tokens. Off = turns 2, 4, 6, 8; on = turns 1, 3, 5, 7. Medians [min-max] over the 4 turns each:

| | ms / cycle | draft ms | split verify ms / step | GPU attention ms / step | rest (ANE calls + handoffs) | GPU MHz | GPU W | CPU W | ANE W | combined W | J / cycle |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 152.5 [151-155] | 13.7 [13.4-13.9] | 138.2 [136.7-140.4] | 30.8 [30.5-31.3] | 107.4 [106.2-109.1] | 857 (DEADLINE 100) | 1.3 | 1.9 | 5.0 | 8.1 | 1.24 |
| on | 137.0 [135-138] | 13.8 [13.6-14.3] | 122.4 [120.6-123.8] | 20.8 [20.1-21.0] | 101.5 [100.0-103.7] | 1735 (PERF 55-87, DEADLINE 10-43) | 3.5 | 1.8 | 4.8 | 10.2 | 1.39 |

(The cold turn 0, off: cycle 149, split 135.6, GPU 29.3 ms.)

- **Verify step:** -15.8 ms per step at 66-70K (-11.4%); the decode cycle drops by 10%.
  - The GPU attention gives -10.0 ms: 1.93 to 1.30 ms per call.
  - The other -5.9 ms is in the ANE calls and handoffs. With the GPU awake, commit to start and completion to wake
    no longer add ~0.35 ms per call.
- **No ANE penalty:** the ANE power and the draft (ANE drafter) are unchanged, so the 16 Nov wake-up penalty (a GPU
  waking mid-ANE-call made that call ~1.3 ms longer) does not appear with a GPU that never sleeps.
- **DRAM:** the clock state `DCS_F10` goes from 79-90% to 100%.
- **Cost:** +2.1 W combined (GPU +2.2 W, CPU and ANE within noise), about +12% energy per decode cycle (DRAM not
  included).
- **Remaining gap:** the server's per-call GPU time with the keep-alive (1.30 ms) is still above the GPU-only kernel
  at full clock (~1.07 ms at these keys). That gap is the kernel on real data next to the ANE (see the 17 Nov
  bandwidth notes), not the clock.

e2e2 / e2e3, 90-93K tokens. Fewer usable turns:
- **e2e2:** at 92K, the follow-ups "Now also cover these additional notes" were answered in 1-3 tokens from turn 3 on.
  Those single verify steps are not representative: the first step of a request, and the clock state carrying over
  from the previous turn.
- **e2e3:** asked for long answers, but the swap watchdog stopped it after its cold turn. During the server's 90K
  cold prefill, system swap grew 888 -> 1919 MB; it was the server's memory, not the client. No further cold runs.

| turn | keep-alive | prompt tokens | split steps | ms / cycle | split verify ms / step | GPU attention ms / step | rest | GPU MHz | combined W |
|---|---|---|---|---|---|---|---|---|---|
| e2e3 0 (cold) | off | 90,060 | 94 | 159 | 142.6 | 35.7 | 106.9 | - | - |
| e2e2 0 (cold) | off | 92,053 | 92 | 157 | 143.4 | 36.9 | 106.5 | 1015 (DEADLINE 100) | 8.8 |
| e2e2 1 | **on** | 92,631 | 33 | 149 | 133.9 | 27.9 | 106.0 | 1735 | 10.0 |
| e2e2 2 | off | 93,117 | 7 | 164 | 149.5 | 38.4 | 111.1 | 1157 (DEADLINE 100) | 9.4 |

- **Verify step, ~92K:** -9.5 to -15.6 ms (-7% to -10%), almost all in the GPU attention: 36-38 to 27.9 ms per step,
  i.e. 2.3-2.4 to 1.74 ms per call.
- **Why the rest gains less than at 64K:** the off state already runs the GPU at ~1.0-1.16 GHz here (more GPU work
  per step), so the wake-up savings shrink. This rests on one on-turn: repeat it with `ka_e2e.py --long` from a warm
  conversation when memory allows.

## Recommendation

1. **Adopt the in-process keep-alive.** Use `libdec_gpu_clk.dylib` with `DEC_KA_HOLD_MS=60`: one simdgroup, 20 us
   serial dispatches, 250 per command buffer, utility-QoS thread.
   - It needs no private API and changes nothing in the kernels or their output.
   - It burns only while split steps run: it stops 60 ms after the last one, so an idle server pays nothing.
   - `pipe/dec_split.py` change: load the new dylib (or make the dylib name an env var).
   - Optionally call `dec_ka_hold(60)` when a request's decode starts. The first verify step then also runs at full
     clock, since the ramp needs 10-20 ms.
2. **Expected savings per verify step** (16 GPU calls):
   - **64K:** -15.8 ms measured end to end (138.2 -> 122.4 ms; GPU attention 30.8 -> 20.8 ms; handoffs -5.9 ms).
   - **95K:** -10 to -15 ms (one end-to-end on-turn at 92.6K: 143-149 -> 133.9 ms). The GPU-only bound is
     16 x (2.35 - 1.54) = -13 ms in kernel time plus up to ~5 ms of handoffs.
3. **Cost:**
   - About +2 W combined during the decode (GPU at 1735 MHz instead of ~860; keep-alive ~0.8 W GPU, ~0.6-1 W CPU
     GPU-only, within noise in the server).
   - About +12% energy per decode cycle, with DRAM held at its top clock state (not measured).
4. **Re-check after OS updates.** The behaviour is a DVFS heuristic, and Apple DTS calls spinning keep-alives not
   future-proof. A 1-minute check: `gpu_clk_run.py OUT 3 2.5 gap=5 gap=5,ka=contm:250:20:1`.
5. **Not worth pursuing:**
   - Per-call kicks or pre-warms of any lead up to 5 ms.
   - Periodic or low-duty keep-alives.
   - Event-gated pre-commit, thread QoS, latencyCritical.
   - The private deadline / priority calls.
   - The performance-state assertion (unsupported for this process).

## Files

Sources:
- `pipe/unit/power/gpu_clk.swift` (`gpu_clk`) and `gpu_clk_run.py`: the regime probe.
- `gpu_keepalive.swift` (`gpu_keepalive MAX_S [US=20] [N=250] [TG=1] [QOS=ut]`): the external keep-alive. It exits on
  SIGTERM or SIGINT.
- `dec_clk_check.py`: the integration check.
- `ka_e2e.py`: the server check.
- `dec_clock.py`: today's first test. Its keep-alive is now stopped with SIGTERM; a background shell leaves SIGINT
  ignored in its children, which is why `gpu_load` never exited.
- `pipe/engine/dec_gpu_clk.swift`: the dylib source.

Raw outputs in `/Volumes/SSD4TB/anemll-forge-research/contention/gpu_clock/`:

| dir | what |
|---|---|
| `r1_screen64/` | 20 regimes x 4 rounds (keep-alive forms, periodic, full-GPU duty, pre-warm, event, QoS, latencyCritical) |
| `r2_confirm/` | 19 regimes x 5 rounds: keep-alive granularity, back to back + keep-alive, server pattern, 95K, gap 1.5 ms |
| `r3_grain_ramp/` | 23 regimes x 4 rounds: 5-100 us dispatches, keep-alive QoS, ramp 1-20 ms, private deadline / priority |
| `r4_cpu_cost/` | 8 regimes x 4 rounds: command buffers per second vs CPU W |
| `r5_dylib_check/` | `libdec_gpu_clk.dylib` through ctypes, server pattern, 4 rounds |
| `e2e1/` | the server check, 66-70K tokens (`requests.jsonl`, `summary.json`, `io.jsonl`, `run.log`) |
| `e2e2_92k/` | the server check, 92-95K tokens (turns 3-6 single-step answers) |
| `e2e3_90k_long/` | 90K, long answers; stopped by the swap watchdog after the cold turn (server prefill swap) |

Each run dir holds `phases.jsonl` (per phase, with every call's ms), `io.jsonl` (IOReport), `summary.json`,
`summary.md` and `run.log`. The powermetrics plist path is in `pm_path.txt`
(`/Volumes/SSD4TB/anemll-forge-research/contention/pm/gpuclk_<run>.plist`).

## Deployed in the server (18 Nov 2026, evening; `e2e_deploy/`)

`pipe/dec_split.py` now loads `libdec_gpu_clk.dylib` by default (`DEC_DYLIB`; `libdec_gpu.dylib` = without) with
`DEC_KA_HOLD_MS=60` (default; 0 = off). The dylib gained `dec_ka_set_hold` / `dec_ka_get_hold` (run-time hold), and
`dec_split` checks a switch file once per split step (`DEC_KA_CTL`, default `contention/ctl/ka`: "off" turns the
keep-alive off, anything else or no file = the 60 ms hold; each change is logged). The previous files are kept as
`dec_split.pre_ka.py`, `dec_gpu_clk.pre_set.swift`, `libdec_gpu_clk.pre_set.dylib`. Server banner at load:
`dec8: decode GPU engine .../engine/libdec_gpu_clk.dylib sha256 2b64e979a91a414f, kernels dec_kernels.metal
00f351b5edc893b7; keep-alive hold 60 ms (DEC_KA_HOLD_MS)`.

`power/ka_e2e2.py` (`ka_e2e.py` with the switch file instead of an external process) on the restarted forge (8765):
one cold request of 66,052 prompt tokens (keep-alive on), then 8 follow-up turns of 300 new tokens, 256 max tokens,
greedy; on = turns 1, 3, 5, 7, off = 2, 4, 6, 8, each switch confirmed by the server log. Medians over 4 turns each:

| keep-alive | ms / cycle | verify ms | GPU attention ms / step | GPU MHz | GPU W | CPU W | ANE W | combined W | J / cycle |
|---|---|---|---|---|---|---|---|---|---|
| on (60 ms hold) | 133 (132-133) | 119.1 (117.6-119.2) | 19.2 | 1735 | 3.40 | 1.90 | 5.70 | 11.02 | 1.46 |
| off | 150 (149-150) | 136.1 (135.4-136.3) | 29.9 | 891 | 1.36 | 1.76 | 5.24 | 8.36 | 1.25 |
| change | -11.3% | -12.4% | -36% | | +2.04 W | +0.14 W | +0.46 W | +2.66 W | +17% |

- **The same gain as the external keep-alive** (verify 138.2 -> 122.4, cycle 152.5 -> 137, +2.1 W at 66-70K).
- **Every turn agrees:** cycle 132-133 on, 149-150 off; the clock holds 1735 MHz on and runs about 880-905 MHz in DEADLINE off.
- **Power:** ANE W rises with the faster cycle (more ANE calls per second); energy per cycle +17%.
- **Cold turn (on):** cycle 131, verify 117.1, 11.1 W combined.
- tok/s differs between turns with the answers (tokens per verify 3.7-4.7), so cycle and verify ms are the measure.
- The switch file is left at "on"; forge stays up with the keep-alive.
