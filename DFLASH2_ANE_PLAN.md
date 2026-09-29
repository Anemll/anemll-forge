> Historical research notes imported on 2026-09-29. Paths were generalized; results were not rerun. Later sections may supersede earlier findings. See the current [release workflow](docs/WORKFLOW.md).

# DFlash2 speculative decoding for Qwen3.8-27B on the ANE: plan and spec

Status: living document. Numbers in section 9 are updated as they are measured.

Target: `Qwen/Qwen3.8-27B` (qwen3_5 hybrid, 64 layers, 48 Gated DeltaNet + 16 gated full attention, hidden 5120,
vocab 248320), running as the quantized ANE build `full_mix25_mixer4_head4`.
Drafter: `ProCreations/Ternary-Bonsai-2-27B-DFlash2` @ `4cfb6ad`, BF16 master in the SN8100 handoff bundle.
It was trained on **Ternary-Bonsai-2-27B** features, not on our target, so acceptance on our target has to be measured
(section 9).

Files (all in `scripts/`):

| file | role |
| --- | --- |
| `dflash2_drafter_ref.py` | torch drafter reference with an ANE-shaped context ring; `validate` compares it with the bundle's `dflash/model.py` |
| `dflash2_target_ref.py` | streamed, stateful torch target (bf16 or dequantized export); block verify with pending commit; `simulate` measures exact greedy acceptance, `replay` re-scores drafter variants offline |
| `dflash2_gdn_lazy.py` | one-layer ANE prototype of the lazy-commit DeltaNet state (decode / verify / prefill functions share one state layout) |
| `dflash2_ane_drafter.py` | drafter Core ML model (context update + 8-row block + head in one call); `check` (parity), `time`, `replay` (exact acceptance on saved traces) |
| `dflash2_quant.py` | drafter quantization variants (RTN / GPTQ via `qwen3_lut_common`) and reduced draft vocabularies, scored by exact replay |

## 1. Drafter math (bundle reference `code/dflash-07ebd93/dflash/model.py`)

Settings: 5 layers, hidden 5120, 32 q heads / 8 kv heads, head dim 128, MLP 17408, RMSNorm eps 1e-6 (plain gain,
**not** zero-centered like the target), RoPE theta 1e7 over the full 128 dims (NeoX half split), block 8
(anchor + 7), sliding window 2048 (non-causal), conv taps 2 / group 16 (320 groups), selector top-k 16 / rank 256,
mask token 248070 (`<|audio_start|>` in our vocab), no embedding scale, no logit multiplier or softcap.

**Target features.** For every token t whose target state is committed, the target provides
`f_t = concat(out_5, out_19, out_33, out_47, out_61)` (25600). `out_i` is the raw residual stream after 0-based
decoder layer i has finished, i.e. after its MLP residual add, which is also the input to layer i+1. No final norm
is applied. (HF `hidden_states[i+1]`; vLLM adds 1 to the config ids for the same reason.)

**Context.** `c_t = RMSNorm_hidden(fc(f_t))` (25600 -> 5120). Per draft layer l:
`K_l(t) = RoPE_t(k_norm_l(k_proj_l(c_t)))`, `V_l(t) = v_proj_l(c_t)`. These come from the drafter's own K/V
projections, not from the target's K/V. They are computed once per committed token and kept in a ring of W = 2048
slots (slot = t mod 2048). Only accepted tokens enter the context; rejected rows never do.

**Query block** at positions p..p+7: `x = E_target([anchor, mask x 7])`. Per layer:

```
n  = RMSNorm_in(x)
d  = conv_proj_attn(n)                  # (8, 1280) = [side 0 | side 1] x [tap 0 | tap 1] x 320 groups
a  = gconv(n, d_side0, base_attn[0])    # a[t] = (b0 + d0[t]) * n[t] + (b1 + d1[t]) * n[t-1]; n[-1] = 0 (block start)
q  = RoPE(q_norm(q_proj(a)));  kb = RoPE(k_norm(k_proj(a)));  vb = v_proj(a)
o  = softmax(q [K_ctx | kb]^T / sqrt(128) + mask) [V_ctx | vb]       # GQA 4:1, non-causal
x  = x + gconv(o_proj(o), d_side1, base_attn[1])   # side-1 coefficients come from n, not from the branch output
n  = RMSNorm_post(x);  d = conv_proj_mlp(n)
x  = x + gconv(down(silu(gate(m)) * up(m)), d_side1, base_mlp[1]),   m = gconv(n, d_side0, base_mlp[0])
```

Output: `h = RMSNorm_final(x)`. Rows 1..7 (positions p+1..p+7) predict the tokens **at** those positions.

**Mask.** Query q at position P sees ring key k at position K iff the slot is valid and |P - K| < 2048. It also sees
all 8 block keys (non-causal inside the block). W = 2048 slots is enough: the block needs keys p-2047..p-1.

**Selector** (greedy):
`logits = lm_head_target(h[1..7])`, then `(unary, cand) = top16(logits)` and `hp = hidden_projection(h[1..7])` (256).
Starting from `pred = anchor`, for i = 1..7:
`score_c = unary_c + <pred_code[pred] * hp_i, succ_code[cand_c]>`, `pred = cand[argmax score]`.
This is a sequential walk, not independent top-1 choices and not Viterbi. The two codebooks (248320 x 256 each) stay
on the host (17 row lookups per position).

**Validation** (`dflash2_drafter_ref.py validate`, fp32, real target embedding/head, random target-like features):
the three cache cycles (37 ctx rows then +3 then +8) match with hidden rel err 3.5e-6 to 5.6e-6. The two window
cases (2100 ctx rows, then +5) match at 6.9e-5 / 1.2e-4; that residual is the reference's fp32 RoPE angles at
positions > 2000. Top-16 sets and all 7 drafted tokens are identical in all five cases.

## 2. Speculative loop contract (greedy)

Convention: the target state covers [0, p). The anchor a at position p has been emitted but not yet consumed.

```
drafter:  ctx += features of the rows committed last cycle (positions p_prev .. p-1)
          d1..d7 = propose(anchor a at p)
target:   verify [a, d1..d7] at p..p+7 (causal)  -> argmax rows t0..t7, feature rows f_p..f_p+7
accept:   m = longest prefix with d_i == t_{i-1} (0..7);  bonus = t_m
emit:     d1..dm, bonus
commit:   target state through a, d1..dm (k = m + 1 rows); drafter context gets f_p..f_p+m
next:     p += m + 1; anchor = bonus
```

Stop at the first EOS (248044 / 248046) among the emitted tokens. Metrics: mean accepted m per block,
tokens per target call = m + 1, and the P(m >= i) vector.

## 3. What the target must provide

- Feature taps: outputs of layers 5, 19, 33, 47 and 61 for **every** consumed row, in **every** function (decode T=1,
  prefill T, verify 8). Otherwise the drafter context has holes. With the current chunk boundaries (L00-11, 12-23,
  24-35, 36-47, 48-59, 60-63) this means extra outputs after layers 5, 19, 33 and 61; the output of L36-47 already is
  tap 47. Shape (1, 5120, 1, T) fp16. No re-chunking is needed.
- A verify function: 8 rows at an **arbitrary** start position p. Causal mask over the committed K/V [0, p) plus
  itself; K/V written at p..p+7; DeltaNet rows left pending (section 4).
- A head for 8 rows (logits (8, V), argmax on the host). The drafter needs the same lm_head without the target's
  final norm; for now `dflash2_ane_drafter.py` carries its own copy.
- Embedding stays on the host (anchor row plus the mask-token row).

## 4. State rollback: lazy commit (no snapshots, no replay)

- Full attention K/V: no rollback is needed. The verify writes K/V for all 8 rows at p..p+7. After accepting k rows,
  the next block starts at p+k <= p+8 and rewrites p+k..p+k+7, which covers every rejected row. The host mask exposes
  only [0, p) plus the block's own causal rows.
- DeltaNet recurrent state cannot be cropped. Options:
  - snapshot + replay: doubles target work per cycle and removes the speedup;
  - per-row state snapshots: 8 x 1.5 MB x 48 layers per cycle;
  - **lazy commit** (chosen): the verify call does not commit its own rows. It stores the per-row quantities of the
    DeltaNet update as pending rows in the state. The **next** call first applies the first k of them (k from the
    host, as a mask; masked rows get beta = g = 0, which is an exact no-op) and then processes its own rows from the
    committed state. The extra cost per call is one 8-row chunked-delta per DeltaNet layer (no projections).
- One shared state layout for decode, prefill and verify (P = 8 pending rows, H = 3 conv history rows):
  - `conv (H + P, 10240)`: [3 raw qkv rows before the last call's first position | the last call's raw qkv rows,
    zero padded]. Every call writes `[prev_sel @ read | its own rows]`, where `prev_sel (3, H + P)` is a host
    one-hot picking the 3 rows before this call's start (rows k, k+1, k+2 after k accepted rows). That allows any
    alignment. The conv reads its history back from the **returned** value.
  - `rec (48, 128 + 3P + 1, 128)`: rows 0..127 hold S; then per pending row its key k_j, its UT-transformed value
    `u = (I + N)^-1 (beta v)` and key `wk = (I + N)^-1 (beta k e^cum)` (N_ij = beta_i k_i.k_j e^(cum_i - cum_j),
    i > j), and one row `[cum_0 .. cum_7, 0 ...]` (inclusive log-decay cumsum). The call that produced the rows
    computes these from its own inputs.
  - commit (next call; host `commit (1, P, 1) = [1]*k + [0]*(P-k)`, `commit_last` = one-hot of row k-1):
    `S' = S e^total + (k_j c_j e^min(total - cum_j, 0))^T (u - wk S)`, total = cum_{k-1}.
    - Exact for any k: the system is unit lower-triangular, so the solution's leading rows do not depend on later
      rows.
    - The `min(., 0)` keeps masked rows finite; otherwise exp overflows and 0 * inf gives NaN.
    - No inverse is taken of state-derived data.
  - verify / decode (lazy): value = concat(S', own k, u, wk, cum rows); outputs come from the returned S'. There
    are no read-derived side consumers.
  - prefill T > P: commits its own rows. The outputs are routed through the same extra rows (the coordinator's
    `REC_OUT=concat`), which fits T <= 3P + 1 = 25; a larger prefill T needs `128 + max(25, SCR_ROWS)` rows. The
    next call passes commit = 0.
- **Host-owned DeltaNet buffers (`GDN_IO`)**: the same scheme with plain inputs/outputs and no state rules.
  - Per DeltaNet layer, inputs are `conv_in (3, 10240)`, `rec_in (48, 128, 128)` and
    `pend_in (48, 3P + 1, 128)`, plus the shared `commit` / `commit_last`.
  - Outputs are `conv_out (T + 3, 10240)` (all rows), `rec_out` = S' (the previous block's accepted rows applied)
    and `pend_out` (this block's k, u, wk, cum rows).
  - Host per cycle: `rec_in <- rec_out`, `pend_in <- pend_out` (plain pass-through, zero-copy with IOSurface
    backings); `conv_in <- conv_out[k : k + 3]` (a 60 KB copy).
  - A committing call (prefill / decode outside speculation) outputs its final S and zero pending rows.
  - For rare mode switches (verify -> plain decode, prefix-cache snapshot) the host can also apply the pending
    commit itself: `S' = S e^total + (k c e^min(total - cum, 0))^T (u - wk S)`, about 0.6 GFLOP over 48 layers.
  - `dflash2_gdn_lazy.py` with `IO=1`: decode1 / verify8 (lazy) / prefill16 load on the ANE. Accuracy is identical
    to the MLState variant: CPU >= 0.9996, ANE 0.83-0.95 on the mixer-only random-input test (same as `gdn_block`).
- ANE state rules this satisfies: one `coreml_update_state` per state; a raw read only feeds its own state's update
  value; every other consumer uses the returned value. (The coordinator confirmed that side consumers fail to load on
  the ANE with -14.)
- **EIR bug found:** `matmul(N, N)` fails with -14 ("Failed to build the model execution plan") whenever N derives
  from a state value, including via the returned conv buffer. It also fails as `matmul(n, n * 1)`, while
  `matmul(I - N, I + N)` loads. The chunked-delta Neumann series therefore uses `N^2 = I - (I - N)(I + N)`. This
  affects `chunked_delta` whenever GDN_CHUNK >= 4.
- Prototype `dflash2_gdn_lazy.py` (layer 30, dense fp16 weights). Mixed sequence: prefill16 -> verify8 with
  commits 0/3/8/1/5 -> decode1 x2 -> verify8 -> prefill16 after a verify -> decode1. Mixer-only token cos vs the
  fp32 step-decode reference:
  - CPU fp16: >= 0.9996, no drift across commits.
  - ANE: all three functions load; cos 0.83-0.95. The existing decode `gdn_block` measured the same way (same
    layer, same random N(0,1) inputs) gives 0.88-0.96 on the ANE (0.99995 on CPU). So this is ANE fp16 behavior on
    a harsh mixer-only test, not the lazy design.

## 5. Drafter on the ANE (`dflash2_ane_drafter.py`)

One call per cycle: context update + 8-row block + selector projection (+ head when it fits under 2 GB).

- Inputs:
  - `feat (1, 25600, 1, 8)`: rows committed last cycle, zero padded;
  - `ctx_write (8, 2048)`: one-hot slot per row, a zero row for padding;
  - `ctx_cos / ctx_sin (8, 128)`;
  - `anchor (1, 5120, 1, 1)`: embedding row (mask rows are constants);
  - `q_cos / q_sin (8, 128)`;
  - `mask (8, 2056)`: additive, [ring slots | block].
- States: `kc0..4 / vc0..4 (8, 2048, 128)` fp16, 40 MB. One masked write per state per call; the attention reads the
  returned value.
- Outputs: `hidden (1, 5120, 1, 8)` (final-normed), `hp (1, 256, 1, 8)`, logits of rows 1..7 when the head is inside.
- Host: embedding row, top-16 per row, codebook gathers, and the 7-step selector walk (~17 x 256 multiply-adds per
  step).
- Dynamic conv without dynamic weights: reshape (1, 5120, 1, 8) -> (320, 16, 8), multiply by
  (base (320, 16, 1) + coef (320, 1, 8)), then add the one-row shift (concat a zero column). The zero at the block
  start is structural.
- Weights: fc 131 M, 5 layers x 356 M, selector projection 1.3 M. LUT4 per-tensor + per-channel scale is about
  0.9 GB; INT8 per-channel is about 1.8 GB. The draft head (target lm_head, LUT4 + pcs) is 0.64 GB.

- fp16 on the ANE (all needed for correctness):
  1. **Massive activations.** Layer 0's MLP writes values around 1.2e6 into the drafter's residual stream. From
     layer 0's attention add on, the graph carries residual / 256, folded into the per-channel scales of LUT
     o_proj / down_proj (applied to the input for INT8, whose scale / 256 would be subnormal). RMSNorm is scale
     invariant.
  2. **Scale-free RMSNorm** (`rms_robust`): `xs = x / max|x|`, `out = xs rsqrt(mean(xs^2) + eps / max^2) w`.
     - Squares stay <= 1, so the massive rows do not overflow and q/k projections of ~460 do not either.
     - eps stays exact where it matters: the mask-token embedding has rms 3e-3, so eps = 1e-6 shrinks those rows
       by ~5%, and dropping it changes the drafts.
     - Zero padding rows stay 0. The old `rms_hidden` gave 0 * inf = NaN on them, and the masked ring write
       spread that NaN into the context state.
  3. **LUT + per-channel scale** runs as `LUT[idx] @ x` first, with the channel scale applied after. That raw
     accumulation overflows where the true output does not (layer 0 down_proj: raw 1.7e5). Fold f = 1/64
     (down_proj) or 1/16 (other LUT matrices) into the per-tensor LUT and 1/f into the channel scale. INT8 does
     not show this.
- Parity (LUT4 RTN, ANE vs torch with the same dequantized weights, 150-row context + 4 cycles with commits
  1/3/5/7): hidden cos >= 0.9991, rel err 1.4-3.0%, top-16 overlap 0.97-0.99, 28/28 drafted tokens identical.
- Timing on the M6 ANE, one call per cycle (context rows + 8-row block + head); the ctx64 prompt call is 3.0 ms:

  | variant | package | draft call |
  | --- | --- | --- |
  | LUT4 + INT8 conv proj, full 248K head | 1.57 GB | 15.3-15.9 ms |
  | same, body only (no head) | 0.93 GB | 9.5 ms |
  | LUT4, 32K-row draft head | 1.02 GB | 10.1 ms |
  | 2-bit vector MLP, full head | 1.24 GB | 13.3 ms |
  | 2-bit vector MLP, 32K-row head | 0.68 GB | 7.9 ms |
  | LUT4, 48K / 64K-row head (2 x 32K parts) | 1.06 / 1.10 GB | 10.2 / 10.7 ms |
  | 2-bit vector MLP, 64K-row head | 0.77 GB | 9.1-9.8 ms |

  These timings are noisy when the coordinator's builds or compiles run (the same package measured up to 2x slower
  then). Re-time on an idle M6 before final numbers. Host work per cycle: top-16 over 7 x 248K takes 0.8-1.0 ms (torch); the selector walk and masks are < 0.3 ms.

## 6. Verifier on the ANE (prototype once prefill is validated)

Per chunk, a `verify` function (T = 8) shares weights and KV states with `infer` / `prefill`. The DeltaNet state
is either MLState (lazy-commit layout, section 4) or host I/O (`GDN_IO`).

Per cycle (target state committed through p0 + k_prev - 1; block [a, d1..d7] at p = p0 + k_prev):

```
host:   x = E[a, d1..d7] (8 rows), cos/sin rows p..p+7, kv_write one-hot rows p..p+7,
        mask (8, CTX): key j visible iff j <= p + t  (committed [0, p) + causal inside the block; stale rows > p+t hidden)
        GDN: commit / commit_last for k_prev (+ prev_sel (MLState) or conv_in = conv_out_prev[k_prev : k_prev + 3] (I/O))
target: 6 chunk calls (verify) -> y (1, 5120, 1, 8) + taps 5/19/33/47/61 (8 rows) [+ GDN buffers (I/O)]
head:   8-row head -> argmax per row (host, 8 x 248K)
host:   m = prefix match, bonus = t_m, emit; k = m + 1 committed rows for the next call; drafter context += taps[:k]
drafter: one call (writes those k rows + drafts the next block) -> 7 tokens
```

- The 8-row verify always writes K/V at p..p+7. The next block starts at p+k <= p+8, so it overwrites every rejected
  row. Its causal mask never looks past its own rows.
- Per cycle cost = verify (6 chunks x 8 rows) + 8-row head + draft call + host (~2-3 ms: argmax, top-16, masks).
  The prefix-cache snapshot of the server must flush the pending rows first (host-side commit), or store them as
  well.
- EOS: stop at the first EOS among the emitted tokens. The anchor may itself be EOS.
- Prompt: target prefill (all rows committed) outputs the taps for every prompt row. The drafter ingests the last
  2047 rows with `ctx64` calls, which can overlap the target prefill. Then the first cycle uses anchor = argmax of
  the last prompt row.

## 7. Open items

- Quantized-target acceptance: the run is in progress (interim close to bf16). Then replay the ANE drafter on
  `traces_q_full_mix25.npz` for the deployment number.
- GPTQ for the drafter is optional: LUT4 RTN is already -1.6% vs bf16. The M3U GPTQ eval was interrupted by memory
  pressure.
- Verify: measure the real 8-row verify cost. It dominates tok/s, and the 1.3x decode figure is a placeholder.
- Target rebuild: taps in every function; verify T=8 at arbitrary p; DeltaNet layout (MLState lazy commit or GDN_IO
  with pend_in/pend_out); head for 8 rows (+ a no-norm variant to share the lm_head with the drafter).
- Prompt-cache snapshots in the server must include or flush the pending DeltaNet rows.
- Beyond the Bonsai drafter as is: a short distillation of the drafter on our (quantized) target's own features and
  outputs could raise acceptance. Not started.

## 8. Commands

```
# M3U (memory-safe, streamed)
python dflash2_drafter_ref.py validate
python dflash2_target_ref.py check                   # target implementation vs bf16 reference log-probs
N_PROMPTS=8 MAX_NEW=256 python dflash2_target_ref.py simulate
TRACE_TAG=bf16 python dflash2_target_ref.py replay   # drafter variants on saved greedy traces
python dflash2_target_ref.py layers                  # fp32 block/pending/commit path vs step-decode reference
DEQ_DIR=... TAG=q python dflash2_target_ref.py simulate   # quantized target (dequantize_export once first)
# M6
UNITS=CPU_AND_NE python dflash2_gdn_lazy.py            # lazy-commit GDN (MLState); IO=1 for host-owned buffers
QUANT=lut4 python dflash2_ane_drafter.py build         # VOCAB_N=65536 VOCAB_FILE=... for a reduced head
QUANT=lut4 python dflash2_ane_drafter.py check         # parity vs torch (same weights)
QUANT=lut4 python dflash2_ane_drafter.py time
TRACES=~/Models/dflash2/traces/traces_bf16.npz QUANT=lut4 python dflash2_ane_drafter.py replay
```

## 9. Results

### Acceptance with our target (exact greedy speculative decoding, `dflash2_target_ref.py simulate`)

Setup:
- 16 prompts from the KL set (code, math, science, writing, agentic, one Chinese), chat template, thinking on, up to
  256 generated tokens.
- Target: streamed fp32 math on the bf16 checkpoint.
- Drafter: bf16 master, head = target lm_head, block 8 = anchor + 7 drafts.
- The simulation is exact: the target runs block verification with pending commit, so the emitted text equals the
  target's greedy output.

| target | blocks | mean accepted m | tokens / target call (m + 1) | draft accept rate | P(m >= 1..7) |
| --- | ---: | ---: | ---: | ---: | --- |
| bf16 Qwen3.8-27B | 1189 | **2.345** | **3.345** | 33.5% | .68 .48 .36 .29 .22 .17 .14 |
| quantized `full_mix25_mixer4_head4` (dequantized, fp32 math), **interim** at pass 43 | 688 | 2.427 | 3.427 | 34.7% | (bf16 at a similar stage: ~2.55) |

The quantized-target run (`sim_q_full_mix25`) is still going and writes `traces_q_full_mix25.npz` when it ends. The
deployment number = the ANE drafter replayed on those traces.

- Histogram of m (0..7): 382 236 138 92 74 66 33 168.
- By prompt: math (coin flips, equation, train, Bayes) 3.5-4.9; code 1.5-4.5; explanations (sky, attention, crypto,
  Hamlet, rate limiter) 1.6-2.0; Chinese 1.0.
- Qwen3.8's thinking style ("We need answer user: ...") is predictable early on; acceptance falls as the reasoning
  gets specific.

### Drafter quantization / reduced head (exact greedy replay on the bf16 traces, all 16 sequences)

`dflash2_ane_drafter.py replay` runs the ANE drafter through the same block schedule on the saved greedy traces
(target tokens + target tap features). The head is the target's LUT4 lm_head, as deployed.

| drafter (ANE) | package | tokens / target call | vs bf16 (3.345) |
| --- | ---: | ---: | ---: |
| LUT4 per-tensor + pcs RTN, INT8 conv projections, full 248K head | 1.57 GB | **3.290** | **-1.6%** |
| INT8 per-channel RTN, 64K head | 1.97 GB | 3.224 | -3.6% |
| LUT4 RTN, 64K head | 1.10 GB | 3.125 | -6.6% |
| 2-bit vector MLP (+ LUT4 rest), full head | 1.24 GB | 3.117 | -6.8% |
| LUT4 RTN, 48K head | 1.06 GB | 3.069 | -8.3% |
| 2-bit vector MLP, 64K head | 0.77 GB | 2.996 | -10.4% |
| LUT4 RTN, 32K head | 1.02 GB | 2.980 | -10.9% |

- ANE vs torch with identical quantized weights: 2.528 vs 2.528 mean accepted over the same 286 blocks (4 sequences).
  The ANE's fp16 numerics cost nothing.
- torch replays on the M3U with the bf16 head: bf16 drafter 2.338 (the simulation gave 2.345; the replay truncates
  at trace ends), INT8 RTN 2.347 (+0.3%).
- Draft heads are frequency ranked: the KL traces of the 48 other prompts + WikiText, blended with BPE id order;
  specials always kept. Held-out token coverage: 32K 93.7%, 48K 96.0%, 64K 97.3%.

### End-to-end estimate

tok/s = (m + 1) / (verify + draft call + host).
- Assumes verify(8 rows) = 1.3 x 83 ms = 108 ms (coordinator's placeholder).
- Host = ~2 ms (8 x 248K argmax, top-16, selector, masks).
- The baseline is plain decode at 83 ms/token = 12.0 tok/s (2K context).

| drafter | m + 1 (bf16 target) | draft call | est. tok/s | speedup |
| --- | ---: | ---: | ---: | ---: |
| LUT4 RTN, full head | 3.290 | 15.3 ms | 26.3 | 2.2x |
| LUT4 RTN, 64K head | 3.125 | 10.7 ms | 25.9 | 2.2x |
| 2-bit MLP, full head | 3.117 | 13.3 ms | 25.3 | 2.1x |
| 2-bit MLP, 64K head | 2.996 | ~9.5 ms | 25.1 | 2.1x |

Verify dominates the cycle, so acceptance is worth more than drafter milliseconds. The <= 8 ms drafter variants lose
more acceptance than they save time. Recommendation: LUT4 drafter with the full head, one 1.57 GB call, ~15 ms
(~12% of the cycle). Sensitivity: with verify at 1.5x decode (125 ms), LUT4 full gives 23.3 tok/s.
