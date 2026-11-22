# DFlow on DFlash2: scoping plan (22 Nov)

Source: arXiv 2609.06498, Sections 4 and 4.3. Motivation: PAPER_SWEEP.md A3. The best 7 nodes from the drafter's own
candidates hold +8.6% / +10.2% more tok/verify, and only the target's verify rows carry the signal to find them. This
is a plan only: nothing was trained, run or captured. All times and sizes are estimates.

## 1. What DFlow adds, mapped to DFlash2

**What the paper adds:**
- After the first rejection at a+1, the target's tap-layer states over the rejected suffix (a+2..K) pass through the
  drafter's shared W_c and an RMSNorm, giving h_i.
- A small MLP_phi takes `[E(bonus); E(first rejected); h_i]` and returns channel-wise beta_i and gamma_i.
- The next round's mask inputs at those positions become `E_mask + beta_i * LN(h_i) + gamma_i`. All other positions
  keep the plain mask embedding.
- Loss: L0 + L1 + L2. Each term is the DFlash position-decayed cross entropy, over k = 2 relay rounds that share
  parameters. The target is frozen and its states are detached.
- The rollouts must come from the real target verify. Without them (teacher forcing), tau falls to 2.22, against
  4.50 with them.
- Setup: 80K ShareGPT samples, 6 epochs, learning rate 6e-4, block 16, drafter trained from scratch. The paper does
  not give the MLP width or the batch size.
- Results: T=1 tau +9-12%. Overhead is 0.51 ms per round (+1.2%). The gain is the relay itself: 3.66 with the relay
  off, 3.93 with it on.

**Mapping onto DFlash2:**
- W_c is our `fc` (25600 -> 5120) followed by `hidden_norm`. The relay inputs are the 5 taps at the verify rows, which
  the target already outputs.
- The verify rows sit at p..p+7. With m accepted, the bonus is the next anchor at p' = p+m+1. Target row q predicts
  q+1 and drafter row q predicts q, so verify row q feeds next-block slot q+1-p'.
- That gives 7-m relay rows (slots 1..7-m), or 6-m if the paper starts at a+2. Test both. There is no relay at m = 7.
- E(bonus) is already the anchor input. The only new embedding is E(d_{m+1}).
- MLP_phi: 15360 -> 1024 -> 10240 (our choice of hidden width). That is about 26M parameters, 1.6% of the drafter.
- Zero-initialize beta and gamma, so an untrained relay equals the served drafter.
- Use `rms_robust`: the mask embedding has an rms of 3e-3, against about 1 for LN(h). Keep mask scale 0.7.
- Relay rows never enter the 2048-row ring.

**What block 8 changes:**
- At most 7 relay rows instead of 15.
- More cycles accept the whole block and get no relay (35 of 53 in greedy trace gsm8k 0001). Sampled serving runs at
  about 4.08 tok/verify, so most cycles reject.
- Use DFlash2's block-8 decay constant.
- tree8 verifies a tree: the relay follows the rejected node's verified spine, and some rejected nodes have no
  verified children. The prototype measures how often relay rows exist.

## 2. Training data

The existing captures (56 traces, about 28K tokens, 1.4 GB, greedy) hold trajectory taps only. They have no
rejected-suffix taps, so they are prompts and drafter context only.

Each training cycle needs:
- the draft, m and bonus;
- the 8 verify-row taps (410 KB fp16);
- the ring's trajectory taps (51 KB per token).

That is about 615 KB per cycle raw, or 120 KB if stored after `fc` (fc frozen).

**Prototype data:** `dflash2_target_ref.py simulate` on M3U, with the dequantized served export and a verify-row
dump. 5K cycles, greedy and T=1: about 3 GB in 2-4 h.

**Full data:** a server capture switch that dumps the verify-row taps.
- 150-250K cycles: code, math, chat and Pi, 2K-32K context, mostly sampled, tree8 plus some chain.
- About 5-8 h of forge windows at roughly 11 cycles/s.
- 25 GB fused or 125 GB raw, on SSD4TB.
- These rollouts are exact on-policy round-1 data for the frozen drafter, with no target pass during training.

## 3. Training recipe

**Arms:**
- **Arm A, first:** train MLP_phi only, with body and fc frozen, on captured k = 1 rounds. The served LUT4 GPTQ body
  is reused unchanged, so precision variants share weights.
- **Arm B:** Arm A plus LoRA on q/v/o.
- **Arm C:** full fine-tune with k = 2.

**M3U (Arms A and B):**
- Precompute the ring K/V per trace once. A cycle then costs about 113 GFLOP: the 8 rows forward and backward, plus
  the target head.
- 200K cycles over 6 epochs is about 140 PFLOP. At 8-12 TFLOPS on MPS that is 4-6 h of compute; allow 10-20 h.
- Memory: under 30 GB.

**Colab (Arm C):**
- Round 2 needs the target online: 54 GB bf16, plus DeltaNet state snapshots at every anchor.
- Full Adam on 1.6B parameters needs about 26 GB.
- That is 2 or more 80 GB GPUs for 1-3 days, via Grok, and only if Arm A or B passes and plateaus.
- Cheaper option: a second capture served with the Arm A candidate (DAgger style).

**Quantize and build:**
- MLP_phi: LUT4 GPTQ, calibrated on the relay inputs. Fallbacks: LUT6, or fp16 at 52 MB.
- Arms B and C: re-run the `drafter_lut4_gptq_q7_cal` GPTQ recipe, then replay parity.
- `coreai/dflash2_coreai_build.py` `DraftEntry` gains three inputs, for 20 in total against the binding limit of about
  51:
  - `rfeat` (1, 25600, 1, 7) through the shared `fc`;
  - `remb` (1, 5120, 1, 1);
  - `rvalid` (1, 1, 1, 7).
- Compile with `forge.py compile` only.

## 4. Serving changes

**`draft_cycle` in `qwen38_server.py`:**
- Slice the verify-row taps (already present) for the rejected spine, shift them by one and pad them to 7.
- Gather E(d_{m+1}) and set `rvalid`.
- Send zeros at m = 7, on lookup cycles, and after a restore or a new conversation.
- On tree8, map the spine nodes to slots.

**ANE cost estimate:**
- fc over 7 more rows costs about 1.8 GFLOP, and its weights are already streamed in that call.
- MLP_phi is 13 MB.
- Expect +0.2-0.5 ms on the roughly 6.5 ms drafter body, under 0.5% of a cycle.

**Lossless A/B:**
- Verification and the accept rule stay unchanged, so greedy output must stay identical with the relay on and off.
- With `DFLOW=0` (rvalid = 0), the new package must match the served package bit for bit, since it only adds zero.
- That gives one package for both arms, and `DFLOW=0` is the rollback.
- ABBA in forge windows on 8765 with same-text requests: tok/verify and decode tok/s, sampled and Pi long session.
  Check reviewer windows first.

## 5. Risks and go/no-go gate

**Risks:**
- A frozen body may ignore an input-level injection. The paper trained jointly. Arm B covers this.
- Under tree8, few relay rows may exist.
- Block 8 leaves fewer slots.
- The torch target used for the prototype data may drift from the ANE target.
- The capture windows compete with serving.

**Prototype gate (M3U, offline, no ANE):**
1. Train Arm A on 4K of the prototype cycles and hold out 1K.
2. On the held-out rejecting cycles, redraft the next block with the relay on and off.
3. Score greedy cycles exactly, as the accepted prefix against the trajectory. Score sampled cycles as the expected
   accepted length from the logged target probabilities.
4. Also report the top-16 recall at relayed slots and the A3 ceiling on the same cycles.
5. Controls:
   - embeddings only (h zeroed), which separates the relay from self-conditioning;
   - chain versus tree8 relay availability;
   - a+1 versus a+2 suffix start.

**Go:** a projected overall tok/verify gain of at least +4% (the gain per rejecting cycle x their share), with the
embeddings-only control at least 2 points lower. Then run the full capture and Arm A.

**Stretch:** +3-4% leads to Arm B on the prototype first.

**Stop:** under +3%, or when Arm B does not beat Arm A by 2 points.

**Approval:** the capture windows and the Grok run are drafted for Anemll's approval, not sent.
