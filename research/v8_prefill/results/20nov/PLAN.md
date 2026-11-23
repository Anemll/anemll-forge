# YOLO overnight plan from Anemll (Nov 19 10:32 PM PT, ~10 hours, no feedback expected)

Anemll's words: continue the tree draft implementation with dynamic 16 and 32. You have permission to stop the server, iterate, fix, improve, benchmark, and make the tree implementation as efficient as possible. Target: about 4 accepted tokens per verify cycle on average. When the tree work is done, continue JetSpec++ (JetX). Work autonomously; do not wait for replies. Keep going item to item.

Rules (unchanged standing rules):
- Never reboot M6, never touch other services. Stopping/restarting forge :8765 is OK. Restore a working forge on :8765 at the end (best served config).
- Every bench prints the one-line banner (resolved flags, kernel variant, tree size mode, dylib/metallib path + hash) outside the timed region and aborts on mismatch. Per-variant output checksums.
- Never invent timings. Greedy output must stay token-identical to the current chain path (check on a fixed prompt set); report KL/identity.
- Keep a running log in research/v8_prefill/results/20nov/README.md (decisions, numbers, rejected ideas) and a one-line status file /tmp/m6_yolo_status.txt: `STATE=<working|blocked|done> ITEM=<..> NOTE=<..>` updated at every item change.
- Commit locally in small commits; do NOT push. Training-style long jobs must checkpoint/resume.

## Phase A: served tree verify, dynamic 8/16/32
Context: yesterday 13b/15 measured ANE 4-layer group verify +26-39% for 16 rows and +87-116% for 32 rows vs 8-row chain, and projected +4.5–4.9% for top-7 in eight rows on 12 low-acceptance greedy requests (964 cycles). The goal now is to make 16/32 cheap enough that wide trees win, and push mean accepted tokens/cycle to ~4 on realistic (non-repetitive) prompts.
Ideas to try (your call, measure each): 
1. Real tree drafting from DFlash-2 (best-first by path prob), ancestor masks in verify attention, per-branch DeltaNet/conv state handled correctly (commit only accepted path).
2. Make wide verify cheap: GPU-side verify rows (Splash 1.7.0 runs 16-row verify on GPU at ~+0.9%), or split (ANE FFN batched rows + GPU attention with ancestor mask), or ANE entries built for 16/32 with better tiling; reuse shared prefix work across siblings.
3. Dynamic per-cycle size choice 8/16/32 from expected accepted tokens per ms using measured costs (online fitted, like Splash), including at >64K split-verify.
4. Tune drafter temperature/top-k branching, depth vs width.
Deliverable: tok/s and mean accepted/cycle at 2K/8K/32K/64K/96K on a realistic prompt set (code, math, chat; not just the 77% repetitive smoke) for chain vs tree8 vs dynamic 8/16/32, plus power. Make the best one the served default if it wins and is token-identical; otherwise leave it behind a flag and explain why.

## Phase B: JetSpec++ (JetX)
Continue item 7: tap capture (32-prompt GSM8K, forge down OK), J3-R3 replay, then compare acceptance/tok/s vs DFlash-2 chain and tree; integrate if it wins. Restart forge on :8765 when done.

When everything is done, write a number-first summary at the top of 20nov/README.md and set STATE=done.
