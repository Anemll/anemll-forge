# Batch T (Anemll, 20 Nov 2026 ~20:45): tree verify, why Splash 1.7.0's tree pays and ours does not

Pasted plan, verbatim:

After closing batch E: tree plan (Batch T), banner on every bench, number-first, no reboot, no push, stop/restore
forge around heavy runs. (T1) Measure Splash 1.7.0 on M6 with its 16-row tree on vs off (chain) on the same 9-prompt
2K/8K/32K rungs: tok/s, tok/verify, verify ms per cycle, draft ms. If Splash cannot turn tree off, use its published
or logged chain numbers and say so. (T2) Same breakdown for forge chain vs tree8 corrx0. (T3) Explain the gap per
component: tokens per verify gained by tree, verify cost multiplier per row count (Splash GPU 16-row vs our ANE
8/16-row), drafter tree quality (top-k breadth, how the tree is shaped), DeltaNet deferred state vs our pending tape.
(T4) Pick the biggest lever and prototype it, e.g. 16-row tree verify on GPU instead of ANE, or a better tree shape
from DFlash2 logits. Report after T3 before spending more than 2 h on T4.
