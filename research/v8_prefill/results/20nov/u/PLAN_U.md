# Batch U (Anemll, 20 Nov 2026 ~22:2x), pasted verbatim

Decision: defer the 16-row build. Tree8 default is Anemll call, keep it behind the flag for now. Next batch U, banner
on every bench, number-first, no reboot, no push, stop/restore forge around heavy runs: (U1) End-to-end measured, not
offline: served tree8 plus lookup-in-8-rows policy vs chain on the 9-prompt 2K/8K/32K/64K rungs plus a code-heavy and
an agent-style set (Pi-like tool calls), tok/s and tok/verify and replies identical count, so Anemll can decide the
default from measured numbers. (U2) Root cause of our 16-row verify cost (+10% at 8K, +16% at 32K per cycle) vs
Splash ~0: per-layer breakdown of the 16-row vs 8-row cycle (ANE FFN, GPU attention, DeltaNet solve, handoffs), HWX
check that 2x bonded compute is engaged at 16 rows, and whether 16 rows hits a compute knee from batch E1 sweep.
Propose the cheapest change that makes 16 rows nearly free, estimate only, no new weight build without asking. (U3)
Write up U1-U2 in README, commit locally, report.
