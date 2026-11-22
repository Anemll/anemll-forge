# Lookup routing on a long synthetic coding session

[Read the completed measurements and decision](RESULTS.md). The fresh two-seed test measured +0.6% for the context gate and +1.5% for the existing yield gate versus served tree8 with lookup off. [Holdout data](holdout_results.json)

This research compares the stored yield rule with request-local corrections based on whether recent lookup proposals paid off. Timing comparisons use the production eight-row verifier with its default sampled tree. Only the approved research launcher can start a device window. The research files live outside the served source.

## Evaluation fixed before the new holdout

- Development uses the existing 19-turn long session with seed 7100 and existing sampled short observer traces. The concurrent context-gate run also uses that 19-turn development session and seed.
- The independent holdout fixes 19 different tracked source files, different edit instructions, and the same system and tool format. Its file manifest was frozen before inspecting new development outcomes. A matched pair shares a seed and complete prompt body. The two blocks use different seeds and reversed launch order. Only digests, timings and counts are stored.
- Baselines are lookup off and the existing yield rule at margin 1.25. Feedback rules adjust the lookup-yield prediction from the past three lookup outcomes in the same short/long match bucket. One alternative vetoes short matches following two poor outcomes. A shadow-feedback alternative also learns from earlier declined proposals after the generated continuation reveals their first mismatch or full seven-token match. The context gate under independent test requires a match of at least 48 below 32K and still requires the same yield margin.
- Short and development replay compares proposals at observed tree anchors and estimates lookup cost by subtracting drafter time. It omits incomplete futures and never takes labels across a discontinuity. Changing a decision changes the subsequent anchors and potentially changes sampled text. A replay rate is a screening result.
- Confirm the strongest credible candidate on the disjoint session. The serving promotion gate is at least 5% pooled improvement over lookup off, with no context band repeatedly losing more than 2% across two fresh seed blocks. Report original per-block rates and paired request outcomes. Keep the served default after every research window.

## Files

- `archive_audit.py` and `archive_audit.json` are a fresh matched-turn analysis of the three saved long arms. The retrospective turn choices reuse recorded outcomes and cannot establish a new policy's speed.
- `extract_short.py`, `short_counts.json`, `prior.json`, `replay.py` and `short_results.json` cover the short observer screen. The priors are the previously published M6 table, already fitted on short data.
- `research_hook.py`, `sitecustomize.py` and `check_hook.py` provide a process-scoped observer or feedback rule. The observer compares outstanding draft proposals in memory, writes scalar counts, and discards unmatched tails at request boundaries.
- `window.zsh` and `long_client.py` use the existing standard M6 launcher, idleness checks, 1024 MB watchdog, banner checks and default restoration. The runner only addresses Forge on port 8765. `holdout.py` rebuilds fixture text in memory from the frozen `holdout.json` manifest.
- `dev_summary.py` and `holdout_result.py` score the developmental observer and the fully paired holdout. Raw output from live windows is under `runs/`.

The M6 runtime source digests are in [runtime.sha256](runtime.sha256). The [development context A/B](ctx_gate_compare.json) is a byte-for-byte copy of the completed comparison in the 21 November lookup-gate results.
