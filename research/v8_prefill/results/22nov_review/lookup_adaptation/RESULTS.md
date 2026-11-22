# Prompt lookup on a fresh M6 coding session

**Keep prompt lookup off for general serving. The context gate missed the qualification target on the fresh session.** Forge finished on its served sampled-tree8 configuration with prompt lookup off. [Holdout comparison](holdout_results.json) · [Service restoration](runs/20261122T021951Z_confirm_79990/restoration.txt)

## Fresh holdout

Nineteen synthetic coding turns exercised 9,072–55,344 prompt tokens. The file payloads and edit instructions differ from development. Each policy saw the same request bodies within a block. The two blocks used different generation seeds and reversed arm order. Six launches produced 114 measured requests plus six warmups. The threshold fixed before testing was ≥5% pooled decode throughput over lookup off, with no context band losing more than 2% in both blocks. [Frozen file manifest](holdout.json) · [Protocol](README.md) · [Raw arms](runs/20261122T021951Z_confirm_79990/)

| Policy | Seed 8700, tok/s | Seed 18700, tok/s | Pooled tok/s | Gain over off | Tokens/verify | Gross J/streamed token |
|---|---:|---:|---:|---:|---:|---:|
| Served tree8, lookup off | 30.24 | 31.06 | 30.66 | +0.0% | 4.01 | 0.873 |
| Existing yield gate | 30.04 | 32.15 | 31.12 | +1.5% | 4.05 | 0.858 |
| Context + yield gate | 29.58 | 32.10 | 30.86 | +0.6% | 4.03 | 0.882 |

| Context | Turns per seed | Existing yield vs off | Context gate vs off | Context gate, seed 8700 | Context gate, seed 18700 |
|---|---:|---:|---:|---:|---:|
| <16K | 3 | +16.7% | +0.1% | +0.3% | −0.2% |
| 16-32K | 7 | −3.7% | +0.6% | −0.4% | +1.6% |
| 32K+ | 9 | +1.1% | +1.1% | −4.5% | +6.1% |
| All | 19 | +1.5% | +0.6% | −2.2% | +3.4% |

The context gate changed throughput by −0.9% against the existing yield rule and beat lookup off on 15/38 paired turn timings. Output hashes agreed with lookup off on 21/38. The two context-gate blocks took 143 lookup cycles versus 324 under the existing yield rule. On the subset with identical output, the context gate was 0.36% slower in the first block and 0.50% slower in the second. That subset is selected by output agreement. [Paired results and per-turn totals](holdout_results.json)

The off and context-gated arms spent 723.8 and 728.5 seconds respectively from request submission through completion across 38 turns. Their streamed token counts were 13,725 and 13,830. Decode throughput divides total completion tokens after the first by total time between the first and last streamed output. Gross device energy uses the same interval with one-second samples. Energy coverage was 99.9% and 100.0% of streamed tokens in these two arms. [Measurements](holdout_results.json)

The existing yield gate gained 15.1% and 18.2% below 16K on the two seeds. This was the same three early turns each time, and their output hashes all changed. Above 32K the context gate swung from −4.5% to +6.1%. Context length alone gave little guidance on which file sequence or seed would benefit. [Matched bands](holdout_results.json)

## What predicted a worthwhile copy

The lookup drafter matches recent history against an earlier suffix, then proposes up to seven following tokens. The yield gate compares predicted tokens per millisecond with recent tree8 throughput. The context gate also requires a 48-token history match below position 32,768. Above that position it uses the existing yield rule. Eight verifier rows still check the anchor and up to seven proposed tokens. [Gate protocol](README.md)

On the original 19-turn session, the existing yield gate measured +0.6% overall. It lost 2.7% at 8–16K and 4.2% at 16–32K, then gained 8.0% at 32–57K. Even an oracle choosing between the two stored outcomes for each complete turn could reach only +4.78% over the averaged off controls on those records. The independent reversed-order context test on the same prompts and seed measured +2.8% pooled and +8.3% at 32–57K. [Archived turns](archive_audit.json) · [Context A/B](ctx_gate_compare.json)

The tree8 observer recorded 832 rounds and retained 789 with a complete future and usable timing. The prospective rules that used past outcomes screened at +0.77% (light prior), +0.73% (heavier prior), +0.73% (short-match veto), +0.74% (including declined proposals). The existing yield rule projected +0.76% and the context gate +1.19%. The context gate ranked first and had already reproduced a live gain, so it advanced to the fresh holdout. [Observer analysis](dev_results.json) · [Scalar round records](runs/20261122T021414Z_observe_76469/observe_counts/)

Recent outcomes offered little separation for short matches. When at least two prior outcomes were available, the next lookup would have beaten tree8's estimated cost on 15/25 short matches after outcomes at or above the prior and 15/27 after below-prior outcomes. Only 11 long matches had comparable feedback history. The 16–32K band had 17 eligible lookup proposals, of which 16 had a history match shorter than 16. At match lengths 8–11, lookup averaged 3.0 tokens per verify and tree8 averaged 6.0 across four such rounds. These are comparisons at the same observed anchors. A live lookup decision can alter later anchors and sampled text. [Per-band counts and cost estimates](dev_results.json)

On the 36-request short screen, the existing yield rule projected +43.9% on code and +4.6% on tool-agent. Counting outcomes of declined proposals shifted these to +43.7% and +5.2%. In the measured context-gate A/B on those cohorts, sampled code gained 18.0% and sampled tool-agent gained 1.1% versus off. Shared fixture structure and the different cohort mix limit transfer to live Pi sessions. [Short screen](short_results.json) · [Measured cohort A/B](ctx_gate_compare.json)

## Next experiment

Use full-file rewrites and ordinary tool-use turns as separate held-out cohorts. First label the task from the request format before generation. Compare routing by that label with lookup off using paired seeds, streaming throughput, whole-request latency and unchanged-output pairs. The measured +18.0% sampled-code result makes copied-file edits the positive control. The fresh long-session result is the guardrail. Keep the general serving decision tied to the 5% pooled and repeated 2% band criteria. [Cohort measurements](ctx_gate_compare.json) · [Fresh holdout](holdout_results.json)

## Reproduction and scope

The full M6 checkout stores the raw arms in `runs/20261122T021951Z_confirm_79990/`. Run `python3 -B holdout_result.py runs/20261122T021951Z_confirm_79990` from this directory to reproduce the holdout table. Run `python3 -B dev_summary.py runs/20261122T021414Z_observe_76469` for the development screen. The first observer launch restored before sending a request. The runner now creates the status file before starting the launcher. The scored observer and six holdout arms each used the standard launcher, checked banners and ran with a +1024 MB swap watchdog. [Run records](runs/)

These measurements cover one disjoint file sequence, two seed blocks, one model build and one M6. The history thresholds, recent-yield weights and speed/energy figures are specific to that setup. General-purpose code drafting from earlier text and target verification can be tested on other models with their own measurements.
