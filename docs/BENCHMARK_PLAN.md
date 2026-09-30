# Quality benchmarks for the Core AI ANE release

Research checked on **2026-09-29**. This is a proposed evaluation plan, not measured results. Keep the model card's KL-only status until a versioned report from the actual release bundle is available. Start with the smallest context that fits each complete prompt and output budget, using the Swift Core AI runtime **with the matching DFlash2 drafter enabled**. Plain target-only runs are separately labeled numerical and acceptance diagnostics.

## What KL can and cannot establish

Teacher-forced KL measures fidelity to a reference model's next-token distribution on fixed contexts. It does not test whether the answer is correct, code passes tests, instructions are followed, or a tool call succeeds. Matching the teacher also preserves its mistakes. Near-tied probabilities can change a greedy token, after which free generation follows a different context. Distillation can deliberately change probabilities while retaining or improving useful task accuracy.

Use three separate kinds of evidence:

- **Fidelity:** held-out KL, top-1 agreement, error percentiles and first greedy divergence.
- **Capability:** instruction checks, code tests, mathematical answer accuracy and tool-call validity.
- **Deployment:** the same tasks on the released compiled graph, with output lengths, failures, latency, memory and runtime configuration recorded.

Perplexity is a useful additional language-model diagnostic, particularly for conversion regressions, but is not a sufficient quality ranking by itself. None of these metrics replaces the others.

Our historical [`qwen38_kl.py`](../scripts/qwen38_kl.py) uses teacher top-256 plus an aggregate tail bucket, scores prompt and answer positions, and averages over tokens in nats. That coarsening loses distinctions within the tail and is a lower bound on full-vocabulary KL for the same distributions. Its MPS evaluation of reconstructed export weights is also distinct from evaluating the compiled Core AI graph. Preserve the historical metric for continuity; add explicitly named public, assistant-only and full-vocabulary checks as new evaluations. See [the quantization guide](QUANTIZATION.md#interpret-kl-carefully).

The [September 29 top-512 check](results/kl_topk_comparison_2026-09-29.json) raises mean KL by only 0.1714% on the same historical trace. It changes the probability partition, while retaining the historical corpus and prompt-plus-answer scoring; it therefore does not make these results directly comparable to Mirai's public evaluator.

## The original model is the primary baseline

The [upstream Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B) reports **Terminal-Bench 2.1 (Terminus) 73.0**, **IFBench 79.5**, **GPQA Diamond 89.2** and **LiveCodeBench v6 90.3**. These are upstream-reported results, not measurements of Forge. Some listed agent evaluations use 256K context. The card does not provide the complete Terminal-Bench protocol or all general-task grading/sampling details, so the table alone is insufficient for exact reproduction.

Use published scores as external context, then run a **matched BF16 baseline** for our own experiments. Match task IDs/revisions, prompt and agent harness, reasoning mode, sampling, output/turn budgets, usable context and compaction policy, grader and number of trials. Our prepared Core AI build reaches roughly 64K context, with a 65,472-row capacity in the largest entry; choose a common usable window with output headroom for BF16 and ANE. That constrained experiment is distinct from an upstream larger-context result.

Run BF16 on M3U or another suitable machine and ANE on M6, or sequentially on one host if each model fits individually. For fixed question benchmarks, save each model's responses and grade them offline. For Terminal-Bench, every model must run its own agent trajectory from a fresh task environment; replaying BF16's actions only tests imitation, not task success. Neither approach requires the models resident together.

Publish three separately labeled values where available: **upstream reported score**, **our BF16 score under our protocol**, and **our ANE score under that same protocol**. Quantization/runtime retention is assessed by the latter pair, with per-task wins/losses and uncertainty. Differences from an upstream score may also come from the evaluation setup. Model-quality and latency differences across engines should remain identifiable.

## What the other providers publish

### Unsloth Dynamic 3.0

The exact [Qwen3.8-27B GGUF collection](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) has multiple storage tiers. Its [Dynamic 3.0 report](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs) emphasizes KL, teacher top-1 agreement and a 300-prompt, 32-token greedy trajectory comparison. These measure fidelity rather than solved-task accuracy. The report excludes MTP from its size axis and distinguishes calibration from evaluation. Pin each downloaded GGUF revision; results for Dynamic 2.0 or a different Qwen model are not interchangeable.

### Prism ML Ternary-Bonsai-2-27B

Use the Qwen3.8-based **Bonsai 2**, not the earlier Qwen3.6 Bonsai. The [card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) reports a 14-task thinking average of 84.78 versus 86.32 for FP16, including two vision tasks. The [newer paper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf) reports 20 tasks and different aggregates. Its budgets reach 81,920 output tokens, with repeated samples for GPQA and AIME. Card/paper IFBench grading labels differ. Preserve source-specific settings instead of combining them. The [demo](https://github.com/PrismML-Eng/Bonsai-demo) requires its supported Prism runtime fork.

### Mirai Labs

[Mirai S experimental](https://huggingface.co/trymirai/Qwen3.8-27B-S-experimental) is the low-bit comparator: nominal 2.4 bpw and 8.45 GB, explicitly provisional. Its card has speed examples but no scored task-accuracy table. [M](https://huggingface.co/trymirai/Qwen3.8-27B-M) uses four-bit PTQ followed by distillation and reports a partly private evaluation mixture. The [public KL evaluator](https://github.com/trymirai/kl-eval) defaults to teacher top-512 plus tail and assistant-only scoring. Its metric and corpus differ from ours. The [L chart](https://huggingface.co/trymirai/Qwen3.8-27B-L/resolve/main/assets/kl-vs-size.svg) concerns eight-bit L, not S quality.

## Recommended sequence

### 1. A small fixed screen

Run **64 IFEval, 32 HumanEval+ and 64 MATH-500 problems**, with sample IDs selected before seeing candidate outputs. Stratify instruction types and math difficulty; do not simply take the first contiguous rows. Begin with **thinking disabled, greedy generation and a 1,024-token output cap**, identically for the original reference and every candidate. This tests useful behavior within a modest budget. It is a screening subset, not an official full benchmark score.

At an **illustrative 20 generated tokens/s**, all 160 outputs hitting that cap would take **2.28 hours of decoding**. Add prompt prefill, warm-up, model loading and scoring. Actual early stops reduce this time; a slower target increases it. This is arithmetic for planning, not a measured speed of the release runtime. Run a 10–20-task pilot first and use its observed token counts and times to estimate the remainder. Keep all selected tasks even if an early screen fails.

Small subsets expose large regressions, but cannot substantiate near-equality: one changed answer in 64 problems is 1.56 percentage points. Repeat with more examples before ranking small differences. Score capped outputs with the pinned verifier and report truncation separately; a capped output may already contain a valid answer. Missing required final answers fail. An additional completion-before-cap success rate can be reported separately.

### 2. First release-quality task report

Prioritize the following full sets, one completion per problem:

- **[IFEval: 541 prompts](https://huggingface.co/datasets/google/IFEval).** Objective instruction checking without an LLM judge. Publish prompt-level strict and loose accuracy, plus instruction-level scores. Use [Google's verifiers](https://github.com/google-research/google-research/tree/master/instruction_following_eval). Allow 2,048 output tokens initially; some prompts require long answers. Score the final answer and retain reasoning separately.
- **[HumanEval+: 164 problems](https://huggingface.co/datasets/evalplus/humanevalplus).** Generate once and run the original and extended tests on the same answer. Publish base HumanEval and HumanEval+ pass@1. [EvalPlus](https://github.com/evalplus/evalplus) supports compatible chat endpoints. Allow 2,048 output tokens initially. Execute generated programs in an isolated test environment with resource limits.
- **[IFBench: 300 prompts](https://huggingface.co/datasets/allenai/IFBench_test), next.** More challenging verifiable constraints; useful for formatting, counting and instruction degradation. Start with a pilot to select an adequate output cap because several prompts require unusually long answers. Its [authors' standard report](https://github.com/allenai/IFBench) is prompt-level loose accuracy; also publish strict accuracy and identify the scorer version. This also addresses the card/paper label discrepancy above.

These have objective local scoring and overlap Prism's published suite. Our capped, greedy or non-thinking configuration must be labeled as such; it does not replicate their long-budget sampled scores. Rerun the chosen reference and competitors under our configuration for a valid comparison.

For IFEval plus HumanEval+, **705 answers averaging 400–800 generated tokens** require **3.92–7.83 decoding hours at 20 tokens/s**, before prefill/scoring. Those output averages are assumptions to replace with the pilot. A 2,048-token cap gives a much higher 20.05-hour decoding ceiling if every answer exhausts it. Do not promise an overnight finish based only on the cap or past speculative-serving speed.

### 3. Reasoning and knowledge within a declared budget

- **[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500):** expand to a preselected, difficulty-stratified 128, then all 500. Use a pinned mathematical-equivalence grader. Add a distinct thinking run with a 4,096- or 8,192-token total output budget after the non-thinking screen. For 128 problems capped at 4,096, the decoding ceiling at 20 tokens/s is **7.28 hours**. Report incomplete reasoning/empty final answers. Prefer this over using only GSM8K, where the reported Bonsai/baseline scores are near a ceiling.
- **[GPQA Diamond](https://huggingface.co/datasets/Idavidrein/gpqa):** an optional harder science/reasoning check. Pilot 64 before the complete Diamond set; fix answer-choice shuffling, prompts, seed and sample count. A one-sample capped run is a different configuration from Prism's repeated long-budget evaluation.
- **[MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro):** use a fixed stratified 280-question screen (20 per domain) if broad knowledge coverage is needed. The full set has over 12,000 questions. A short direct-answer variant or subset is not the published CoT benchmark. Do not substitute forced-choice likelihood scoring for generated reasoning without labeling the change.
- **Tool use:** a pinned non-live BFCL subset, or an explicitly custom public set of 50–100 tool-schema cases, can cheaply expose malformed calls, wrong arguments and unnecessary repeated calls. Do not label the custom set as BFCL or agent success. Multi-turn environment execution is a separate evaluation.

Defer full SWE/Terminal-Bench agents and repeated long-budget AIME/LiveCodeBench runs until the short tests and integration checks work; the Terminal-Bench pilot below can start after endpoint integration. A small number of AIME questions does not make it cheap when each is sampled repeatedly with tens of thousands of reasoning tokens. Vision benchmarks do not apply to our text-only runtime bundle.

### 4. A public fidelity report on the compiled ANE graph

Choose a held-out public mixture of chat, code, math and multilingual text totaling **32–64K scored token positions**, distinct from calibration and plan selection. Save token IDs, prompt/assistant masks, dataset revisions, a stable hash and pinned teacher probabilities once. Compare all candidates on the same teacher-forced contexts. A public subset of Mirai's mixture can be useful, but does not reproduce its private-data result.

Report mean/median/p95/p99 KL, teacher top-1 agreement, target-token NLL/perplexity and scores by domain. Use the same top-K/tail partition across engines, or stream full-vocabulary logits and accumulate metrics without keeping all logits. Label approximate KL explicitly. Record a same-weight floating-point versus compiled-graph check to separate quantization error from conversion/runtime error.

The current HTTP server provides generated chat responses, not a logprob scoring endpoint. Adapt direct runtime scoring for this stage: `CoreAIQwen.call(ids)` returns all logits for up to eight teacher-forced rows. Follow each call with `accept(len(ids))` to commit those rows before the next batch, and reset state between independent sequences. `call()` alone does not advance the committed position. The 64-row prefill API returns only the final row's logits, so it cannot supply all-position KL by itself. [`qwen38_coreai_verify.py`](../scripts/qwen38_coreai_verify.py) demonstrates eight-row scoring, but its default comparison is Core ML versus Core AI on a local WikiText stream; it is not a ready-made upstream-BF16 quality benchmark. Estimate time using measured **teacher-forced rows/s**, not ordinary generated tokens/s. Building the teacher reference is a separate cost and can be done on another machine.

An additional 100–300 public-prompt, 32-token greedy comparison can test first divergence and short free-generation behavior. Call this our own trajectory screen unless the exact Unsloth prompt set and scorer are available. Output mismatch alone does not establish incorrect answers.

### 5. Terminal-Bench pilot against BF16

Terminal-Bench tests completed work in container environments and can reveal cumulative mistakes, recovery failures, command/output handling and repeated-action loops. It measures the **agent plus model plus environment**, so hold the rest of that system fixed. Use **2.1**, the version named by Qwen, and pin its task revision rather than mixing results with 2.0. [Official dataset](https://github.com/harbor-framework/terminal-bench-2-1).

[Harbor's Terminus-2 reference agent](https://www.harborframework.com/docs/agents/terminus-2) supports a custom model endpoint, turn limits and context summarization. First run an integration task to verify that our server's final-answer JSON, reasoning fields and stop behavior work with the agent parser; a compatible chat API alone does not establish working agent integration. Execute task tools in the benchmark's Linux containers, separately from the macOS ANE model server. Verify container architecture and dependencies before timing model performance.

Then preselect **10–15 tasks**, spanning code repair, shell/file operations and data processing, before viewing model outcomes. Run BF16 and ANE with the same task verifier, agent commit/parser, resources, context policy, reasoning effort, sampling and budgets. Start with one trial per model/task as a labeled pilot. Retain every selected task, including failures; publish IDs and trajectories. Expand to repeated trials and the complete set after estimating cost from the pilot. It is not a leaderboard score.

Record task success, first-pass and repeated-trial outcomes, total generated tokens including reasoning and summarization, turns, malformed agent messages, command failures, loops, timeouts, context compactions and elapsed time. Separate environment/verifier failures from model failures without silently dropping trials. If summaries are used, pin that policy and identify which model generates them.

For quality retention, allow enough wall time that a slower ANE server is not simply denied the matched token/turn budget. Also report a separate fixed-deadline deployment test if completion speed is part of the objective. Altered timeouts must be disclosed and cannot be presented as the standard benchmark protocol.

As a planning illustration, **10 trials consuming 20K generated tokens each at 20 tokens/s** take **2.78 hours of decoding per model**, or **5.56 hours for the BF16/ANE pair if both run at that rate**. This is not an enforced Harbor budget or measured duration. Repeated multi-turn prefill, summaries, environment setup, command execution and verification add time; BF16 may have a different rate. Use pilot observations for scheduling.

The current [submission documentation](https://github.com/harbor-framework/terminal-bench-2-1/blob/main/README.md) specifies at least five trials per task and currently closes community submissions. With 89 tasks, five trials would mean 445 agent runs per model, or 890 for the pair, before any competitor models. Local research needs no public upload. Review transcripts, task outputs, and data rights before sharing benchmark artifacts; use public-upload flags only when intentionally publishing reviewed results.

## Fair comparison and implementation details

Include the original pinned Qwen3.8 reference, our Core AI bundle, nearby Unsloth size tiers such as Q2_K_XL/Q3_K_XL, a higher-precision Unsloth control, Bonsai 2 and Mirai S; add Mirai M when resources permit. Recheck current revisions and supported runtimes before downloading. Q8 is a practical secondary control, not the unquantized ground truth.

Run Forge on ANE and competitors on their supported engines. There is no requirement to convert all comparator formats to Core AI. Save raw requests and outputs and use one evaluator to score them. Record engine numerical differences as a confounder; task quality and ANE speed are separate comparisons.

Keep upstream checkpoint/tokenizer identity, rendered prompts, thinking mode, stop tokens, total output cap, context, sampling/seed, penalties and scorer fixed. Run the ANE release with its pinned DFlash2 drafter. Record target/drafter hashes, accepted drafts per cycle, emitted tokens per verifier call and drafter/verify/context-update time. Use `--plain` only for a separately labeled matched diagnostic; its timings do not represent release serving performance. For sampled tests use identical sampling parameters and sufficient repetitions; a common seed alone cannot guarantee identical random streams across engines.

Our server expects `chat_template_kwargs: {"enable_thinking": false}`; Mirai S's documented API uses a top-level `enable_thinking`. Adapters must verify rendered behavior rather than silently sending the same JSON field to incompatible engines. Set explicit temperature; do not inherit different server defaults. Use concurrency one for our single-request runtime.

For a thinking comparison, set our `thinking_budget: 0` to disable injected early reasoning closure unless every engine implements the same disclosed closure policy. Still enforce the shared total token cap and record truncations. Preserve `reasoning_content` and `content`, and verify that the grader receives the final answer. Disable optional DRY/loop guards or record their termination as part of an explicitly named serving-policy evaluation.

Report each task separately with paired score changes and confidence intervals, including reference-correct → candidate-wrong and the reverse. Do not average unrelated metrics or combine text-only results with a vendor aggregate containing vision and long-horizon agents. Do not tune the quantizer on the final test set. Record crashes, parser failures, empty answers, loops, timeouts and truncation rates alongside accuracy.

Show **effective weight bytes**, full download bytes, and loaded memory separately, with common inclusion rules for embeddings, head, MTP and vision. Our historical 9.069 GiB weight estimate and approximately 13.2 GB target-only staging inventory have different scopes. The complete speculative bundle adds the drafter and selector assets; report their download bytes and loaded memory explicitly. Neither can be directly placed on a vendor's weight-only size axis without reconciliation.

Before claiming a release score, save the HF/Git commits, artifact hashes, hardware/OS/compiler, harness/dataset revisions, sample IDs, request configuration, output tokens, wall time, scorer outputs and uncertainty. The existing [validation status](VALIDATION.md) remains unchanged until these runs are executed. No new model downloads or benchmark executions were performed for this research plan.
