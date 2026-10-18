# Capstone final evidence report

Generated on **2026-10-18** from committed real-model artifacts. No mock
backend scores are used as research results. The source PDF defines three main
deliverables: a unified optimizer API, reproducible logs/costs, and concrete
recommendations. It asks Q1–Q3 about OOD gaps, robustness, and factors affecting
generalization; Q4 and cross-model transfer are stretch goals.

## Executive conclusion

The engineering deliverables are complete. Across the strongest controlled
sentiment evidence, OPRO selected the unchanged base prompt in all three search
seeds, producing **95.33% mean OOD accuracy** and a
**+0.33 pp source-minus-OOD gap**. A fresh simple-procedure
pilot found at most a **+0.31 pp** observed OOD gain for an unguarded rewrite,
while its independent confirmation guard rejected every replacement. A separate
round sweep increased mean logical search tokens from **0** at zero rounds to
**62,776** at 16 rounds without changing the selected prompt or final accuracy.

The six-method Qwen reasoning pilot likewise did not establish a reliable
optimizer advantage: the human-written prompt had the highest observed SVAMP
score, while OPRO and EvoPrompt-lite selected the original instruction. These
results support a scoped recommendation: **retain a strong base or human prompt
unless source-only confirmation establishes a gain, and always report target
transfer and search cost separately from development accuracy.**

## Slide-deck requirement status

| Requirement | Status | Evidence |
| --- | --- | --- |
| Unified API for prompt optimizers | met | Shared Optimizer/Candidate interface and common harness across fixed prompts, Random Search, OPRO, EvoPrompt-lite, and DSPy adapters. |
| Optimize on a source development split under a fixed budget | met | BudgetedBackend enforces request caps; controlled sentiment searches use frozen source-only train/validation data and explicit evaluation/proposal caps. |
| Evaluate the same selected prompt on source test and OOD targets | met | Sentiment evaluates SST-2, Amazon, and tweets; reasoning evaluates GSM8K and SVAMP. |
| Reproducible logs and costs | met | Committed protocols, seeds, frozen selections, request/token accounting, verification records, and selected run logs. |
| Q1: quantify OOD gaps across optimizer settings | met_with_pilot_scope | Controlled OPRO, simple-procedure sentiment pilot, and six-method Qwen reasoning pilot report source/OOD scores and gaps. |
| Q2: compare variance and worst-case OOD robustness | met_for_sentiment | Two sentiment OOD domains support worst-case and cross-domain variance; the one-domain reasoning pilot cannot inform variance. |
| Q3: identify factors affecting generalization | partially_met | Optimizer/search procedure, prompt length, and optimization-round cost are analyzed; broad causal conclusions require more models and partitions. |
| Concrete recommendations and report | met | Generated capstone report and README summary provide scoped recommendations, limitations, and reproducible provenance. |
| Cross-model transfer | stretch_not_required | Exploratory branch work exists, but it is excluded from the main conclusions. |
| Code/formal tasks | stretch_not_required | Not implemented; the deck explicitly labels this stretch work. |

## Main controlled sentiment result: OPRO

Protocol: Gemma 4 31B cloud alias; three source-only OPRO search seeds; 64 source
training and 64 source-validation examples; 300 frozen final examples each from
SST-2, Amazon, and binary tweet sentiment; 512 candidate/example evaluations and
at most 16 proposals per seed. Final target data was not used during search.

All three seeds retained this prompt:

> Classify the sentiment of the following text as Positive or Negative.

| Source test | Amazon | Tweets | Mean OOD | Worst OOD | OOD gap | OOD variance |
| --- | --- | --- | --- | --- | --- | --- |
| 95.67% | 95.00% | 95.67% | 95.33% | 95.00% | +0.33 pp | 0.000011 |

The identical selected prompt reuses the same frozen final predictions across
seeds; these are three search replications, not three independent final-test
replications.

### OPRO search costs

| Seed | Logical evals | Proposals | Logical tokens | Physical calls | Wall sec |
| --- | --- | --- | --- | --- | --- |
| 0 | 512 | 14 | 37,119 | 150 | 144.4 |
| 1 | 512 | 10 | 36,753 | 331 | 323.6 |
| 2 | 512 | 8 | 35,710 | 332 | 426.5 |

## Simple OOD procedures

This fresh one-seed Gemma 4 pilot compared the base and human prompts with three
custom history/feedback search procedures. These are component ablations, not
canonical OPRO, GEPA, or Random Search implementations.

| Method | Source | Amazon | Tweets | Mean OOD | Worst OOD | OOD delta | Prompt words |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 93.75% | 98.75% | 94.38% | 96.56% | 94.38% | +0.00 pp | 11 |
| human | 95.31% | 98.75% | 93.12% | 95.94% | 93.12% | -0.62 pp | 24 |
| history_full | 93.75% | 98.75% | 95.00% | 96.88% | 95.00% | +0.31 pp | 15 |
| history_full+guard | 93.75% | 98.75% | 94.38% | 96.56% | 94.38% | +0.00 pp | 11 |
| feedback_full | 95.31% | 98.75% | 94.38% | 96.56% | 94.38% | +0.00 pp | 32 |
| feedback_full+guard | 93.75% | 98.75% | 94.38% | 96.56% | 94.38% | +0.00 pp | 11 |
| feedback_racing | 93.75% | 98.75% | 94.38% | 96.56% | 94.38% | +0.00 pp | 14 |
| feedback_racing+guard | 93.75% | 98.75% | 94.38% | 96.56% | 94.38% | +0.00 pp | 11 |

The raw `history_full` rewrite had the largest observed OOD delta, **+0.31 pp**
with a paired 95% interval of **[0.00, +0.94] pp**. However, none of the three
raw winners improved the independent 80-example source confirmation set, so the
predeclared guard retained the base prompt in every arm. The full pilot used
**3,549 requests** and
**328,842 returned tokens**.

## Optimization rounds as a generalization factor

| Round cap | Confirmation accuracy | Mean logical search tokens/seed |
| --- | --- | --- |
| 0 | 95.31% | 0 |
| 1 | 95.31% | 8,004 |
| 2 | 95.31% | 10,861 |
| 4 | 95.31% | 16,995 |
| 8 | 95.31% | 31,131 |
| 16 | 95.31% | 62,776 |

All 0/1/2/4/8/16-round checkpoints across three seeds selected the same base
prompt. The frozen prompt scored **95.31%
mean OOD accuracy** and **93.75%
worst-domain accuracy**. This supports a cost conclusion for this pilot—not a
universal round-count rule: extra rounds increased search cost without changing
the selected instruction.

## Qwen 2.5 7B reasoning pilot

Protocol: one seed; 30 GSM8K development, 30 GSM8K test, and 30 SVAMP examples;
temperature zero; 512-token cap. GEPA/MIPROv2 rows are instruction-transfer
adapters rather than full compiled-program evaluations.

| Method | Dev | GSM8K | SVAMP | OOD gap | Opt calls/cap | Total calls |
| --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | 93.33% | 86.67% | 86.67% | +0.00 pp | 30/30 | 90 |
| Human-written | 96.67% | 86.67% | 93.33% | -6.67 pp | 30/30 | 90 |
| OPRO | 96.67% | 90.00% | 90.00% | +0.00 pp | 278/300 | 338 |
| EvoPrompt-lite | 96.67% | 90.00% | 86.67% | +3.33 pp | 278/300 | 338 |
| GEPA | 93.33% | 86.67% | 90.00% | -3.33 pp | 102/300 | 162 |
| MIPROv2 | 93.33% | 83.33% | 90.00% | -6.67 pp | 86/300 | 146 |

Because this pilot has one OOD domain, OOD variance is necessarily zero and
worst-case OOD equals SVAMP accuracy. It supports a cross-task consistency check,
not a broad robustness ranking.

## Answers to the research questions

### Q1: How large is the OOD gap across optimizer settings?

- Controlled sentiment OPRO/base: **+0.33 pp**.
- Fresh simple-procedure rows range from **-3.12 pp** to **-0.63 pp** source-minus-mean-OOD.
- Qwen reasoning rows range from **-6.67 pp** to **+3.33 pp**, but the 30-example, one-seed pilot is too small for a stable method ranking.

### Q2: Which prompts look robust?

In the controlled sentiment replay, the unchanged base prompt achieved
**95.00% worst-domain accuracy** with very low
cross-domain variance. In the fresh procedure pilot, `history_full` had the best
observed worst-domain score, but failed independent source confirmation; the
guarded choice therefore remained the base prompt. Robustness claims should use
multiple target domains—SVAMP alone cannot estimate cross-domain variance.

### Q3: Which factors affect generalization?

The measured factors with the clearest evidence are:

1. **Search procedure and confirmation:** target gains from raw rewrites were
   small and were not supported by independent source confirmation.
2. **Optimization rounds:** more rounds sharply increased tokens without
   changing the selected prompt in the round-sweep pilot.
3. **Prompt specificity/length:** longer prompts did not consistently improve
   mean or worst-case OOD accuracy in the fresh procedure pilot.
4. **Task/domain choice:** sentiment supports variance/worst-case analysis with
   two targets; the one-target reasoning pilot does not.

These are pilot-level associations, not general causal laws across models.

## Recommendations

1. Use a strong fixed prompt and human-written prompt as mandatory controls.
2. Add Random Search to future live matrices so search intelligence is separated
   from simply evaluating more candidates; it is implemented on `apinero` but no
   fresh real-model run was possible in the current credential-free environment.
3. Select replacements using source-only confirmation before any target test is
   opened; retain the base prompt on ties or unconfirmed gains.
4. Report development, source-test, every target-domain score, worst-domain
   accuracy, cross-domain variance, requests, tokens, and elapsed time.
5. Do not assume more optimization rounds are beneficial; justify them with an
   independent confirmation gain.
6. Treat DSPy instruction extraction and full compiled-program evaluation as
   different experiments.

## Scope and limitations

- The strongest sentiment evidence uses one model alias and one frozen partition.
- Identical final prompts share exact predictions across search seeds.
- The simple-procedure pilot uses custom component ablations, not every optimizer
  in the unified harness.
- The Qwen reasoning comparison uses one seed and 30 examples per split.
- No new real-model calls were made while consolidating this report: the current
  environment has GPUs but no local model weights/server, no provider API keys,
  and no installed real-data/DSPy packages. Existing measured artifacts were
  consolidated rather than replacing them with mock results.
- Cross-model transfer and code/formal tasks remain stretch work.

## Provenance

- `apinero` base revision: `0af8f3eee7b7ed952486c9e7ab63287becd7090f`.
- Controlled sentiment source revision: `origin/albert` at `98a911810a5defcbd772af22035ed46b08274560`.
- Compact source artifacts are committed under `capstone_results/controlled_sentiment/source/`.
- Qwen pilot artifacts remain under `prompt_opt_harness/prompt_opt_harness/logs/`.
- Rebuild with `python3 tools/build_capstone_report.py`.
