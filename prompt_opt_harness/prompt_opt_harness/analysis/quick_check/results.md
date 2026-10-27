# Results: when do optimized prompts overfit?

Model `qwen2.5:7b`, max_tokens 512, seeds [42], 10 source-test and 20 new-dataset questions per seed. Optimizers were tuned on the source dev sample only (the oracle excepted).

## Findings (auto-generated from the logs)

- **Token cap:** Cut-off answers are a real problem: up to 3.3% of answers were truncated (Zero-shot), and dropping them moves a gap by up to 0.050. Rerun with a larger cap (`MAX_TOKENS=1024 LOG_DIR=logs/after_1024 bash run_all.sh`) before drawing conclusions.
- **Versus Zero-shot (95% bootstrap CIs over questions):** no reliable gain over Zero-shot even on the source: OPRO, EvoPrompt-lite, GEPA-lite, TextGrad-lite, Random-search.
- **Versus the human-written prompt:** No optimizer reliably beats the human-written prompt on the new dataset.
- **Dev optimism:** the largest dev − test gap is OPRO (+0.00); a large value means the search picked a prompt that looked better on the dev sample than it really is.
- **Most robust:** OPRO has the best worst-case accuracy over the new datasets (0.90).
- **Headroom:** tuning on the new dataset (oracle) reaches 0.85 on it, no better than 0.90 for the best source-tuned optimizer (OPRO): seeing the target domain did not help here.
- **Cost:** no optimizer improved source-test accuracy over Zero-shot, so there is no gain per API call to report.

## 1. Headline metrics (mean ± sd over seeds)

| Method | Seeds | Dev | Source test | Dev − test | New data | Source − new gap | Worst case | Variance | API calls | Tokens | Gain / 1000 calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Zero-shot | 1 | 0.80 ± 0.00 | 0.90 ± 0.00 | -0.10 ± 0.00 | 0.80 ± 0.00 | +0.10 ± 0.00 | 0.70 ± 0.00 | 0.0100 | 40 | 11k | n/a |
| Human-written | 1 | 0.90 ± 0.00 | 0.80 ± 0.00 | +0.10 ± 0.00 | 0.90 ± 0.00 | -0.10 ± 0.00 | 0.90 ± 0.00 | 0.0000 | 40 | 14k | -2.50 |
| OPRO | 1 | 0.90 ± 0.00 | 0.90 ± 0.00 | +0.00 ± 0.00 | 0.90 ± 0.00 | +0.00 ± 0.00 | 0.90 ± 0.00 | 0.0000 | 84 | 29k | +0.00 |
| EvoPrompt-lite | 1 | 0.90 ± 0.00 | 0.90 ± 0.00 | +0.00 ± 0.00 | 0.90 ± 0.00 | +0.00 ± 0.00 | 0.90 ± 0.00 | 0.0000 | 84 | 28k | +0.00 |
| GEPA-lite | 1 | 0.90 ± 0.00 | 0.90 ± 0.00 | +0.00 ± 0.00 | 0.90 ± 0.00 | +0.00 ± 0.00 | 0.90 ± 0.00 | 0.0000 | 88 | 29k | +0.00 |
| TextGrad-lite | 1 | 0.90 ± 0.00 | 0.90 ± 0.00 | +0.00 ± 0.00 | 0.90 ± 0.00 | +0.00 ± 0.00 | 0.90 ± 0.00 | 0.0000 | 88 | 32k | +0.00 |
| Random-search | 1 | 0.90 ± 0.00 | 0.90 ± 0.00 | +0.00 ± 0.00 | 0.90 ± 0.00 | +0.00 ± 0.00 | 0.90 ± 0.00 | 0.0000 | 84 | 30k | +0.00 |
| Oracle-OPRO | 1 | 0.90 ± 0.00 | 1.00 ± 0.00 | -0.10 ± 0.00 | 0.85 ± 0.00 | +0.15 ± 0.00 | 0.80 ± 0.00 | 0.0025 | 71 | 21k | +1.41 |

Dev − test and Source − new gap are signed: positive means accuracy fell. Worst case is the lowest accuracy over the new datasets. Gain / 1000 calls is the source-test gain over Zero-shot per 1000 API calls (all calls, search included).

## 2. Do the gains transfer? Paired comparison with Zero-shot

| Method | Gain on source (pts) | 95% CI | Gain on new data (pts) | 95% CI | Transfer shortfall (pts) | 95% CI | Reading |
|---|---|---|---|---|---|---|---|
| Human-written | -10.0 | [-30.0, +0.0] | +10.0 | [+0.0, +25.0] | -20.0 | [-45.0, +0.0] | baseline |
| OPRO | +0.0 | [+0.0, +0.0] | +10.0 | [+0.0, +25.0] | -10.0 | [-25.0, +0.0] | no reliable gain on source (nothing to overfit) |
| EvoPrompt-lite | +0.0 | [+0.0, +0.0] | +10.0 | [+0.0, +25.0] | -10.0 | [-25.0, +0.0] | no reliable gain on source (nothing to overfit) |
| GEPA-lite | +0.0 | [+0.0, +0.0] | +10.0 | [+0.0, +25.0] | -10.0 | [-25.0, +0.0] | no reliable gain on source (nothing to overfit) |
| TextGrad-lite | +0.0 | [+0.0, +0.0] | +10.0 | [+0.0, +25.0] | -10.0 | [-25.0, +0.0] | no reliable gain on source (nothing to overfit) |
| Random-search | +0.0 | [+0.0, +0.0] | +10.0 | [+0.0, +25.0] | -10.0 | [-25.0, +0.0] | no reliable gain on source (nothing to overfit) |
| Oracle-OPRO | +10.0 | [+0.0, +30.0] | +5.0 | [-10.0, +20.0] | +5.0 | [-20.0, +30.0] | oracle (tuned on the new data; upper bound) |

Gains are in percentage points, paired by question and pooled over seeds; CIs resample questions. Transfer shortfall = gain on source − gain on new data; positive means the improvement did not carry over. OVERFITS requires a reliably positive source gain and a reliably positive transfer shortfall (both CIs above 0); a gain whose shortfall CI includes 0 is reported as inconclusive, not as overfitting.

## 3. Against the human-written prompt

| Method | Gain on source (pts) | 95% CI | Gain on new data (pts) | 95% CI | Transfer shortfall (pts) | 95% CI | Reading |
|---|---|---|---|---|---|---|---|
| Zero-shot | +10.0 | [+0.0, +30.0] | -10.0 | [-25.0, +0.0] | +20.0 | [+0.0, +45.0] | baseline |
| OPRO | +10.0 | [+0.0, +30.0] | +0.0 | [+0.0, +0.0] | +10.0 | [+0.0, +30.0] | no reliable gain on source (nothing to overfit) |
| EvoPrompt-lite | +10.0 | [+0.0, +30.0] | +0.0 | [+0.0, +0.0] | +10.0 | [+0.0, +30.0] | no reliable gain on source (nothing to overfit) |
| GEPA-lite | +10.0 | [+0.0, +30.0] | +0.0 | [+0.0, +0.0] | +10.0 | [+0.0, +30.0] | no reliable gain on source (nothing to overfit) |
| TextGrad-lite | +10.0 | [+0.0, +30.0] | +0.0 | [+0.0, +0.0] | +10.0 | [+0.0, +30.0] | no reliable gain on source (nothing to overfit) |
| Random-search | +10.0 | [+0.0, +30.0] | +0.0 | [+0.0, +0.0] | +10.0 | [+0.0, +30.0] | no reliable gain on source (nothing to overfit) |
| Oracle-OPRO | +20.0 | [+0.0, +50.0] | -5.0 | [-15.0, +0.0] | +25.0 | [+0.0, +55.0] | oracle (tuned on the new data; upper bound) |

## 4. Robustness across new datasets (Q2)

| Method | gsmplus | svamp | Worst dataset | Spread between datasets |
|---|---|---|---|---|
| Zero-shot | 0.70 ± 0.00 | 0.90 ± 0.00 | 0.70 | 0.20 |
| Human-written | 0.90 ± 0.00 | 0.90 ± 0.00 | 0.90 | 0.00 |
| OPRO | 0.90 ± 0.00 | 0.90 ± 0.00 | 0.90 | 0.00 |
| EvoPrompt-lite | 0.90 ± 0.00 | 0.90 ± 0.00 | 0.90 | 0.00 |
| GEPA-lite | 0.90 ± 0.00 | 0.90 ± 0.00 | 0.90 | 0.00 |
| TextGrad-lite | 0.90 ± 0.00 | 0.90 ± 0.00 | 0.90 | 0.00 |
| Random-search | 0.90 ± 0.00 | 0.90 ± 0.00 | 0.90 | 0.00 |
| Oracle-OPRO | 0.80 ± 0.00 | 0.90 ± 0.00 | 0.80 | 0.10 |

Accuracy per new dataset, mean ± sd over seeds. Spread = best minus worst dataset; a small spread means the prompt holds up evenly across datasets. With one new dataset there is no spread to measure: run with two or more (`OOD=svamp,gsmplus`).

## 6. Truncation audit (is the 512-token cap the confound?)

| Method | Answers | Truncated | Source | New data | Unparseable | Mean output tokens | Gap (all) | Gap (cut-off dropped) |
|---|---|---|---|---|---|---|---|---|
| Zero-shot | 30 | 3.3% | 0.0% | 5.0% | 6.7% | 160 | +0.100 | +0.050 |
| Human-written | 30 | 3.3% | 0.0% | 5.0% | 6.7% | 203 | -0.100 | -0.150 |
| OPRO | 30 | 0.0% | 0.0% | 0.0% | 3.3% | 173 | +0.000 | +0.000 |
| EvoPrompt-lite | 30 | 0.0% | 0.0% | 0.0% | 3.3% | 174 | +0.000 | +0.000 |
| GEPA-lite | 30 | 0.0% | 0.0% | 0.0% | 3.3% | 174 | +0.000 | +0.000 |
| TextGrad-lite | 30 | 0.0% | 0.0% | 0.0% | 3.3% | 174 | +0.000 | +0.000 |
| Random-search | 30 | 0.0% | 0.0% | 0.0% | 3.3% | 174 | +0.000 | +0.000 |
| Oracle-OPRO | 30 | 0.0% | 0.0% | 0.0% | 3.3% | 203 | +0.150 | +0.150 |

**Verdict: CONFOUNDED.** Cut-off answers are a real problem: up to 3.3% of answers were truncated (Zero-shot), and dropping them moves a gap by up to 0.050. Rerun with a larger cap (`MAX_TOKENS=1024 LOG_DIR=logs/after_1024 bash run_all.sh`) before drawing conclusions.

## Caveats

- Near ceiling: Zero-shot already scores 0.90 on the source and 0.80 on the new data, so a 3-point difference is only 1-2 questions out of a few dozen. Treat small gains as noise unless the CI excludes 0.
- Question-level CIs are wide with 10 source and 20 new-data questions per seed; a method listed as having no reliable gain may still have a small real one. Raise `N_PER_SPLIT` to tighten them.
- One model, one task family (math word problems), one source → target shift (GSM8K → SVAMP). The conclusions are about this setting.
- Seeds change the question sample and the optimizer's proposal sampling, so seed spread mixes data noise and search noise.
- GEPA-lite and TextGrad-lite are simplified re-implementations; DSPy-GEPA and DSPy-MIPROv2 are the real DSPy optimizers, restricted to instruction-only search through this harness.
