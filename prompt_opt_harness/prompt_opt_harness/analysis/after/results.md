# Results: when do optimized prompts overfit?

Model `qwen2.5:7b`, max_tokens 512, seeds [42, 43, 44], 50 source-test and 100 new-dataset questions per seed. Optimizers were tuned on the source dev sample only (the oracle excepted).

## Findings (auto-generated from the logs)

- **Token cap:** Some answers were cut off (up to 3.8%, TextGrad-lite), but dropping them moves no gap by more than 0.015. Probably not driving the results; confirm with a rerun at a larger cap.
- **Versus Zero-shot (95% bootstrap CIs over questions):** no reliable gain over Zero-shot even on the source: OPRO, EvoPrompt-lite, GEPA-lite, TextGrad-lite, Random-search.
- **Versus the human-written prompt:** No optimizer reliably beats the human-written prompt on the new dataset.
- **Dev optimism:** the largest dev − test gap is EvoPrompt-lite (+0.07); a large value means the search picked a prompt that looked better on the dev sample than it really is.
- **Most robust:** GEPA-lite has the best worst-case accuracy over the new datasets (0.81).
- **Headroom:** tuning on the new dataset (oracle) reaches 0.88 on it, versus 0.87 for the best source-tuned optimizer (Random-search): 1.3 points left on the table by not seeing the target domain.
- **Cost:** best gain per API call (source test vs Zero-shot) is Random-search (+0.04 points per 1000 calls).

## 1. Headline metrics (mean ± sd over seeds)

| Method | Seeds | Dev | Source test | Dev − test | New data | Source − new gap | Worst case | Variance | API calls | Tokens | Gain / 1000 calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Zero-shot | 3 | 0.85 ± 0.04 | 0.87 ± 0.03 | -0.03 ± 0.05 | 0.84 ± 0.04 | +0.03 ± 0.07 | 0.77 ± 0.09 | 0.0080 | 200 | 55k | n/a |
| Human-written | 3 | 0.88 ± 0.04 | 0.87 ± 0.05 | +0.01 ± 0.04 | 0.88 ± 0.03 | -0.00 ± 0.07 | 0.81 ± 0.06 | 0.0058 | 200 | 70k | +0.00 |
| OPRO | 3 | 0.94 ± 0.02 | 0.89 ± 0.03 | +0.05 ± 0.03 | 0.86 ± 0.05 | +0.03 ± 0.07 | 0.78 ± 0.07 | 0.0069 | 506 | 186k | +0.03 |
| EvoPrompt-lite | 3 | 0.93 ± 0.04 | 0.86 ± 0.00 | +0.07 ± 0.04 | 0.86 ± 0.03 | -0.00 ± 0.03 | 0.79 ± 0.08 | 0.0076 | 506 | 188k | -0.03 |
| GEPA-lite | 3 | 0.94 ± 0.02 | 0.87 ± 0.03 | +0.07 ± 0.04 | 0.86 ± 0.05 | +0.01 ± 0.06 | 0.81 ± 0.06 | 0.0035 | 513 | 184k | +0.00 |
| TextGrad-lite | 3 | 0.93 ± 0.03 | 0.87 ± 0.01 | +0.05 ± 0.03 | 0.86 ± 0.05 | +0.01 ± 0.05 | 0.78 ± 0.07 | 0.0069 | 512 | 217k | +0.00 |
| Random-search | 3 | 0.93 ± 0.02 | 0.89 ± 0.04 | +0.04 ± 0.05 | 0.87 ± 0.04 | +0.03 ± 0.07 | 0.78 ± 0.07 | 0.0083 | 510 | 187k | +0.04 |
| Oracle-OPRO | 3 | 0.89 ± 0.03 | 0.88 ± 0.00 | +0.01 ± 0.03 | 0.88 ± 0.03 | +0.00 ± 0.03 | 0.81 ± 0.05 | 0.0047 | 452 | 143k | +0.01 |

Dev − test and Source − new gap are signed: positive means accuracy fell. Worst case is the lowest accuracy over the new datasets. Gain / 1000 calls is the source-test gain over Zero-shot per 1000 API calls (all calls, search included).

## 2. Do the gains transfer? Paired comparison with Zero-shot

| Method | Gain on source (pts) | 95% CI | Gain on new data (pts) | 95% CI | Transfer shortfall (pts) | 95% CI | Reading |
|---|---|---|---|---|---|---|---|
| Human-written | +0.0 | [-4.0, +4.0] | +3.3 | [+0.3, +6.7] | -3.3 | [-9.0, +1.7] | baseline |
| OPRO | +1.3 | [-3.3, +6.0] | +1.7 | [-1.7, +5.0] | -0.3 | [-6.0, +5.3] | no reliable gain on source (nothing to overfit) |
| EvoPrompt-lite | -1.3 | [-5.3, +2.7] | +2.0 | [-1.3, +5.7] | -3.3 | [-8.3, +2.0] | no reliable gain on source (nothing to overfit) |
| GEPA-lite | +0.0 | [-4.7, +4.7] | +2.0 | [-1.3, +5.7] | -2.0 | [-8.0, +3.7] | no reliable gain on source (nothing to overfit) |
| TextGrad-lite | +0.0 | [-5.3, +5.3] | +1.7 | [-2.3, +5.3] | -1.7 | [-8.3, +5.0] | no reliable gain on source (nothing to overfit) |
| Random-search | +2.0 | [-2.7, +6.7] | +2.3 | [-1.7, +6.0] | -0.3 | [-6.3, +5.3] | no reliable gain on source (nothing to overfit) |
| Oracle-OPRO | +0.7 | [-3.3, +5.3] | +3.7 | [+0.7, +7.0] | -3.0 | [-8.7, +2.0] | oracle (tuned on the new data; upper bound) |

Gains are in percentage points, paired by question and pooled over seeds; CIs resample questions. Transfer shortfall = gain on source − gain on new data; positive means the improvement did not carry over. OVERFITS requires a reliably positive source gain and a reliably positive transfer shortfall (both CIs above 0); a gain whose shortfall CI includes 0 is reported as inconclusive, not as overfitting.

## 3. Against the human-written prompt

| Method | Gain on source (pts) | 95% CI | Gain on new data (pts) | 95% CI | Transfer shortfall (pts) | 95% CI | Reading |
|---|---|---|---|---|---|---|---|
| Zero-shot | +0.0 | [-4.0, +4.0] | -3.3 | [-6.7, +0.0] | +3.3 | [-1.7, +8.7] | baseline |
| OPRO | +1.3 | [-2.7, +5.3] | -1.7 | [-5.0, +1.7] | +3.0 | [-2.3, +8.7] | no reliable gain on source (nothing to overfit) |
| EvoPrompt-lite | -1.3 | [-5.3, +2.0] | -1.3 | [-4.3, +1.7] | +0.0 | [-4.7, +5.0] | no reliable gain on source (nothing to overfit) |
| GEPA-lite | +0.0 | [-5.3, +5.3] | -1.3 | [-4.3, +1.7] | +1.3 | [-4.7, +7.3] | no reliable gain on source (nothing to overfit) |
| TextGrad-lite | +0.0 | [-4.7, +5.3] | -1.7 | [-4.7, +1.3] | +1.7 | [-4.3, +7.7] | no reliable gain on source (nothing to overfit) |
| Random-search | +2.0 | [-2.0, +6.7] | -1.0 | [-4.0, +2.0] | +3.0 | [-2.7, +8.7] | no reliable gain on source (nothing to overfit) |
| Oracle-OPRO | +0.7 | [-4.0, +4.7] | +0.3 | [-2.7, +3.7] | +0.3 | [-5.0, +5.7] | oracle (tuned on the new data; upper bound) |

## 4. Robustness across new datasets (Q2)

| Method | gsmplus | svamp | Worst dataset | Spread between datasets |
|---|---|---|---|---|
| Zero-shot | 0.79 ± 0.12 | 0.90 ± 0.05 | 0.79 | 0.11 |
| Human-written | 0.81 ± 0.06 | 0.95 ± 0.01 | 0.81 | 0.14 |
| OPRO | 0.78 ± 0.07 | 0.94 ± 0.02 | 0.78 | 0.16 |
| EvoPrompt-lite | 0.79 ± 0.08 | 0.94 ± 0.02 | 0.79 | 0.15 |
| GEPA-lite | 0.81 ± 0.06 | 0.92 ± 0.03 | 0.81 | 0.11 |
| TextGrad-lite | 0.78 ± 0.07 | 0.94 ± 0.02 | 0.78 | 0.16 |
| Random-search | 0.78 ± 0.07 | 0.95 ± 0.01 | 0.78 | 0.17 |
| Oracle-OPRO | 0.81 ± 0.05 | 0.95 ± 0.01 | 0.81 | 0.13 |

Accuracy per new dataset, mean ± sd over seeds. Spread = best minus worst dataset; a small spread means the prompt holds up evenly across datasets. With one new dataset there is no spread to measure: run with two or more (`OOD=svamp,gsmplus`).

## 6. Truncation audit (is the 512-token cap the confound?)

| Method | Answers | Truncated | Source | New data | Unparseable | Mean output tokens | Gap (all) | Gap (cut-off dropped) |
|---|---|---|---|---|---|---|---|---|
| Zero-shot | 450 | 1.1% | 0.0% | 1.7% | 4.7% | 163 | +0.030 | +0.015 |
| Human-written | 450 | 1.8% | 0.7% | 2.3% | 2.2% | 208 | -0.003 | -0.015 |
| OPRO | 450 | 1.3% | 1.3% | 1.3% | 3.8% | 183 | +0.027 | +0.028 |
| EvoPrompt-lite | 450 | 1.8% | 2.0% | 1.7% | 3.3% | 194 | -0.003 | +0.001 |
| GEPA-lite | 450 | 0.9% | 0.0% | 1.3% | 3.1% | 180 | +0.010 | +0.002 |
| TextGrad-lite | 450 | 3.8% | 3.3% | 4.0% | 5.6% | 226 | +0.013 | +0.003 |
| Random-search | 450 | 1.6% | 1.3% | 1.7% | 3.8% | 175 | +0.027 | +0.025 |
| Oracle-OPRO | 450 | 0.9% | 0.7% | 1.0% | 2.9% | 203 | +0.000 | -0.003 |

**Verdict: CHECK.** Some answers were cut off (up to 3.8%, TextGrad-lite), but dropping them moves no gap by more than 0.015. Probably not driving the results; confirm with a rerun at a larger cap.

## Caveats

- Question-level CIs are wide with 50 source and 100 new-data questions per seed; a method listed as having no reliable gain may still have a small real one. Raise `N_PER_SPLIT` to tighten them.
- One model, one task family (math word problems), one source → target shift (GSM8K → SVAMP). The conclusions are about this setting.
- Seeds change the question sample and the optimizer's proposal sampling, so seed spread mixes data noise and search noise.
- GEPA-lite and TextGrad-lite are simplified re-implementations; DSPy-GEPA and DSPy-MIPROv2 are the real DSPy optimizers, restricted to instruction-only search through this harness.
