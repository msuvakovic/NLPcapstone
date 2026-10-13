# Gemma4 optimization-round pilot

**Smallest cap within 1 percentage point of the best mean source confirmation score: 0 rounds.** This is an exploratory choice, not proof of equivalence or a universal optimum. The base prompt remains the default.

Three search seeds; caps 0, 1, 2, 4, 8, 16 are nested prefixes of the same trajectory within each seed. One round means one rewrite attempt, even when duplicate, empty, too long or truncated. This custom source-feedback search is not a DSPy/GEPA round benchmark.

Every new candidate is scored on the same 32 source selection examples. Proposals receive baseline feedback from 32 training examples. Base wins selection ties, then shorter instructions. All 18 checkpoint choices freeze before 64 separate source examples select the round cap. Final tests (32 source, 64 Amazon, 64 tweets) only audit frozen base, chosen cap and 16-round prompts. They never choose the cap or rewrite a prompt.

SST-2 official validation was largely exhausted by prior runs. This experiment samples the official training pool with at least 12 words per example; it may contain phrases. Source percentages are not comparable to earlier validation-based pilots. All splits are balanced and exclude normalized texts and shared contiguous eight-word spans recovered from earlier local logs/manifests and from other current splits. This cannot guarantee sentence-family separation, semantic independence or absence from model pretraining.

Classification is fixed at 16 output tokens, temperature 0, seed 12345; proposals use 768 tokens, temperature 0.7, seed 1000*search_seed+round. Search-only cost is charged logically for every seed, regardless of exact-cache sharing. Round 0 needs no search; baseline measurements are evaluation overhead. Worst-case planned requests: 3,792; persistent hard cap: 4,000 including retries and interrupted attempts.

| Rounds | Search selection accuracy | Confirmation accuracy (mean of seeds) | Delta vs base, pp (95% CI) | Mean search evaluations | Mean search tokens |
| --- | ---: | ---: | --- | ---: | ---: |
| 0 | 100.00% | 95.31% | +0.00 [+0.00, +0.00] | 0.0 | 0 |
| 1 | 100.00% | 95.31% | +0.00 [+0.00, +0.00] | 96.0 | 8004 |
| 2 | 100.00% | 95.31% | +0.00 [+0.00, +0.00] | 128.0 | 10861 |
| 4 | 100.00% | 95.31% | +0.00 [+0.00, +0.00] | 181.3 | 16995 |
| 8 | 100.00% | 95.31% | +0.00 [+0.00, +0.00] | 298.7 | 31131 |
| 16 | 100.00% | 95.31% | +0.00 [+0.00, +0.00] | 554.7 | 62776 |

Confirmation is development data used to choose the cap, not an unbiased final performance estimate. Intervals resample examples within labels with all seeds kept paired. They are descriptive, unadjusted for multiple comparisons, and may be degenerate when predictions match. Three seeds share one partition; small gains or rare failures remain unresolved. A 1 pp selection tolerance does not establish statistical noninferiority.

| Frozen final arm | Source | Amazon | Tweets | Mean OOD | OOD delta, pp (95% CI) | Invalid |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| base | 100.00% | 96.88% | 93.75% | 95.31% | +0.00 [+0.00, +0.00] | 0 |
| seed_0_rounds_0 | 100.00% | 96.88% | 93.75% | 95.31% | +0.00 [+0.00, +0.00] | 0 |
| seed_0_rounds_16 | 100.00% | 96.88% | 93.75% | 95.31% | +0.00 [+0.00, +0.00] | 0 |
| seed_1_rounds_0 | 100.00% | 96.88% | 93.75% | 95.31% | +0.00 [+0.00, +0.00] | 0 |
| seed_1_rounds_16 | 100.00% | 96.88% | 93.75% | 95.31% | +0.00 [+0.00, +0.00] | 0 |
| seed_2_rounds_0 | 100.00% | 96.88% | 93.75% | 95.31% | +0.00 [+0.00, +0.00] | 0 |
| seed_2_rounds_16 | 100.00% | 96.88% | 93.75% | 95.31% | +0.00 [+0.00, +0.00] | 0 |

Identical selected prompts reuse predictions; repeated rows are not independent replications. Macro F1, per-seed scores, raw predictions, rejected attempts and costs are saved in JSON. Timing/cache/provider load do not establish latency or billed savings. The cloud alias cannot guarantee immutable model weights.
