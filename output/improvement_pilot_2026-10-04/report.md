# Fresh Gemma4 improvement pilot

One optimizer seed; three custom history-based search variants. These are component ablations, not a new reference OPRO or GEPA reproduction. All arms use 512 candidate/example evaluations and at most 16 proposals. All pay for the same 64-example baseline training diagnostic.

Training/selection/confirmation/source-test sizes: 64/64/80/64 SST-2 examples; 160 fresh Amazon and 160 fresh tweet examples. Exact normalized texts recovered from previous manifests and logs (1686 unique texts) were excluded. This cannot exclude unlogged human exposure, semantic paraphrases, or pretraining contamination.

History-full and feedback-full differ only in the source-training examples included in proposals. Feedback-racing screens 12 new candidates on 16 training examples, expands four to 32, then compares two survivors plus the base on the same 64 selection examples. All proposal variants include concise/simplification edits and an 80-word limit; brevity is not a separate ablation.

All raw winners were frozen before source confirmation. The optional guard compares each winner with the base using a one-sided exact paired test, Bonferroni alpha 0.05/3 and at least 1 pp gain. A rejection means insufficient confirmation, not proof of equal quality. All deployed and raw prompts were frozen before final evaluation; the same saved predictions serve identical prompts.

| Method | Source | Amazon | Tweets | Mean OOD | Worst OOD | OOD delta, pp (95% CI) | Input tokens/example |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| base | 93.75% | 98.75% | 94.38% | 96.56% | 94.38% | +0.00 [+0.00, +0.00] | 87.9 |
| human | 95.31% | 98.75% | 93.12% | 95.94% | 93.12% | -0.62 [-1.56, +0.00] | 101.9 |
| history_full | 93.75% | 98.75% | 95.00% | 96.88% | 95.00% | +0.31 [+0.00, +0.94] | 91.9 |
| history_full+guard | 93.75% | 98.75% | 94.38% | 96.56% | 94.38% | +0.00 [+0.00, +0.00] | 87.9 |
| feedback_full | 95.31% | 98.75% | 94.38% | 96.56% | 94.38% | +0.00 [-0.94, +0.94] | 111.9 |
| feedback_full+guard | 93.75% | 98.75% | 94.38% | 96.56% | 94.38% | +0.00 [+0.00, +0.00] | 87.9 |
| feedback_racing | 93.75% | 98.75% | 94.38% | 96.56% | 94.38% | +0.00 [+0.00, +0.00] | 91.9 |
| feedback_racing+guard | 93.75% | 98.75% | 94.38% | 96.56% | 94.38% | +0.00 [+0.00, +0.00] | 87.9 |

| Search | Candidates screened | Full selection scores | Evaluations | Proposals | Search tokens | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| history_full | 7 | 7 | 512 | 7 | 40007 | 373.4 |
| feedback_full | 7 | 7 | 512 | 7 | 43685 | 274.8 |
| feedback_racing | 13 | 3 | 512 | 15 | 48104 | 345.2 |

## Confirmation and interpretation

- history_full: 0 fixes, 0 regressions; p=1.0000; base retained.
- feedback_full: 0 fixes, 0 regressions; p=1.0000; base retained.
- feedback_racing: 0 fixes, 0 regressions; p=1.0000; base retained.

Macro-F1 and invalid counts are in summary.json. Paired final-test intervals resample within domain/label groups; they describe sample uncertainty conditional on this one seed. They are exploratory and not adjusted for all final comparisons. Repeated seeds, independent data partitions and another model are needed for broad claims. A smaller source/OOD gap alone is not a performance gain.

Tokens are reported, not equated: matching evaluation/proposal caps is not matching token cost. Timing includes cache reuse, provider load and execution order, so it does not establish a controlled latency speedup. Output caps stay at 16 in every arm.
