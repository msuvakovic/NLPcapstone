# Findings from the fresh Gemma4 pilot

Started October 4, 2026; completed October 5. The three planned implementations
are available and tested. **The pilot did not establish a reliable OOD accuracy
gain from source-example feedback, staged evaluation, or the confirmation guard.**

| Method before confirmation guard | Source accuracy | Mean OOD | Worst OOD | Difference from base, pp (paired 95% interval) |
| --- | ---: | ---: | ---: | --- |
| Base | 93.75% | 96.56% | 94.38% | -- |
| Human-written | 95.31% | 95.94% | 93.12% | -0.62 [-1.56, 0.00] |
| History-only control | 93.75% | 96.88% | 95.00% | +0.31 [0.00, +0.94] |
| Source-example feedback | 95.31% | 96.56% | 94.38% | 0.00 [-0.94, +0.94] |
| Feedback + staged evaluation | 93.75% | 96.56% | 94.38% | 0.00 [0.00, 0.00] |

The small history-control gain is **one additional correct target prediction**
out of 320, with no target regressions. A one-sided exact paired test gives
p=0.5. Its bootstrap interval includes zero and does not establish superiority.
The feedback variant fixes one target error and introduces one; the racing
variant matches every base final prediction. The human control has two target
regressions, including one invalid `Neutral` response that strict scoring counts
as incorrect. All other final prompts have zero invalid labels.

Every search winner improves the source selection score from 58/64 to 59/64,
but **none improves source confirmation**: all three match the base's correctness
on all 80 independent confirmation examples. The predeclared guard therefore
retains the base for every variant, yielding 96.56% mean OOD accuracy. This is
evidence that the selection gains did not replicate on confirmation; it is not
proof that every candidate overfits or that the methods are generally equivalent.

Staged evaluation screens **13 candidates instead of 7** under the same
512-evaluation limit. It fully evaluates three finalists, versus seven in each
full-evaluation arm. Search consumes 48,104 tokens versus 40,007 for the
history-only control (**20.24% more**) and 43,685 for feedback alone. Equal
evaluation/proposal ceilings are not equal token expenditures. All variants
include concise edits and an 80-word ceiling, so this experiment does not
separately identify the effect of shortening prompts.

The pilot made **3,549 physical requests**, with zero failed API attempts,
256 exact-cache hits, 320,665 returned input tokens and 8,177 output tokens.
The persistent 4,000-attempt cap includes retries and interrupted dispatches
across restarts. Tokens are not a billing estimate. Recorded search timings
are descriptive: cache reuse, provider load and sequential execution prevent
an isolated latency-speedup claim.

There are **592 distinct fresh examples**: 64 training, 64 selection, 80
confirmation, 64 final source, 160 Amazon and 160 tweets. They exclude the
1,686 normalized texts recovered from earlier local manifests and logs.
The five distinct final prompts share the same 384 final examples; they are
not five independent dataset replications. One optimizer seed, one partition,
and one model limit generalization. Existing target domains were known from
earlier work, even though these individual examples were newly held out.

**Recommendation:** keep the base prompt as the default. Retain the experimental
options and confirmation guard for future controlled studies. The new guard
correctly declined unsupported replacements; the measured accuracy benefit of
that guard in this run is zero. Further accuracy claims would require another
predeclared experiment with new confirmation evidence, not choosing a deployed
prompt using this final test.

All **90 offline checks passed**. The independent audit verified all three
searches, selection-stage scores, confirmation decisions, and 2,240 stored
confirmation/final prediction rows against saved real responses. It performed
4,560 cache-backed checks with **zero new model requests**. Every returned model
identifier was `gemma4:31b`; the service alias does not pin immutable weights.

See [full tables](report.md), [machine-readable metrics](summary.json),
[protocol](protocol.json), [execution totals](execution.json),
[offline verification](verification.json), and [artifact audit](artifact_audit.json).
