# Optimization rounds: findings and limits

**Keep the base prompt. Zero optimization rounds is the cheapest choice in this
pilot, but the experiment does not identify a general optimal round count.**

All 0/1/2/4/8/16-round checkpoints across three search seeds selected exactly
the same base instruction. The reason is a selection ceiling: the base already
scored **32/32**, and the predeclared rule retains it on ties. Candidate scores
ranged from 30/32 to 32/32. This experiment therefore cannot tell whether 4, 8
or 16 rounds would be enough on a larger selection set where the base has errors.

All checkpoint choices scored **61/64 (95.31%)** on separate source confirmation
data. The frozen final prompt scored **32/32 source, 62/64 Amazon and 60/64 tweets**:
95.31% mean OOD accuracy, with no invalid responses. Equal results here reflect
identical selected programs and shared exact predictions; they are not three
independent replications or a statistical noninferiority result.

| Round cap | Confirmation accuracy | Mean logical search tokens per seed |
| --- | ---: | ---: |
| 0 | 95.31% | 0 |
| 1 | 95.31% | 8,004 |
| 2 | 95.31% | 10,861 |
| 4 | 95.31% | 16,995 |
| 8 | 95.31% | 31,131 |
| 16 | 95.31% | 62,776 |

Zero rounds requires no optimization. Its research accuracy measurements still
cost requests. Nonzero search costs include the baseline training diagnostic,
selection evaluations and proposals; they are charged per seed even when exact
responses are shared. Sixteen-round searches used 64,472 / 61,725 / 62,132 tokens
for seeds 0 / 1 / 2. There were 48 rewrite attempts, including two duplicates.
The task output cap stayed at 16; proposal output cap was 768.

The whole pilot used **1,776 actual requests**, zero failed attempts and 160
cache hits: **189,707 input + 6,033 output = 195,740 returned tokens**. It completed
in about 25 minutes, but this was not a controlled latency comparison. The
4,000-attempt limit covers restarts and retries. Billing was not measured.

The 288 fresh examples exclude 2,278 previously exposed normalized texts plus
shared eight-word spans. Because earlier pilots largely exhausted SST-2 official
validation, source splits come from its training pool with at least 12 words
per example. Phrase-family relationships and model pretraining contamination
are not ruled out. Do not compare these percentages directly with older pilots.

The runner froze all source-search checkpoints before confirmation, then froze
the chosen cap and final arms before target evaluation. All 123 offline checks
passed, including a synthetic overfitting case that improves source selection
but fails confirmation. The real pilot selected no replacement, so it cannot
establish how rejected alternatives generalize or whether they overfit.

The [offline verifier](../../tools/prompt_bench/verify_round_sweep.py) reconstructed
all three searches, every saved held-out prediction and the selection decision
from exact cached responses with **zero new model requests**. It serializes
replay reads to avoid the observed shared SQLite connection issue in the frozen
runner's embedded audit. Live cache reads were already locked and unaffected.
The original execution code/protocol fingerprints were preserved.

For a more informative future study, freeze a larger source selection set and
repeat on independent partitions before using final target data. This pilot
supports retaining the existing base; it does not support a universal 4-round,
8-round or 16-round setting, nor a conclusion about DSPy/GEPA iteration counts.

Evidence: [full report](report.md), [protocol](protocol.json),
[raw summary](summary.json), [request accounting](execution.json),
[offline tests](offline_tests.json), [cache audit](verification.json).
