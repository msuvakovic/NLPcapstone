# NLP capstone: prompt optimization under domain shift

The project studies whether instructions optimized on source examples transfer to unseen domains. Project proposals and presentations are the PDFs in this directory. [log.txt](log.txt) contains the latest verified results followed by preserved historical output.

**Default sentiment prompt, confirmed October 5, 2026:**
`Classify the sentiment of the following text as Positive or Negative.`
Use this base instruction for sentiment inference. The experimental variants
remain available for explicitly requested research comparisons.

## Branch notes: `albert`

Updated October 13, 2026. This section describes the implementation and saved experiment evidence on `albert`. The work focuses on trustworthy source-to-target evaluation, testing newer prompt optimizers, and reducing inference overhead. Implementation status and measured benefits are distinguished below. Local response caches and optional dependency environments are not included in Git.

### Changes and implementations

The October 7 DSPy integration adds explicit `DSPyGEPA`, `DSPyMIPROv2`, and
`DSPyCOPRO` adapters with counted/capped search attempts, disjoint source
training/validation, matching search/final prompt formatting, and recorded
seeds/settings. COPRO is named correctly; MIPROv2 runs with no demonstrations
in the selected task prompt. These are optional instruction-only adaptations;
the base sentiment prompt remains the default. See the [DSPy setup and
limitations](prompt_opt_harness/prompt_opt_harness/README.md#dspy-adapters-october-7-2026).
Validation on October 7 passed 21 DSPy integration tests using the actual pinned
compilers with an offline backend, plus 44 existing harness/isolation/task/smoke
checks (65 total). No new Gemma performance result is claimed.

| Area | Implementation on this branch | Main files |
| --- | --- | --- |
| OPRO data isolation | Removed final-target feedback from the harness and OPRO ranking. Direct target inputs are rejected; duplicate IDs/text are checked before backend creation. New runs record `source-dev-only-v1`. | [Harness](prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/harness.py), [OPRO](prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/optimizers/opro.py) |
| Search safeguards | Reset candidate history, isolate randomness by seed, and enforce an OPRO search-attempt cap including retries. Reserve complete scores before proposing candidates; exclude partial scores. Disable the old oracle because it trained on final target tests. | [OPRO](prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/optimizers/opro.py), [budget guard](prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/budget.py), [oracle](prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/optimizers/oracle.py) |
| Harness reliability | Replace semantic prediction reuse and prefix grading with exact per-batch caching and complete-label parsing. Accept slow successes, log failed attempts, disable hidden SDK retries, and report search/final costs separately. | [Scoring](prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/optimizers/base.py), [backends](prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/llm_backends.py), [reliability tests](prompt_opt_harness/prompt_opt_harness/tests/test_harness_reliability.py) |
| Controlled benchmark | Add explicit source-data allowlists, disjoint frozen splits, strict sentiment-label parsing, exact request caching, logical evaluation budgets, retries, checkpointing and bounded concurrency. Versioned protocols reject incompatible resumes before client creation; search/final artifacts carry a protocol fingerprint. | [Transport/evaluator](tools/prompt_bench/core.py), [data preparation](tools/prompt_bench/data.py), [runner](tools/prompt_bench/run.py), [protocol guard](tools/prompt_bench/protocol.py) |
| Task and DRO consistency | Make classification schemas independent of optimized instructions. Use the same seeded, sorted-ID source subsets for every DRO candidate and record those subsets in results. | [Task schemas](prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/tasks.py), [DRO](prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/optimizers/dro.py), [invariance tests](prompt_opt_harness/prompt_opt_harness/tests/test_tasks_and_dro.py) |
| Optimizer experiments | Integrate official GEPA 0.1.4 and repository OPRO through the controlled evaluator. Add an ESPO-inspired diagnosis/proposal experiment, bootstrap candidate selection, and worst-source-group selection in a separate multi-source regime. These adaptations are not full ESPO or DRO-InstructZero reproductions. | [Methods](tools/prompt_bench/methods.py) |
| GEPA validation correctness | Use globally unique example IDs to prevent training/validation cache collisions. Independently check GEPA's per-example validation scores; invalidated exploratory runs are preserved separately. | [Methods](tools/prompt_bench/methods.py), [artifact verification](tools/prompt_bench/verify.py) |
| Overfitting analysis | Measure baseline-adjusted validation optimism and source-versus-target gains, with paired final-test intervals. Audit historical mock traces and check instructions for copied source-text spans. | [Analysis](tools/prompt_bench/analyze.py), [diagnostics](tools/prompt_bench/overfit.py), [historical audit](tools/prompt_bench/audit_old_logs.py) |
| Output-cap tuning | Test caps 1/2/4/8/16 on source training data, freeze candidate caps, then confirm them on separate source validation data. Record accuracy, macro F1, output validity, finish reasons, actual tokens and latency. | [Cap experiment](tools/prompt_bench/token_caps.py), [cap tests](tools/prompt_bench/test_token_caps.py) |
| Results and reproducibility | Preserve historical root-log entries, label simulated versus real-response results, and provide an offline replay that refuses new model requests. | [Root log](log.txt), [offline replay](tools/prompt_bench/replay_opro.py), [benchmark instructions](tools/prompt_bench/README.md) |

The working tree also contains toy NLI examples and an entropy-regularized OPRO experiment. These have not established a real MultiNLI result or an accuracy benefit in the controlled sentiment pilot. The entropy wrapper's whitespace perturbations should not be interpreted as controlled high-temperature sampling.

### Optimization-round experiment — October 13, 2026

The [round-count runner](tools/prompt_bench/round_sweep.py) compares **0, 1, 2,
4, 8 and 16 optimization rounds** across three search seeds. One round means
one proposed rewrite attempt, including rejected or duplicate proposals. Each
seed's earlier caps use checkpoints from the same 16-round trajectory. This
tests the custom source-feedback optimizer; it does not establish an optimal
round count for DSPy, GEPA or other algorithms.

All checkpoint prompts freeze before a separate 64-example source set selects
the smallest cap within 1 percentage point of the best mean accuracy. Final
source/Amazon/tweet tests audit the frozen choices and never select the cap.
The classifier stays at 16 output tokens so round count is the variable under
test. The request cap is 4,000 attempts including retries and restarts.

Fresh source samples come from SST-2's official training pool because previous
experiments largely exhausted unused validation examples. Samples contain at
least 12 words and exclude exact normalized texts and shared eight-word spans
from recoverable previous data and current splits. This does not guarantee
sentence-family or semantic independence. Earlier source percentages are not
directly comparable. The test is a small pilot on one partition, not a proof of
statistical equivalence or a universal stopping rule.

See [runner instructions](tools/prompt_bench/README.md#optimization-round-pilot--october-13-2026)
for execution and cache-only audit commands. The existing base instruction
remains the default.

**Completed result: extra rounds did not change the selected prompt.** All
18 checkpoint choices retained the base and scored 95.31% on confirmation.
The base already scored 32/32 on the source selection set, and ties retain it;
this ceiling makes the pilot inconclusive about a general optimal round count.
Zero rounds is the cheapest choice for this observed outcome.

| Round cap | Source confirmation accuracy | Mean search input + output tokens per seed |
| --- | ---: | ---: |
| 0 | 95.31% | 0 |
| 1 | 95.31% | 8,004 |
| 2 | 95.31% | 10,861 |
| 4 | 95.31% | 16,995 |
| 8 | 95.31% | 31,131 |
| 16 | 95.31% | 62,776 |

Final base accuracy was **100% source (32/32), 96.88% Amazon (62/64), 93.75%
tweets (60/64), and 95.31% mean OOD**, with zero invalid labels. Repeated seed
rows reuse the same instruction and predictions; they are not independent
replications. This run cannot establish whether rejected alternatives overfit.
The overfitting regression test separately verifies that simulated selection
gains can fail confirmation and leave the base selected.

The run used **1,776 requests, zero failed attempts, 160 cache hits and 195,740
returned tokens**. All **123 offline checks** passed, including 12 round-pilot
regressions. A complete cache replay reconstructed all three searches and every
held-out prediction without model requests. The standalone verifier serializes
SQLite replay reads; the original execution source remains hash-locked.
See [findings](output/round_sweep_2026-10-13/findings.md),
[report](output/round_sweep_2026-10-13/report.md),
[offline checks](output/round_sweep_2026-10-13/offline_tests.json), and
[cache audit](output/round_sweep_2026-10-13/verification.json).

### Fresh performance experiment — October 4, 2026

The [improvement pilot](tools/prompt_bench/improvement_pilot.py) adds three
controlled options: source-training examples in prompt proposals, staged
candidate evaluation, and an independent source confirmation guard. The
[search implementation](tools/prompt_bench/adaptive_search.py) compares an
identical history-based proposal loop with/without feedback, then adds a
16-to-32-example screening schedule. Full evaluation admits at most seven
candidates including the base; staged evaluation can screen thirteen under the
same 512-evaluation budget. These are custom component ablations, not new
reference OPRO/GEPA reproductions.

All proposal variants include concise rewrites and an 80-word limit. An
independent 80-example confirmation set can reject a proposed replacement and
retain the base, using a predeclared paired test with correction for the three
comparisons. Both raw and guarded choices are frozen before final scoring.
The dataset builder excludes normalized texts recovered from previous local
manifests and logs. The small pilot uses one seed, 64/64 source training/selection
examples, 80 confirmation examples, 64 final source examples, and 160 final
examples each from Amazon and tweets.

The request budget is **4,000 physical attempts across the entire experiment**,
including retries and restarts. A SQLite ledger reserves each attempt before
dispatch; exact-cache hits remain available when that cap is reached. The
classification cap stays at 16 tokens. Equal search evaluation limits do not
imply equal token costs, and small sample sizes may leave gains inconclusive.

**90 offline checks passed**, including 15 new checks for the staged schedule,
feedback isolation, confirmation decisions, artifact freezes, and persistent
request accounting. See the [runner instructions](tools/prompt_bench/README.md)
and [offline verification](output/improvement_pilot_2026-10-04/verification.json).
The completed real-model pilot used **3,549 requests, zero failed API attempts**,
and 328,842 returned input-plus-output tokens. The independent artifact audit
passed 4,560 cache-backed checks with zero new model calls.

| Fresh pilot method | Mean OOD accuracy | Change vs base | Search input + output tokens |
| --- | ---: | ---: | ---: |
| Base | 96.56% | -- | 0 |
| Human-written | 95.94% | -0.62 pp | 0 |
| History-only control | 96.88% | +0.31 pp | 40,007 |
| Source-example feedback | 96.56% | 0.00 pp | 43,685 |
| Feedback + staged evaluation | 96.56% | 0.00 pp | 48,104 |

**No reliable OOD accuracy improvement was established.** The history-only
control's gain is one additional correct target example out of 320; its paired
95% interval includes zero (0.00 to +0.94 pp), and a one-sided exact paired test
gives p=0.5. Feedback fixes one target error but introduces another. Racing's
final predictions match the base. All three challengers have identical
correctness to the base on the independent 80-example confirmation set, so
every guarded variant retains the base and scores 96.56% OOD.

Staged evaluation screened 13 candidates versus 7 within the same 512-evaluation
budget, but used 20.24% more search tokens than the history-only control. This
establishes broader screening, not an accuracy or cost improvement. The
human-written control returned one invalid `Neutral` answer on a tweet, scored
as incorrect; all other final prompts had zero invalid labels. Keep the base
as the default. These are fresh test samples, so their percentages are not
before/after comparisons with the historical 95.33% pilot.

Evidence: [findings](output/improvement_pilot_2026-10-04/findings.md),
[full results](output/improvement_pilot_2026-10-04/report.md),
[metrics and macro-F1](output/improvement_pilot_2026-10-04/summary.json), and
[artifact audit](output/improvement_pilot_2026-10-04/artifact_audit.json).

### Harness reliability fixes — October 4, 2026

The legacy harness now shares one scoring path for aggregate and per-example accuracy. Only identical complete prompts share predictions within a batch. Sentiment and binary NLI require complete labels; partial words, explanations and multiple-label responses count as invalid. This update introduced `scoring_protocol: exact-label-v1`; the later task-schema update below supersedes it for new runs.

Successful responses are accepted regardless of latency. Failed requests and retries count toward calls and request time, and SDK retries are disabled so they cannot bypass accounting. Results include `search_api_calls`, `evaluation_api_calls`, `failed_attempts`, `unknown_usage_attempts` and `backoff_time`, including a separate evaluation backend. Successful token counts remain local estimates; failed-request token usage is unknown, not measured as zero.

OPRO and its inherited wrappers reserve complete scoring batches and check the remaining allowance before every native backend attempt. A budget too small for the initial score fails before requests; retry exhaustion of the budget cannot add a partial candidate to selection. The cap applies to search, while final evaluation is reported separately. Other experimental optimizers still have their own budget behavior. The controlled benchmark retains its distinct-evaluation/proposal limits.

Offline verification passed **21 reliability tests, 10 isolation tests, 15 benchmark tests and 5 smoke checks**. The earlier malformed answer (`Positive or Negative`) now scores zero, a simulated 31-second success makes one request, and a four-example initial batch with budget one sends zero requests and raises an explicit budget error. A separate replay reproduced all three saved controlled OPRO searches and final predictions with **zero model requests**. These checks establish correctness and removal of unnecessary retries, not a new real-model accuracy or speed result. See the [validation report](output/harness_reliability_2026-10-04/report.md) and [replay evidence](output/harness_reliability_2026-10-04/replay/replay_summary.json).

### Protocol, DRO and task fixes — October 4, 2026

New benchmark runs freeze model/endpoint, budgets, decoding settings, methods, seeds, worker count, source-code hashes and the dataset fingerprint in a version-2 protocol. Changed settings, changed splits and artifacts belonging to another protocol are rejected before the model client is created. The historical September pilot remains readable and replayable; its unversioned protocol is deliberately not upgraded or resumed automatically. New API-enabled work needs a new output directory. The benchmark README contains updated commands.

DRO now compares every candidate on identical source subsets, reproducible across process hash seeds and input order. NLI and sentiment schemas are immutable `TaskSpec` values, passed through formatting, parsing, search, reflection and final scoring. Removing a task word from an optimized instruction cannot change the required output labels. New harness records use `exact-label-v2-explicit-task` and include the task specification and DRO subset IDs.

**70 offline checks passed:** 11 protocol tests, 8 task/DRO tests, 21 reliability tests, 10 isolation tests, 15 benchmark tests and 5 smoke checks. The historical artifact audit and three saved OPRO replays also passed with zero fresh model calls. This establishes correctness and compatibility, not an additional real-model accuracy improvement. See the [validation report](output/protocol_tasks_dro_2026-10-04/report.md).

This is a useful branch checkpoint once the intended source, tests, documentation and selected result artifacts are committed. `.env`, virtual environments, temporary files and new bytecode/SQLite caches are ignored. Previously tracked `.pyc` files still need deliberate exclusion or untracking before a clean commit. The fixes did not stage, commit or push anything; unfinished experiments remain labeled incomplete.

### Results established so far

The main sentiment pilot uses Gemma 4 31B, three optimizer seeds sharing one frozen partition, and 300 final examples each from SST-2, Amazon and binary tweet sentiment. In the single-source regime, Amazon and tweets are unseen during optimization.

| Experiment | Observed result | Interpretation |
| --- | --- | --- |
| Base prompt / corrected OPRO | **95.33% mean OOD accuracy**; all three OPRO seeds retain the base prompt | No optimization gain over the base was established. The October 4 fix was validated by replaying saved real responses, with zero fresh model calls. |
| Official GEPA | **93.67% mean OOD accuracy** | This tested configuration underperformed the base; installing a newer optimizer did not itself improve transfer. |
| GEPA + bootstrap selection | **92.22% mean OOD accuracy** | This selection adaptation worsened transfer in the pilot. |
| ESPO-inspired method | Search complete; final evaluation incomplete | No final accuracy conclusion yet. |
| Request concurrency | Four workers averaged **20.83s**, versus **61.52s** serially, for 24 requests | About **2.95x observed speedup** in a small timing experiment; provider caching and service load were uncontrolled. This was request concurrency, not GEPA parallel-proposal search. |
| Classification output cap | Caps 1/4/16 all scored **96.88% accuracy**, **0.9687 macro F1**, and zero invalid labels on 64 confirmation examples | Cap 1 used one output token instead of two: **50% fewer output tokens**, but only **1.54% fewer total tokens**. No latency improvement was demonstrated. These source-validation scores must not be compared directly with the OOD scores above. |

GEPA's average validation gain was 2.08 percentage points, while final-source accuracy changed by -0.11 points and OOD accuracy by -1.67 points. This is consistent with prompt overfitting in this pilot. Historical source/OOD gaps are not sufficient evidence on their own: the original harness leaked target feedback, and the audited September 21 traces reproduce simulated mock-backend behavior.

### Verification, limitations and remaining work

- **Verified checks:** 10 data-isolation tests, 15 benchmark tests and 5 smoke checks passed after the OPRO repair; 3 additional output-cap tests passed. Changing target labels/text leaves the search unchanged, while final scores respond to the changed labels. Three cached OPRO searches and their final predictions exactly matched the earlier clean benchmark.
- **Experiment status:** all 12 optimizer searches are complete, but only 5 of 8 distinct final prompts have complete saved predictions. ESPO-inspired evaluation must finish before a full historical benchmark report is possible. The new runner rejects automatic resumption of that unversioned experiment; completing it requires an explicitly reviewed migration or the original experiment implementation. No unfinished evaluation was resumed.
- **Cap recommendation scope:** cap 1 is supported only for the exact tested base-prompt Positive/Negative classifier with strict output validation. It reports a limit finish despite returning a complete label. Cap 4 is the lowest tested cap with natural stopping; cap 3 was not tested. Proposal/reflection caps and existing frozen benchmark defaults remain unchanged.
- **Research limits:** one model and one data partition do not establish general superiority. Identical prompts reuse cached predictions across seeds. The small cap confirmation does not establish OOD or rare-failure behavior. Billing savings were not measured.
- **Remaining implementation limits:** experimental optimizers outside the OPRO family still need their own budget audit; token accounting in legacy backends is estimated, and the entropy sampler still uses whitespace perturbations. A target-trained oracle needs separate target-development and final-test splits before reintroduction.

Detailed evidence: [OPRO replay](output/leakage_fix_2026-10-04/replay_summary.json), [fix validation](output/leakage_fix_2026-10-04/validation.json), [output-cap report](output/token_caps_2026-10-04/report.md), [cap artifact checks](output/token_caps_2026-10-04/verification.json), and [research notes](docs/performance_research_2026-09-29.md). The offline commands below verify the implementation without restarting the unfinished cloud evaluation.

## Current implementation

- [Project harness](prompt_opt_harness/prompt_opt_harness/README.md): backend wrappers, OPRO, baselines, experimental optimizers, evaluation and logging. Local GEPA and TextGrad classes are simplified implementations.
- [Controlled benchmark](tools/prompt_bench/README.md): official GEPA 0.1.4, repository OPRO with exact evaluation, an ESPO-inspired component experiment, bootstrap selection and a separate multi-source selection experiment.
- The benchmark uses Gemma 4 31B through Ollama, three search seeds, 64 source training and 64 validation examples, and 300 final examples per domain. It caps each search at 512 distinct candidate/example evaluations and 16 proposals, caches exact requests and uses bounded concurrency.

## Data-leakage fix — October 4, 2026

The affected optimizer is **OPRO**. The harness now supplies only source development examples to optimization; source and target test sets are evaluated after selection. OPRO rejects a non-`None` `ood_examples` argument, including in its scoring helper and inherited wrappers. Candidate scores, proposal history and final selection use source development feedback only. Each search starts with fresh history and its own seeded random generator.

Before creating a backend, the harness rejects duplicate IDs or case/whitespace-normalized text across or within its splits. New result records include `evaluation_protocol: source-dev-only-v1`. This check cannot detect semantic paraphrases or model pretraining contamination.

The old `OracleOPRO` path is disabled because it pooled final target tests into training. It needs separate target development and final test sets before it can be used again. The comparison scripts no longer advertise target-score-based progressive selection or an oracle upper bound.

## Verified results and status

An additional [output-cap experiment](output/token_caps_2026-10-04/report.md) found that **one output token** preserved all 64 confirmation predictions for the fixed base sentiment prompt, with zero invalid labels. Caps 1, 4 and 16 all scored 96.88% source-validation accuracy and 0.9687 macro F1. Cap 1 used one output token versus two, saving 50% of output tokens but only 1.54% of total input-plus-output tokens. Median latency did not improve. Cap 4 was the smallest tested cap with natural stopping (cap 3 was not tested); cap 1 reached the limit with a complete label. This setting is specific to the tested classifier, not prompt proposals or reflections. See [recommended settings](output/token_caps_2026-10-04/recommended_settings.json); original benchmark defaults remain unchanged.

The repaired OPRO was replayed against the September 29 saved real-model responses with network requests disabled. All three candidate pools, validation vectors, selected prompts and final predictions matched the earlier clean source-only benchmark. All three seeds selected the unchanged base prompt.

| Method | SST-2 source test | Amazon | Tweets | Unseen-domain mean |
| --- | ---: | ---: | ---: | ---: |
| Base / repaired OPRO | 95.67% | 95.00% | 95.67% | 95.33% |

These are 900 unique final examples, reused across identical selected prompts; they are not three independent dataset replications. This was a code-validation replay, not fresh inference or a before/after causal estimate of the leak's effect. Evidence: [replay summary](output/leakage_fix_2026-10-04/replay_summary.json).

Earlier completed single-source GEPA tests scored 93.67% OOD, or 92.22% with bootstrap selection, versus 95.33% for the base. ESPO-inspired final evaluation remains incomplete. All 12 searches finished, but only 5 of 8 distinct final prompts have complete saved predictions. The full benchmark report/plot stage requires the remaining results.

Separate timing tests observed about 2.95x speedup with four workers (24 requests: 61.52 seconds serial versus 20.83 seconds with four workers). This is a small connection-specific measurement with application caching disabled; provider caching and load were uncontrolled. It is not GEPA parallel-proposal search. No dollar-cost conclusion was established.

## Offline verification

Run from this repository root. The working interpreter on the development machine is `C:\Python313\python.exe`; the old `.venv` references an unavailable interpreter.

```powershell
& 'C:\Python313\python.exe' -X utf8 -B prompt_opt_harness/prompt_opt_harness/tests/test_data_isolation.py
& 'C:\Python313\python.exe' -X utf8 -B prompt_opt_harness/prompt_opt_harness/tests/test_harness_reliability.py
& 'C:\Python313\python.exe' -X utf8 -B prompt_opt_harness/prompt_opt_harness/tests/test_tasks_and_dro.py
& 'C:\Python313\python.exe' -X utf8 -B tools/prompt_bench/test_protocol.py
& 'C:\Python313\python.exe' -X utf8 -B tools/prompt_bench/test_benchmark.py
& 'C:\Python313\python.exe' -X utf8 -B tools/prompt_bench/verify.py --search-only
& 'C:\Python313\python.exe' -X utf8 -B tools/prompt_bench/replay_opro.py --output output/harness_reliability_2026-10-04/replay
```

The replay requires the local `requests.sqlite` cache and saved split/results files. It enforces zero physical model requests and fails on missing responses. See the benchmark README for dependencies and API-enabled resume commands.

Regression checks cover target-label/text changes, final evaluation ordering, direct OPRO and wrapper calls, duplicate splits, stale history, random seeding, the disabled oracle, strict labels, exact caching, retries, accounting and hard OPRO search budgets. They protect the known paths; future changes must continue running these checks. The controlled benchmark independently enforces exact evaluation and hard batch reservation for research results.
