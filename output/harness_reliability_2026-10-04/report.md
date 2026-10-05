# Harness reliability fixes — October 4, 2026

Implemented on branch `albert`, including uncommitted changes. This is an offline correctness and compatibility check, not a new model-performance benchmark. No fresh model requests were made.

## Changes and demonstrated behavior

| Case | Before | After |
| --- | --- | --- |
| Malformed answer `Positive or Negative` on positive examples | Prefix grading accepted it (100% in the artificial reproduction). | Invalid; scores 0%. Both accuracy helpers use the same parser. |
| Similar texts with different sentiment | Optional embedding similarity could reuse another text's prediction. | Separate requests; only identical full prompts within a batch share a prediction. No embedding model is loaded for scoring. |
| Successful request taking 31 seconds, default retries | The simulated reproduction made 3 attempts but logged 1 and 31 seconds. | One attempt, accepted immediately after success, with all 31 seconds logged. |
| Timeout followed by success | Failed attempt absent from call/time statistics. | Both attempts and their durations recorded; retry waiting time recorded separately. Failed token usage explicitly unknown. |
| Four distinct source examples, OPRO budget 1 | Initial scoring made 4 requests. | Explicit `CallBudgetExceeded` before any request. |
| Retry exhausts budget during a candidate evaluation | Retry attempts were invisible to the budget. | No further attempts beyond the cap; incomplete candidate excluded from history/selection. |
| Separate evaluation backend or previously used backend | Evaluation-backend calls were omitted; prior calls could contaminate totals. | Search and final evaluation deltas reported separately; only this run contributes to totals. |

Exact label parsing supports Positive/Negative and Entailment/Contradiction, case-insensitive, optional matching quotes and one optional terminal period/exclamation mark. Multiple labels, prefixes and explanations are rejected. The entropy wrapper also uses this parser; its sampling approximation was not replaced.

Native backend SDK retries are disabled so the harness owns retries and accounting. OPRO reserves complete candidate scores, including the extra requests in SAPO/entropy wrappers, and enforces its search cap at each native backend attempt. Final scoring takes place after search on the restored original backend and is reported separately. If retries prevent even the initial score from finishing, the search fails explicitly.

## Verification

- **21 reliability tests passed**: exact caching, no embedding imports, malformed and NLI labels, slow success, failures/retries, SDK retry settings, initial and subsequent budget boundaries, partial scores, repeated searches, inherited wrappers, separate evaluation backends and reused-backend accounting. [Test source](../../prompt_opt_harness/prompt_opt_harness/tests/test_harness_reliability.py), [output](reliability_tests.txt).
- **10 data-isolation tests passed** after the accounting changes. [Output](isolation_tests.txt).
- **15 controlled benchmark tests passed**, including source-data guards, separate logical-budget accounting and the official GEPA adapter.
- **5 smoke checks passed**, including the tightened budget assertion. [Output](smoke_tests.txt).
- **Three saved OPRO searches and their final predictions replayed exactly**: candidate pools, validation vectors, selected prompts, evaluation/proposal counts and all final predictions matched the earlier controlled benchmark. **0 physical model calls; 4,268 exact cache hits.** [Replay summary](replay/replay_summary.json).

All three replayed searches retain the base prompt. Preserved scores are SST-2 95.67%, Amazon 95.00%, tweets 95.67%, mean unseen-domain accuracy 95.33%. These are the same 900 unique final examples reused across identical prompts, not fresh inference or three independent dataset replications. The controlled benchmark retains its original distinct-evaluation/proposal protocol and delegates reservation to its own evaluator.

## Limits

The fixes establish accurate label scoring, removal of redundant slow-success retries and enforcement of OPRO budgets. They do not demonstrate a new real-model accuracy gain or end-to-end speedup. Correct exact scoring can require more requests than approximate semantic reuse. Other experimental optimizer classes still need separate budget review. Custom backends must count each attempt, and custom scoring wrappers must declare extra requests through `_evaluation_calls()`.

Successful legacy token counts remain local estimates rather than provider/billing usage. Failed-request token usage is unknown; aggregate token totals exclude it. `wall_time` is summed request duration, while retry waits are recorded in `backoff_time`; neither alone is total experiment elapsed time. Historical artifacts and measurements were preserved.
