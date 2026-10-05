# Protocol, DRO and task-schema verification — October 4, 2026

Implemented on `albert`, including the current uncommitted working tree. No model requests, commits or pushes were made.

## Changes

1. **Versioned experiment configuration.** Prepare/search creates a version-2 protocol using exclusive file creation. All runner stages validate the model, endpoint, budgets, decoding settings, methods, seeds, worker count, source-code hashes and split fingerprint before constructing a model client. Existing search/final artifacts must match its protocol fingerprint, including files skipped during resumption. Saved splits are checked for changed content and overlapping IDs/text. Timing runs validate the same execution context. Workflow preparation validates the protocol before later steps.
2. **Comparable DRO scores.** Every candidate is evaluated on identical source subsets selected by the configured seed from sorted example IDs. Subsets do not depend on instruction wording, input order or process hash seed. Their IDs are included in result metadata. Invalid sampling parameters and duplicate IDs fail before requests. This is still a source-subset robustness adaptation, not domain-group DRO or a measured improvement in OOD accuracy.
3. **Explicit task schemas.** Immutable `TaskSpec` values fix label choices independently of instructions. The dataset specifies sentiment or binary NLI; the harness checks labels before creating a backend. Scoring, reflection, entropy sampling and final evaluation use that task. Wrong-task outputs are invalid. New results record `task_spec` and `exact-label-v2-explicit-task`.

Direct NLI callers must pass `task=BINARY_NLI`; the NLI dataset loader already does so for harness runs. Sentiment callers retain their default. A custom task can declare additional single-word labels, but no new real dataset or model-performance claim was added.

## Verification

| Offline suite | Passed | Evidence |
| --- | ---: | --- |
| Protocol validation | 11 | [Output](protocol.txt) |
| Task and DRO invariance | 8 | [Output](tasks_and_dro.txt) |
| Previous reliability regressions | 21 | [Output](reliability.txt) |
| Data isolation | 10 | [Output](isolation.txt) |
| Controlled benchmark | 15 | [Output](benchmark.txt) |
| Smoke checks | 5 | [Output](smoke.txt) |
| **Total** | **70** | |

The protocol suite includes a complete fake-client search, final evaluation and matching resume. Its printed request counters represent simulated requests. Mismatches are checked before client creation for prepare, search, evaluate and timing stages. Source subsets were compared in separate Python processes with different `PYTHONHASHSEED` values. NLI tests cover zero-shot, OPRO, local GEPA/TextGrad, reflection, DRO and entropy wrappers, including instructions without the word “entailment.”

The historical artifact audit still passes for 1,156 distinct examples and 12 completed capped searches. All three saved controlled OPRO searches and final predictions replay exactly: zero physical calls, 4,268 exact cache hits. See the [audit output](historical_audit.txt) and [replay summary](replay/replay_summary.json). Preserved mean OOD accuracy remains 95.33%; this is a compatibility replay of earlier responses, not a new accuracy experiment.

## Compatibility and branch checkpoint

Historical unversioned experiments are preserved for read-only verification and replay. The new execution path rejects their automatic resumption; use a new output directory for new model runs. Completing the unfinished historical ESPO evaluation requires an explicitly reviewed migration or the original experiment implementation. Cloud aliases are not immutable model revisions, so a configuration fingerprint cannot prevent a provider from changing the model behind the same name.

This is a useful harness checkpoint for the `albert` branch, with incomplete research results labeled as such. Source, tests, documentation and selected evidence should form a reviewed commit. `.gitignore` now excludes credentials, virtual environments, temporary files, new bytecode and output SQLite caches. Already tracked `.pyc` files still require deliberate exclusion or untracking; ignore rules do not remove them from Git. The index was left untouched, and nothing was committed or pushed. This assessment is not a full secret or release audit.
