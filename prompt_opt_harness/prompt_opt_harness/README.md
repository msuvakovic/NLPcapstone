# Prompt optimizer harness

Implements the harness from the capstone proposal: dataset loader, optimizer
wrapper, OOD evaluator, logger/reporter.

## Current data-isolation contract (October 4, 2026)

`run_experiment` validates disjoint IDs and normalized texts before backend creation, then passes only `source.dev` to the optimizer. Final source and OOD sets are scored only after prompt selection. OPRO rejects any non-`None` `ood_examples` argument and ranks exclusively on source development scores. New run JSON includes `evaluation_protocol: source-dev-only-v1`. Its inherited DRO, regularized, SAPO and entropy wrappers share that target-input guard.

`OracleOPRO` is disabled until separate target-development and final-test splits exist. `compare_runs.py` and `compare_nli.py` are now source-only mock pipeline checks; their historical target-score-based comparisons should not be interpreted as untouched OOD results.

Run `python -B tests/test_data_isolation.py` to check the boundaries without model requests. From the repository root, `python -B tools/prompt_bench/replay_opro.py` verifies the repaired search using saved real responses with network access disabled. See the [root README](../../README.md) and [root log](../../log.txt) for results and provenance.

## Scoring, retries and OPRO budgets (October 4, 2026)

`evaluate()` and `evaluate_per_example()` share exact per-batch caching and complete-label parsing. Similar texts always get separate requests. Valid labels come from an explicit `TaskSpec`: Positive/Negative by default, or Entailment/Contradiction for binary NLI. Parsing is case-insensitive, with surrounding whitespace, optional matching quotes and one optional terminal period/exclamation mark. Explanations, partial words, multiple labels and labels from another task are invalid. Per-example predictions remain lowercase, with `INVALID` for malformed responses. Scoring never loads an embedding model.

`LLMBackend.generate()` accepts successful responses even when slow; `LLM_SLOW_THRESHOLD` no longer triggers retries. `LLM_RETRIES` still controls failure retries (default 2). Every failed or successful attempt contributes to `calls`, request time and traces. Permanent HTTP client errors are not retried; SDK automatic retries are disabled. Failed traces have unknown token usage, and successful token counts remain local estimates. `wall_time` is summed request time; `backoff_time` records retry waits separately, not total experiment elapsed time.

OPRO's `budget` is a per-search attempt cap, including retries and proposals, separate from final evaluation. It reserves whole scores before requesting proposals. If the initial score cannot fit, `CallBudgetExceeded` is raised before any request; if retries consume the allowance later, only fully scored candidates remain eligible. SAPO/entropy wrappers reserve their extra scoring requests. Custom backends must count every attempt in `stats.calls`, and custom scoring wrappers must implement `_evaluation_calls()` when they make extra calls. Other optimizer classes are not covered by OPRO's search guard.

Run results expose `search_api_calls`, `evaluation_api_calls`, `failed_attempts`, `unknown_usage_attempts`, `backoff_time` and `scoring_protocol: exact-label-v2-explicit-task`. Costs include both search and evaluation backends, exclude earlier calls on reused backends, and keep failed-request tokens explicitly unmeasured. Historical logs retain their original semantics, including the earlier `exact-label-v1` marker.

Run `python -B tests/test_harness_reliability.py` for 21 offline regressions. The [validation report](../../output/harness_reliability_2026-10-04/report.md) includes the saved-response replay; no fresh model inference was needed to validate these fixes. The entropy sampler's whitespace approximation remains an experiment, not genuine temperature-controlled sampling.

## Explicit tasks and reproducible DRO subsets

`Dataset.task` fixes the output schema independently of prompt wording. The NLI demo sets `BINARY_NLI`; other existing datasets default to `SENTIMENT`. The harness checks every dataset label before backend creation and passes the task through optimization, reflection, entropy scoring and final evaluation. `RunResult.task_spec` records the schema. Direct callers must pass the task explicitly for NLI:

```python
from prompt_opt_harness.tasks import BINARY_NLI
optimizer = OPRO(backend, budget=100, task=BINARY_NLI)
score = evaluate(backend, instruction, examples, task=BINARY_NLI)
```

The DRO wrapper now draws subsets from sorted source IDs using only the configured seed. Every candidate faces identical subsets, independent of instruction wording, input order or Python's process hash seed. Selected subset IDs are recorded in `optimizer_metadata.source_subset_ids`. This remains a source-subset robustness experiment, not a method trained on actual target domains. Run `python -B tests/test_tasks_and_dro.py` for eight offline invariance tests.

## Run it

No API keys needed, uses a mock backend:

```bash
python3 demo.py
python3 tests/smoke_test.py
```

## What's actually built

OPRO and EvoPrompt-lite work against `LLMBackend`; zero-shot and human-written
baselines are implemented. Local GEPA and TextGrad classes are simplified
experimental implementations, with additional experimental wrappers in
`optimizers/`. Official GEPA 0.1.4 and an ESPO-inspired adaptation are integrated
in the separate [controlled benchmark](../../tools/prompt_bench/README.md).
`optimizers/dspy_adapters.py` additionally provides optional DSPy 3.2.1
instruction-only adapters: `DSPyGEPA`, `DSPyMIPROv2`, and `DSPyCOPRO`.
These are separate from the local experimental classes and the standalone
GEPA benchmark. COPRO is never reported as TextGrad.

`OpenAIBackend`, `AnthropicBackend`, `GroqBackend`, `OllamaBackend` are real
API wrappers. The harness itself (budget tracking, OOD eval, logging,
report table) is backend-agnostic and tested.

This isn't meant to replace the OPRO wrapper already running in Colab, more
a second implementation plus the evaluator/logger/reporter pieces that
probably aren't in the notebook yet.

## Mock backend

`MockBackend` isn't a language model. It's a probability model built to
reproduce the thing we're actually testing for: a couple of "quality"
instruction fragments give a small accuracy bump on every domain, a couple
of "domain-cue" fragments (movie/cast/cinematography language) give a much
bigger bump but only on the source domain. Candidate generation is biased
toward whichever fragment looks like the bigger dev-set win, same as what a
real optimizer does when it's just chasing accuracy.

The source-domain bonus deliberately simulates overfitting. Numbers from this
backend are pipeline checks, not measured language-model performance.

## Running with a real model

```bash
export GROQ_API_KEY=...   # free, no card, console.groq.com
pip install openai
python3 run_groq_demo.py
```

Default model is `openai/gpt-oss-20b` on Groq's free tier. Open-weight,
fast, cheap. Groq has retired specific Llama versions before so if this
one 404s check console.groq.com/docs/models and pass a different model
name to `GroqBackend`.

No account alternative: `OllamaBackend`, runs fully local against Ollama
(`ollama pull llama3.1`). Slower without a GPU.

`run_groq_real_data.py` does the same thing against real data instead of
the toy dataset. Needs `pip install datasets`.

## Real data

`real_datasets.py` pulls SST-2 (source), amazon_polarity and
tweet_eval/sentiment (OOD) from Hugging Face. Not the exact datasets named
in the proposal (real SemEval-2017 tweets and the McAuley Amazon corpus
would need their own loader), but close and one line to swap later.

Note: HF renamed some of these repos recently (`glue` → `nyu-mll/glue`,
`amazon_polarity` → `fancyzhx/amazon_polarity`, `tweet_eval` →
`cardiffnlp/tweet_eval`). Already fixed here, just flagging in case you hit
the same error pulling a dataset elsewhere.

## Swapping backend or dataset

Backend: change what `backend_factory()` returns.

```python
def backend_factory():
    return OpenAIBackend(model="gpt-4o-mini")
```

Dataset: write a function that returns the same shape as
`load_demo_dataset()`.

```python
Dataset(
    source_domain="sst2",
    source=DomainSplit(dev=[...], test=[...]),
    ood={"amazon": [...], "tweets": [...]},
)
```

## DSPy adapters (October 7, 2026)

Install the pinned optional dependency in its own virtual environment. DSPy
3.2.1 depends on GEPA 0.0.27; the historical controlled benchmark uses GEPA
0.1.4, so keep their environments separate. Existing baseline runs do not need
DSPy, scikit-learn, or the embedding optimizer's dependencies.

```bash
pip install -r requirements-dspy.txt
python -B tests/test_dspy_adapters.py
```

```python
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import OllamaBackend
from prompt_opt_harness.optimizers import DSPyMIPROv2  # or DSPyGEPA / DSPyCOPRO

# dataset is a Dataset with disjoint source.dev, source.test and OOD examples.
# This call makes real model requests; the tests above use an offline backend.
result = run_experiment(
    "DSPy-MIPROv2", DSPyMIPROv2,
    lambda: OllamaBackend(model="gemma4:31b-cloud", temperature=0, max_tokens=16),
    dataset, budget=300, seed=37,
)
```

The adapters support the harness's explicit classification `TaskSpec`, including
sentiment and binary NLI. They do not implement the other branch's math task API.
The base sentiment prompt remains the default; these optimizers must be selected
explicitly.

- **Request accounting:** Every proposal, task score, retry and exported-prompt
  validation passes through the same backend. The hard search cap includes the
  initial base score and reserves a full final validation pass. Final source/OOD
  evaluation is counted separately by the harness. If compilation exhausts its
  allowance, only a candidate already scored on every validation example can be
  recovered and rescored with the reserved calls; otherwise the base is retained.
  Partial scores never enter the candidate pool. Provider
  errors unrelated to the cap remain visible. Tokens are the backend's local
  estimates, not provider billing. LM copies share the budget; calls are serialized.
- **Same inference behavior:** Task predictions use the existing exact prompt
  formatter, inference settings and strict label parser during search and final
  evaluation. MIPROv2's final demonstrations are disabled. Unexpected demos are
  rejected rather than silently discarded. DSPy proposal messages are carried
  as a role-preserving JSON conversation through the backend's text interface.
  COPRO can propose output prefixes, but this instruction-only adaptation does
  not apply them during search or final evaluation. It is not a full DSPy-program
  benchmark.
- **Source isolation:** Source development examples are split approximately
  50/50 into train/validation, stratified by label and stable under input reordering
  for a fixed seed. Each label needs at least two examples. Duplicate IDs/normalized
  texts and target arguments are rejected. GEPA/MIPROv2 receive explicit disjoint
  sets. COPRO searches only source training data, then its export is scored on
  source validation. The final source/OOD sets remain with `run_experiment`.
- **Selection and reproducibility:** The returned instruction is checked against
  the base on the same source validation set; ties retain the base. This rescore
  verifies exported behavior, not performance on a new confirmation set.
  Metadata records both split ID lists, seeds, settings, evaluated instructions,
  and stop reason. The public candidate pool contains only the base and a fully
  validated exported instruction. GEPA/MIPROv2 receive the supplied optimizer seed;
  COPRO has no optimizer-seed parameter. Provider determinism is not guaranteed.
- **Decoding:** Inference retains the supplied backend's temperature/token cap.
  Proposal defaults are temperature 0.7 and 2,048 output tokens, configurable on
  the adapter constructor. DSPy may change proposal temperatures internally;
  those overrides are honored without changing task inference settings.
  Each backend attempt trace records the effective model, temperature and cap.

The offline tests exercise the real pinned DSPy compilers with deterministic
fake responses and blocked network access. They check budgets/retries, prompt
and label equivalence, data boundaries, configuration propagation, and all three
optimizers. They establish integration correctness, not a Gemma accuracy gain.

Verified October 7, 2026: 21 DSPy regressions and 44 existing harness reliability,
data-isolation, explicit-task/DRO and smoke checks passed. The pinned GEPA,
MIPROv2 and COPRO compilers all completed offline search tests. Real-model
performance still requires a new controlled experiment with fresh held-out data.

## Not done

- Broad real-model transfer experiments; the harness already supports a separate evaluation backend
- Reasoning/code tasks (GSM8K, MBPP, etc.) — demo is sentiment only
- Target-trained reference with separate target-development and final-test sets (the old oracle is disabled)
