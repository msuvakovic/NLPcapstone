# Domain Generalization of Prompt Optimizers — harness prototype

A working, tested implementation of the harness architecture from the
capstone proposal / slide deck:

```
Dataset Loader -> Optimizer Wrapper -> Prompt Candidate Pool -> OOD Evaluator -> Logger & Reporter
```

Run it right now, no API keys, no network:

```bash
python3 demo.py
python3 tests/smoke_test.py
```

## What's real vs. mocked

| Component | Status |
|---|---|
| Dataset Loader (`datasets.py`) | Real, but the shipped dataset is a small hand-written stand-in for SST-2 → Amazon/Tweets (6 dev / 6 test / 8+8 OOD). Swap in real corpora — see below. |
| Unified Optimizer Interface (`optimizers/base.py`) | Real. `optimize(task_desc, dev_examples) -> Candidate`, `evaluate(backend, prompt, examples) -> accuracy`, exactly as in the diagram. |
| OPRO (`optimizers/opro.py`) | Real algorithm (single-trajectory, history-guided meta-prompting), runs against any `LLMBackend`. |
| EvoPrompt-lite (`optimizers/evoprompt.py`) | Real algorithm (population + crossover/mutation), same interface. |
| Baselines (`optimizers/baselines.py`) | Real — zero-shot and human-written, the two *required* controls from the deck. |
| GEPA / MIPROv2 (`optimizers/dspy_adapters.py`) | **Stubbed.** These are DSPy teleprompters, not standalone prompt-string optimizers, so wiring them in is a different shape of work (a `dspy.Signature`/`Module` + metric fn) than OPRO/EvoPrompt. The file has verified-current `pip install`/import/usage snippets so this is a few hours of work, not a redesign. |
| TextGrad | Not started. Same story as GEPA/MIPROv2 — has its own API (`pip install textgrad`), would need its own thin adapter to the `Candidate` interface. |
| OOD Evaluator, Logger, Reporter (`harness.py`, `logger.py`, `report.py`) | Real, backend-agnostic. |
| LLM calls (`llm_backends.py`) | `OpenAIBackend` / `AnthropicBackend` / `GroqBackend` / `OllamaBackend` are real, thin API wrappers — untested here because this sandbox can't reach any of those hosts. `MockBackend` is a deterministic offline simulation (see below) used only to prove the plumbing works end-to-end. |
| Real-data loader (`real_datasets.py`) | Real code, untested here (no route to `huggingface.co` in this sandbox). Run it on your own machine first. |

**Bottom line:** the architecture, budget accounting, OOD evaluation, and
reporting are real and tested. Two of five optimizers are fully implemented
against a real-backend-ready interface. What's *not* here is real model
calls (needs your API keys — see `run_real_backend_example.py`, literally a
one-line change from `demo.py`) and the GEPA/TextGrad/MIPROv2 adapters
(scoped, not built).

This is meant to sit alongside — not replace — the OPRO wrapper you already
have running in Colab with the 50-call budget cap; this prototype's OPRO is
a from-scratch reference implementation against the same idea, useful for
comparing against or cannibalizing, and the OOD-evaluator/logger/reporter
pieces are the parts the Colab notebook probably doesn't have yet.

## Why MockBackend gives non-trivial results without an LLM

It's not a language model — it's a probability model calibrated to
reproduce the exact phenomenon Q1–Q3 ask about, so the demo output is
illustrative of the mechanism, not a claim about real numbers:

- A couple of "quality" instruction fragments (e.g. *"Consider the overall
  tone, not just individual words"*) give a small accuracy bump on **every**
  domain — generically good instructions really do transfer a little.
- A couple of "domain-cue" fragments (e.g. *"weigh comments about the
  cinematography and pacing heavily"*) give a **much bigger** bump, but only
  on the source (movie-review) domain — because they only make sense there.
- Candidate generation is weighted toward whichever fragments look like the
  bigger dev-set win, which is exactly what a pure dev-accuracy-maximizing
  search does in the real methods, and exactly why it finds domain-specific
  shortcuts.

Run `demo.py` a few times (it's seeded, so it's the same every time) and
you'll see OPRO converge on a prompt loaded with cinematography/plot/cast
language and post a real OOD gap on the `amazon`/`tweets` domains — a toy
version of exactly the failure mode in the proposal's Motivation slide.
**Treat every number from MockBackend as a pipeline sanity check, never as
an experimental result.**

## Fastest path to a real result (e.g. before a meeting today)

```bash
# 1. free API key, no credit card: https://console.groq.com
export GROQ_API_KEY=...
pip install openai
python3 run_groq_demo.py          # toy dataset + real open-weight model, ~2 min setup
```

**Model:** `openai/gpt-oss-20b` (the default in `GroqBackend`) — OpenAI's
smaller open-weight model, running on Groq's free tier (~30 req/min, 200k
tokens/day, comfortably enough for this harness's small dataset). It's fast,
cheap, and genuinely open-weight, which is why it's the default rather than
a Llama model — Groq has retired specific Llama versions before, so if
`openai/gpt-oss-20b` ever 404s, check `console.groq.com/docs/models` for the
current lineup and pass `GroqBackend(model="...")`. `openai/gpt-oss-120b` is
a larger, higher-quality open-weight option on the same free tier if you
want to compare model scale.

No-signup alternative: `OllamaBackend` in `llm_backends.py` runs fully
locally against Ollama (`ollama pull llama3.1` or `qwen2.5`/`mistral`/`phi4`)
— slower without a GPU, but no account and no rate limits.

Once `run_groq_demo.py` works, `run_groq_real_data.py` is the same thing
against real corpora (SST-2 → Amazon reviews / tweets) via
`real_datasets.py` — needs `pip install datasets` and Hugging Face Hub
access. **This loader is untested in the sandbox this project was built in**
(no route to `huggingface.co` there); the `datasets` library calls it makes
are standard and stable, but do a quick
`load_real_dataset(n_per_split=5)` sanity check on your own machine first.
It approximates the proposal's datasets rather than matching them exactly —
see the docstring in `real_datasets.py` for the specific substitutions
(GLUE SST-2 validation split in place of train/test, `amazon_polarity` in
place of the McAuley Amazon corpus, `tweet_eval/sentiment` with neutral
dropped in place of SemEval-2017 Task 4A) and swap in the exact ones later
if needed — only that one file would change.

## Swapping in a real backend

```python
# demo.py currently has:
def backend_factory():
    return MockBackend(dataset.lookup(), source_domain=dataset.source_domain)

# change to:
def backend_factory():
    return OpenAIBackend(model="gpt-4o-mini")   # needs OPENAI_API_KEY
    # or: AnthropicBackend(model="claude-3-5-haiku-20241022")  # needs ANTHROPIC_API_KEY
```

Nothing else changes — see `run_real_backend_example.py` for the full file.

## Swapping in real datasets

Write a function returning the same `Dataset` shape as
`datasets.load_demo_dataset()`:

```python
Dataset(
    source_domain="sst2",
    source=DomainSplit(dev=[...], test=[...]),
    ood={"amazon": [...], "tweets": [...]},
)
```

For the actual corpora named in the proposal (SST-2, Amazon Reviews,
SemEval-2017 Task 4A, HotpotQA, TriviaQA), the Hugging Face `datasets`
library is the natural loader — this sandbox can't reach
`huggingface.co` to demonstrate it, but the shape above is all `harness.py`
needs regardless of where the examples came from.

## Wiring up GEPA / MIPROv2 (verified against current DSPy docs)

```bash
pip install "dspy>=3.2.1,<3.3"          # stable series as of this writing
pip install "dspy[optuna]>=3.2.1,<3.3"  # MIPROv2's Bayesian search backend
```

```python
import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key=...))

# MIPROv2 — the proposal's required prior-SOTA baseline
from dspy.teleprompt import MIPROv2
teleprompter = MIPROv2(metric=your_metric, auto="light")  # "light" | "medium" | "heavy"
optimized = teleprompter.compile(your_dspy_module, trainset=trainset)

# GEPA — the proposal's state-of-the-art baseline
gepa = dspy.GEPA(metric=your_metric, auto="light", reflection_lm=dspy.LM("openai/gpt-5"))
optimized = gepa.compile(your_dspy_module, trainset=trainset, valset=devset)
```

Both need a `dspy.Signature`/`Module` for sentiment classification and a
`metric(gold, pred) -> float` function, not just a prompt string — that's
the real adapter work, sketched in `optimizers/dspy_adapters.py`. Once you
have `optimized.signature.instructions`, it drops straight into this
harness's `evaluate()` / OOD evaluator like any other optimizer's output.

## What's deliberately out of scope here

- Cross-model transfer (optimize on one model, evaluate on another) —
  `harness.run_experiment` takes a `backend_factory`, so running the same
  `best.instruction` through a second backend for the OOD evaluator step is
  a small extension, not implemented.
- Reasoning/code task families (HotpotQA, GSM8K, MBPP, etc.) — the demo is
  sentiment-only; the reasoning task would need its own prompt template and
  scoring function alongside `prompts.build_classification_prompt`.
- Oracle upper bound baseline (OPRO run directly on each OOD domain's own
  dev set) — straightforward with `run_experiment`, just pass a
  single-domain `Dataset` per target; not wired up here.
