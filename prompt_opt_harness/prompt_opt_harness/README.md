# Prompt optimizer harness

Implements the harness from the capstone proposal: dataset loader, optimizer
wrapper, OOD evaluator, logger/reporter.

## Pilot results: Qwen 2.5 7B, October 9, 2026

Executed locally via Ollama on **one NVIDIA GB300**, sequentially, with
`qwen2.5:7b` (local model ID `845dbda0ea48`), temperature 0 and a 512-token
generation cap. Dataset sampling and optimizer seed: **42**. Source development
data: 30 examples from `openai/gsm8k` (`main`, train); source test: 30 GSM8K test
examples; OOD test: 30 examples from `MU-NLPC/Calc-svamp` (`default`, test).

| Method | Dev | GSM8K test | SVAMP | Optimization calls / cap | Held-out calls | Total calls |
| :-- | --: | --: | --: | --: | --: | --: |
| Zero-shot | 0.933 | 0.867 | 0.867 | 30 / 30 | 60 | 90 |
| Human-written | 0.967 | 0.867 | 0.933 | 30 / 30 | 60 | 90 |
| OPRO | 0.967 | 0.900 | 0.900 | 278 / 300 | 60 | 338 |
| EvoPrompt-lite | 0.967 | 0.900 | 0.867 | 278 / 300 | 60 | 338 |
| GEPA (conservative search) | 0.933 | 0.867 | 0.900 | 102 / 300 | 60 | 162 |
| MIPROv2 (conservative search) | 0.933 | 0.833 | 0.900 | 86 / 300 | 60 | 146 |

The optimization cap includes development scoring, candidate proposals and
DSPy reflection/task requests. Held-out evaluation is outside that cap.
Baselines do no search: their 30 calls score the fixed prompt on development
data. All six selected artifacts have `optimizer_budget_exhausted=false`.
**300 is a maximum, not an equal amount of search actually consumed.**

### Result provenance

Selected compact artifacts under `logs/`:

- `zero-shot_20261009T004302Z.json`
- `human-written_20261009T004415Z.json`
- `opro_20261009T004949Z.json`
- `evoprompt-lite_20261009T005506Z.json`
- `gepa_20261009T010718Z.json`
- `miprov2_20261009T010937Z.json`

The first four came from the corrected full run. GEPA and MIPROv2 were rerun
individually after reducing their internal search settings; their earlier
`gepa_20261009T010022Z.json` and `miprov2_20261009T010225Z.json` runs reached the
request cap and evaluated fallback instructions. Those earlier rows are
**excluded**. Historical logs are not interchangeable with this selected set.
Runtime transcripts are kept locally in `run_logs/`, not committed.

### Interpretation and limitations

- Human-written has the highest observed SVAMP accuracy (28/30); this is not
  evidence of statistically significant superiority. One example changes an
  accuracy by 3.33 percentage points, and only one seed was run.
- OPRO and EvoPrompt-lite selected the original reasoning instruction, not a
  newly improved prompt. Their SVAMP scores differ despite identical selected
  instructions, so temperature 0 has not established deterministic evaluation.
  Do not infer optimizer gains from those differences.
- GEPA and MIPROv2 return an instruction extracted from a compiled DSPy program.
  The harness evaluates it with its ordinary prompt builder, **not the full
  compiled program**. MIPRO demonstrations and DSPy formatting are not carried
  into held-out evaluation. These are instruction-transfer pilots, not faithful
  full-program benchmark results.
- Conservative GEPA uses 67 metric calls; MIPRO uses two instruction candidates
  and `num_trials=1` at this configuration (its log includes the default program
  plus another evaluated trial). These small searches are not the auto-light
  presets and should not be ranked as equivalent optimization effort.
- Native TextGrad is unavailable in installed DSPy 3.2.1. It is explicitly
  skipped; COPRO is no longer silently reported as TextGrad.
- With one OOD domain, OOD-only variance is necessarily zero and worst-case OOD
  accuracy simply equals SVAMP accuracy. Neither adds evidence of robustness.
- Responses are capped at 512 tokens, and the scorer can fall back to the last
  number when `Answer:` is absent. Truncation/formatting can therefore affect
  accuracy. Provider-side SDK retries still need auditing: the wrapper accounts
  for its explicit retries, but does not disable the SDK's own retry layer.

### How to improve the experiment

1. **Make the evaluated object consistent.** For an instruction-only comparison,
   disable MIPRO demonstrations during search and use the same prompt format for
   search and evaluation. Alternatively, save and evaluate each full DSPy program
   end-to-end and label that as a separate experiment.
2. **Validate request accounting end-to-end.** Disable SDK-internal retries or
   account for each transport attempt; test failures/timeouts and compare counters
   with server request logs. Mark capped/fallback runs invalid automatically.
3. **Increase evaluation size and repeat seeds.** Separate search/training and
   validation data within the source domain; keep source-test/OOD data untouched
   during optimizer tuning. Report confidence intervals and paired comparisons,
   and repeat evaluation of identical prompts to quantify inference variation.
4. **Compare cost-quality curves.** Log exact search settings and proposal,
   reflection, scoring, token and elapsed-time costs. Evaluate several budgets
   rather than assume every optimizer spends the same 300 requests.
5. **Audit answer validity.** Save per-example outputs and finish reasons;
   distinguish wrong arithmetic, parse failures and truncation. Test a larger
   common token cap without choosing settings based on OOD scores.
6. **Expand domain coverage.** Add another math OOD domain and later sentiment/NLI
   tasks. Implement real TextGrad through its own adapter before including it.
   Cross-model transfer and OOD-trained oracle baselines remain stretch goals.

### Reproduce this pilot on one GPU

Run from the project directory containing this README. Install Ollama separately
and use a fresh Python environment; the recorded direct Python dependency versions
are DSPy 3.2.1, datasets 3.6.0, openai 2.54.0 and Optuna 5.0.0. Ollama was 0.40.1.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install 'dspy[optuna]==3.2.1' 'datasets==3.6.0' 'openai==2.54.0' 'optuna==5.0.0' pytest
CUDA_VISIBLE_DEVICES=0 ollama serve
```

In a second terminal, with the server running:

```bash
ollama pull qwen2.5:7b
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/smoke_test.py
.venv/bin/python run_groq_reasoning_real_data.py --backend ollama --model qwen2.5:7b --budget 300 --n-per-split 30
# Targeted runs use the same seed/data sampling:
.venv/bin/python run_groq_reasoning_real_data.py --backend ollama --model qwen2.5:7b --budget 300 --n-per-split 30 --methods GEPA MIPROv2
```

A rerun may differ numerically; preserve sampled IDs, model digests and full
environment metadata in future runs. Current JSONs do not contain all that metadata.

### Repository hygiene

The root `.gitignore` excludes virtual environments, Python caches, local secrets,
model weights/Ollama state and runtime transcripts. Only the six selected new
result JSONs above are allowlisted; existing tracked historical JSONs are retained.
Unselected/fallback results remain local. The large previously tracked Ollama log
and bytecode are removed from the current index, **not deleted locally**. This does
not purge them from existing Git history; no history rewrite is performed.



## Run it

No API keys needed for the mock smoke tests (DSPy tests require the optional
DSPy dependency). The legacy demo also attempts DSPy/TextGrad methods and is
not a standalone standard-library-only entry point:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/smoke_test.py
```

## What's actually built

OPRO and EvoPrompt-lite implement instruction search; zero-shot and human-written
baselines use fixed instructions. GEPA/MIPROv2 have DSPy adapters with shared request
accounting and conservative settings. Their instruction-only transfer limitations
are described above. TextGrad is conditional and unavailable under DSPy 3.2.1;
requesting it raises rather than silently substituting COPRO.

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

Run `demo.py` a few times, it's seeded so the output doesn't change. OPRO
converges on a prompt full of cinematography/plot language and posts a real
OOD gap on amazon/tweets. Numbers from this backend are a pipeline check,
not a result.

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

Reasoning tasks (GSM8K -> SVAMP) are supported and implemented via `real_datasets_reasoning.py` and run via `run_groq_reasoning_real_data.py`.

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

## GEPA / MIPROv2 via DSPy

The following is a standalone DSPy example, not this harness's capped pilot
configuration. Full-program evaluation requires saving/running the returned module;
extracting just its instruction does not preserve demonstrations or formatting.

```bash
pip install "dspy>=3.2.1,<3.3"
```

```python
import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key=...))

from dspy.teleprompt import MIPROv2
teleprompter = MIPROv2(metric=your_metric, auto="light")
optimized = teleprompter.compile(your_dspy_module, trainset=trainset)

gepa = dspy.GEPA(metric=your_metric, auto="light", reflection_lm=dspy.LM("openai/gpt-5"))
optimized = gepa.compile(your_dspy_module, trainset=trainset, valset=devset)
```

Both need a signature/module and metric function, not just a prompt string.
The current adapter extracts `compiled_module.prog.signature.instructions` and
evaluates that instruction separately; see the limitations and improvement plan.

## Reasoning task

Sentiment turned out to be too easy for gpt-oss-20b, everything saturated
near 1.0. Added a second task: math word problems, GSM8K-style, scored by
extracting the final number instead of matching a label string.

`tasks.py` holds a `Task` (prompt builder + correctness check), one for each
task. Every optimizer, `harness.run_experiment`, and `evaluate()` now take a
`task=` argument, defaults to sentiment so nothing existing changes.

```bash
python3 run_groq_reasoning_demo.py        # toy word problems
python3 run_groq_reasoning_real_data.py   # real GSM8K -> SVAMP, needs `pip install datasets`
```

All four `run_groq_*.py` scripts take `--backend groq|ollama`, `--model`, and
`--budget` (the real-data ones also take `--n-per-split`). Ollama runs fully
local, no API key:

```bash
python3 run_groq_reasoning_real_data.py --backend ollama
python3 run_groq_reasoning_real_data.py --backend ollama --n-per-split 10 --budget 60   # quick run
```

Backends cap generation at 512 tokens and retry on 429s and timeouts. Without
the cap a local model can ramble until the request times out.

MockBackend still only supports sentiment (it's a scripted simulation, would
need its own fragment model for math). It raises clearly instead of silently
producing garbage if you point it at the reasoning task, use a real backend
for that.

Budget is 180 now instead of 40, at n=30/split, 40 wasn't enough for the
optimizer to get more than one search step in.

## Not done

- Cross-model transfer (optimize on one model, eval on another)
- Oracle upper-bound baseline (optimizer run directly on each OOD domain)
