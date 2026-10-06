# Prompt optimizer harness

Implements the harness from the capstone proposal: dataset loader, optimizer
wrapper, OOD evaluator, logger/reporter.

## Run it

No API keys needed, uses a mock backend:

```bash
python3 demo.py
python3 tests/smoke_test.py
```

## What's actually built

OPRO and EvoPrompt-lite are real implementations, both work against any
`LLMBackend`. Zero-shot and human-written baselines are done. GEPA, MIPROv2,
and TextGrad are not implemented, just stubbed in `optimizers/dspy_adapters.py`
with notes on how to wire them up (they're DSPy teleprompters, need a
`dspy.Signature`/`Module` and a metric function, different shape of work
than OPRO/EvoPrompt).

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
Once you have `optimized.signature.instructions` it drops into this
harness's `evaluate()` like any other optimizer's output.

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
- Reasoning/code tasks (GSM8K, MBPP, etc.) — demo is sentiment only
- Oracle upper-bound baseline (optimizer run directly on each OOD domain)
