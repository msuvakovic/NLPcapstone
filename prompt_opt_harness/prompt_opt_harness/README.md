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

OPRO, EvoPrompt-lite, GEPA, MIPROv2, and TextGrad are real implementations, and all work against any `LLMBackend`. Zero-shot and human-written baselines are done. The DSPy optimizers (GEPA, MIPROv2, TextGrad) dynamically map the datasets to `dspy.Signature`s for seamless running in this harness.

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

## Not done

- Cross-model transfer (optimize on one model, eval on another)
- Oracle upper-bound baseline (optimizer run directly on each OOD domain)
