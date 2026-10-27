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

Every optimizer works against any `LLMBackend`, shares the same budget guard,
and logs every graded answer.

| Name | What it does |
| --- | --- |
| Zero-shot, Human-written | baselines, no search |
| `opro` OPRO | asks the model for a better prompt, shown the best earlier prompts and their scores |
| `evoprompt` EvoPrompt-lite | mutates and crosses over a small population of prompts |
| `gepa` GEPA-lite | reads one wrong answer, writes what the prompt missed, rewrites it, keeps it if better. A small greedy version: no Pareto frontier, so not full GEPA |
| `textgrad` TextGrad-lite | critiques a batch of wrong answers, then rewrites the prompt from the critique. Not the TextGrad library |
| `random` Random-search | independent rewrites of the starting prompt with no feedback. Shows what the search signal adds beyond just trying more prompts |
| `pbt` PBT | population-based training: crossover of the best two, or mutation, with a periodic cull (Kanika) |
| `ensemble` Ensemble | paraphrases the prompt, then scores one prompt that combines the best few (Kanika) |
| `bayesopt` CostBayesOpt | Gaussian-process search over prompt text, prefers shorter prompts. Needs `scikit-learn` (Kanika) |
| `reg-opro` Reg-OPRO | OPRO with a per-word length penalty (Kanika) |
| `dro-opro` DRO-OPRO | OPRO scored by its worst accuracy over fixed subsets of the tuning questions (Kanika) |
| `sapo-opro` SAPO-OPRO | OPRO scored on the prompt and its paraphrases, worst wins. Gets 3x the budget because each score costs about 3 passes (Kanika) |
| `entropy-opro` Entropy-OPRO | OPRO penalizing answers that change between samples. Gets 5x the budget (Kanika) |
| `guard-opro`, `guard-gepa` | the search winner must also beat the starting prompt on separate source questions (`source.confirm`) the search never saw, else the starting prompt is kept (Albert) |
| `dspy-gepa`, `dspy-mipro` | the real DSPy GEPA and MIPROv2, instruction-only, through this harness's own model calls and budget. Need `pip3 install "dspy[optuna]==3.2.1"` (Albert, apinero) |
| `oracle` Oracle-OPRO | upper bound: OPRO tuned on a held-out sample of the new dataset. Not a fair optimizer, see below |

GEPA-lite, TextGrad-lite and Random-search follow the team's branches
(Kanika's budget-guard branch); the oracle follows Milos's `ood_val` idea.
They're rewritten here against this harness so they get the same seeds, token
counts and per-question logs as OPRO and EvoPrompt-lite.

`--optimizers` takes names or the groups `core`, `extended`, `dspy`, `all` (the list lives in `registry.py`).
Optimizers whose package is missing are skipped with the install command, not an error. One optimizer failing
does not stop the others; `--resume` retries it.

Not ported from the other branches: EmbeddingBayesOpt (needs sentence-transformers and falls back to character
counts without it), Reflective (the same idea as GEPA-lite), DSPy COPRO (a different algorithm from TextGrad, and
the team's rule is not to label it TextGrad), and Albert's round-sweep and improvement-pilot tools (sentiment
experiments that run on his own pipeline). A real TextGrad does not exist in DSPy 3.2, so TextGrad-lite stays.

### Budget guard

`budget` is a hard limit on calls during the search. Before each step an
optimizer checks that the whole step (the proposals plus a full pass over the
tuning questions) still fits, so no optimizer goes over, and they all get the
same budget. The budget has to cover at least the starting prompt's pass over
the tuning questions. Final testing is counted separately in `api_calls`.

### Data-leak checks

`run_experiment` refuses to start if any tuning question also appears in a
reported test set, ignoring case and spacing. Optimizers only ever receive the
source tuning set. The one exception is the oracle, which receives `ood_val`: a
separate sample of the new dataset that is never reported. `tests/smoke_test.py`
checks both rules.

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

`run_groq_reasoning_real_data.py` has a few more flags:

- `--seeds N` runs N seeds (42, 43, ...) and prints mean +/- sd at the end. Each
  seed draws a different sample of questions. The optimizers themselves run at
  temperature 0, so the seed alone doesn't change what they propose.
- `--proposal-temp 0.8` sets the temperature for the optimizers' proposal calls
  only. Left unset, proposals use the backend default (0).
- `--split-mode same|train_test`. `same` (default) takes tuning and test
  questions from disjoint slices of GSM8K test. `train_test` is the old setup,
  tuning questions from GSM8K train and test questions from GSM8K test.
- `--start default|bare`. `bare` starts the optimizers from "Solve the math word
  problem." instead of the prompt that already says to show your work.
- `--log-dir logs/after` keeps a run's logs in their own folder.
- `--resume` skips any method and seed already saved in `--log-dir`, so an
  interrupted run can pick up where it stopped.
- `--optimizers opro,evoprompt,gepa,textgrad,random,oracle` picks which
  optimizers to run (default `opro,evoprompt`; the baselines always run).
  Each one runs once per seed at the full `--budget`, so five optimizers take
  about two and a half times as long as two.

Budget counts every call, dev evaluations included. One OPRO candidate costs
n + 1 calls, so a budget of 180 gives roughly six candidates at n=30 and three
at n=50 (GEPA-lite and TextGrad-lite cost n + 2). Scale the budget with
`--n-per-split`.

MockBackend still only supports sentiment (it's a scripted simulation, would
need its own fragment model for math). It raises clearly instead of silently
producing garbage if you point it at the reasoning task, use a real backend
for that.

Budget is 180 now instead of 40, at n=30/split, 40 wasn't enough for the
optimizer to get more than one search step in.

## Logs

`save_run` writes one JSON per method and seed. Besides the scores it records the
model, backend, seed, question counts, budget, token counts, and the stats of
every candidate prompt (calls used so far, answers cut off, answers with no
final `Answer:` line). It also keeps every question, the model's full response,
the parsed answer, the gold answer, and the `finish_reason` for the winning
prompt on its tuning questions, the same-domain test questions, and each OOD set.
`finish_reason: length` means the 512-token cap cut the answer off, which
counts as wrong.

Logs from before this change have no per-question data and are skipped by the
demo page.

## Demo page

`make_demo.py` turns a log folder into one self-contained HTML file that works
offline (no internet, no server):

```bash
python3 make_demo.py logs/after --out demo.html --footer "UC Santa Cruz NLP capstone"
python3 make_demo.py logs/after --out demo.html --headline "Your one-line takeaway" \
  --next-step "A harder different dataset" --next-step "More optimizers"
```

It opens as a guided walkthrough you can talk through, one click at a time:
the three piles of questions, the tuning loop, a prompt being rewritten live,
every winner graded question by question, one question where two prompts
disagree, the token-limit catch (only shown if answers were cut off), a chart of
accuracy as questions get further from the tuning data, and takeaways.

Keys: right and left arrows (or space) step forward and back, `N` shows what to
say on the current step, `A` shows the whole talk track, `P` starts hands-free
auto-play (or open `demo.html?booth=1`), `E` opens the data explorer where every
square is clickable, `W` returns to the walkthrough, `T` switches light and dark,
`F` goes full screen. The first press while an animation is running skips to its
end. The speaker notes are generated from the numbers in your logs, so they stay
true when you rerun.

Logs from the mock backend or a test fixture get a banner saying the data is
simulated. Oracle runs are left off the page, because their tuning numbers are
measured on the new dataset; they still appear in the text report.

## Final run and analysis

One command runs the real experiment and builds everything you need for the report and the demo:

```bash
bash run_all.sh
```

It checks Ollama and the packages, runs a 10-question quick check (and prints how long the full run will take),
runs the full experiment (3 seeds, every optimizer that is installed, GSM8K to SVAMP + GSM-Plus), then `analyze.py` and `make_demo.py`.
Stop it and start it again at any time: finished runs are skipped. Settings are environment variables
(`SEEDS`, `N_PER_SPLIT`, `BUDGET`, `MAX_TOKENS`, `OOD`, `MODEL`, `LOG_DIR`); see the top of the script.

**Budget and sample size go together.** Scoring one prompt costs `N_PER_SPLIT` calls and the budget counts every call,
so keep `BUDGET` at 6x `N_PER_SPLIT` or more. Accuracy near 90% needs a few hundred questions to separate methods:
raise `N_PER_SPLIT` (and `BUDGET`) if time allows.

`analyze.py` (standard library only) reads the logs and writes `results.md`, `results.csv` and `fig_gap.svg`:

| Section | Answers |
| --- | --- |
| Findings | auto-written summary of everything below, from the logs only |
| 1 Headline metrics | dev, source test, new-data accuracy, dev-test gap, OOD gap, worst case, variance, calls, tokens, gain per call (Q1) |
| 2 Do the gains transfer? | paired bootstrap CIs over questions against Zero-shot: gain on source, gain on new data, transfer shortfall, and a label per optimizer (Q1, Q3) |
| 3 Against the human prompt | the same, against the stronger baseline |
| 4 Robustness across new datasets | accuracy on each new dataset, worst dataset and spread (Q2) |
| 5 Confirmation guard decisions | what each guarded optimizer kept (only if guarded runs exist) |
| 6 Truncation audit | share of answers cut off per method, and each gap recomputed without them. Verdict OK / CHECK / CONFOUNDED |
| 7 Token-cap sensitivity | with `--compare`: the same metrics at two token caps |

The labels are conservative. OVERFITS needs a source gain and a transfer shortfall whose 95% intervals are both above zero;
a source gain with an inconclusive shortfall is reported as inconclusive, never as overfitting.

**Is the 512-token cap the confound?** Read section 5. If the verdict is not OK, rerun at a larger cap and compare:

```bash
MAX_TOKENS=1024 LOG_DIR=logs/after_1024 bash run_all.sh
python3 analyze.py --logs logs/after_1024 --compare logs/after --out analysis/final
```

Report the larger-cap numbers when the two disagree.

**New datasets** (`--ood`, or `OOD=` for the script): `svamp` (original), `gsmplus` (GSM-Plus, harder perturbed problems),
`asdiv`, `multiarith`. Two or more are needed for cross-domain variance and a meaningful worst case. Column names for
everything except SVAMP come from the dataset cards and have not been run against the real downloads; if a load fails the
loader stops with a clear message before any model call, and `OOD=svamp ./run_all.sh` falls back to the original.

## Not done

- Cross-model transfer (optimize on one model, eval on another)
- Code tasks (MBPP etc.)
- Oracle baseline for the sentiment loader (only the reasoning loader provides `ood_val`)
- The analysis assumes the reasoning logs (per-question records, schema v2); the sentiment evidence bundle on `apinero` uses its own format
- A real TextGrad (DSPy 3.2 has none; TextGrad-lite is the stand-in)
