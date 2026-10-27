# Which codebase do we finish with? A proposal

Written Oct 21, 2026. The table comes from searching each branch on `origin` for the files and functions named; it is not a line-by-line review.

## What each line of work has

| | `main` | `apinero` | `albert` | `kanika/optimizer-budget-guard` | `milostest2` | this harness |
| --- | --- | --- | --- | --- | --- | --- |
| Split-overlap / leak check | no | no | yes | yes | no | yes |
| Budget guard | no | yes | yes | yes | no | yes |
| Cut-off answers (`finish_reason`) logged | no | no | yes | yes | no | yes |
| Oracle optimizer (tunes on the target) | no | no | yes | yes | yes | yes |
| Random-search baseline | no | yes | no | yes | no | yes |
| Hand-written GEPA / TextGrad | no | no | yes | yes | yes | simplified "-lite" versions |
| Confirmation guard (winner must beat the start on separate source questions) | no | reports Albert's guard results only (no guard code) | yes (in his pilot code) | no | no | yes (`guard-opro`, `guard-gepa`) |
| DSPy adapters (GEPA, MIPROv2) | yes | yes | yes, with tests | yes | yes | ported: `dspy-gepa`, `dspy-mipro` (budget-capped, exact task prompt) |
| PBT, Ensemble, CostBayesOpt, DRO / SAPO / length / entropy variants | no | no | yes | yes | yes | ported |
| EmbeddingBayesOpt, Reflective, COPRO, round-sweep tools | no | no | yes | yes | partly | not ported (see README) |
| Analysis tools (CIs, token caps, protocols, round sweeps) | no | report builder | yes | yes | no | `analyze.py` |
| One-command run of everything | no | no | no | no | no | `bash run_all.sh` |

Real-model evidence so far, from `apinero`'s report of Oct 18:

- Sentiment, Gemma 4 31B: Albert's artifacts (`albert` at `98a9118`), three seeds, 300 final examples per dataset, two new datasets.
- Reasoning, Qwen 2.5 7B: one seed, 30 questions per split, **one** new dataset (SVAMP), no cut-off-answer logging. The report itself says this cannot estimate variance or rank methods.

## The gap that matters

The reasoning pilot is too small and has one new dataset. This harness fixes both: 3 seeds, 50+ questions per split, two new datasets (SVAMP + GSM-Plus), cut-off answers logged and audited, confidence intervals on every comparison.

## Options

**A. Keep both lines, freeze where each result came from (recommended).**
- Sentiment results cite `albert@98a9118`. Reasoning results cite this harness on a new branch, `reasoning-final`, cut from `main`.
- Nothing has to be merged before the deadline, and every number in the report points to the commit that produced it.
- The cost: two codebases exist at the end. Say so in the report.

**B. Move the reasoning run onto Albert's tooling.** More methods and tooling, but the reasoning task and the Ollama path would have to be ported and re-verified. Risky with the time left.

**C. Merge everything into one line.** The cleanest end state, and weeks of work. Not realistic now.

## What to ask the team

1. Agree on option A (or pick another).
2. Decide who runs the final Qwen reasoning run, and on which machine.
3. Ask mentors (Leo, Rohan, Samson) whether the oracle baseline and the GSM-Plus shift are what they want in the final report.

## Message to paste to the team

> Proposal for finishing: keep both lines and pin every result to the commit that produced it. Sentiment results = `albert@98a9118`. For reasoning I've built a branch (`reasoning-final`, off `main`) that adds 3 seeds, a second new dataset (GSM-Plus), a truncation audit and CIs, so we can answer Q2 and Q3 for reasoning, which the Oct 18 report marks as pilot-only. It also carries over Albert's leak check, confirmation guard and DSPy GEPA/MIPROv2, Kanika's budget guard, random search, PBT, Ensemble, BayesOpt and DRO/SAPO/length/entropy variants, and Milos's oracle idea, rewritten to share one log format and one budget rule, so a single `bash run_all.sh` runs all of them. Objections to this plan, or a better one? Also: who has a machine free to run the final Qwen run? `bash run_all.sh` prints a time estimate after its 10-question quick check.
