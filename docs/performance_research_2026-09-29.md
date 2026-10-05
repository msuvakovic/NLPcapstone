**Performance improvement research and implementation plan**

Research checked September 29, 2026. This is a source and saved-result review; no new model experiments were run and no existing implementation was modified.

The strongest practical direction is to integrate the official GEPA implementation, establish a trustworthy OOD benchmark, and then test structured error diagnosis plus stable prompt selection. The potential research contribution is whether these additions improve transfer to unseen domains under a matched optimization budget. That improvement remains a hypothesis to test.

The project documents define performance as source accuracy, OOD accuracy, worst-domain accuracy, transfer across models, and optimization cost. They also require target domains to remain unseen until final evaluation. Context: [research proposal](C:/Users/alber/NLPcapstone/NLP_271A_Project_Proposal.pdf), [presentation](<C:/Users/alber/NLPcapstone/271A Capstone Proposal Presentation.pdf>), and [mentor brief](C:/Users/alber/NLPcapstone/BoytsovKumarTan.pdf).

**Prioritized techniques**

| Priority | Technique | Why it fits this project | Evidence and limits |
| --- | --- | --- | --- |
| 1 | Official GEPA with its full candidate selection and reflection machinery | Provides a credible modern baseline and replaces the current greedy reflection approximation. | The [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0e9e708b6f48e14fd0ac29e167413f76-Abstract-Conference.html) reports strong results across six tasks. Those results do not establish OOD sentiment performance. |
| 2 | ESPO-inspired diagnosis and stable selection | Addresses noisy selection and accumulation of overly specific prompt rules. | The [September 3, 2026 paper](https://arxiv.org/abs/2609.04197) reports +3.76 percentage points over GEPA on seven benchmarks and 47% shorter prompts. Its main comparison starts from deliberately weak prompts. |
| 3 | Worst-group selection on permitted source domains | Directly aligns search with the project's robustness question. | [DRO-InstructZero](https://arxiv.org/abs/2510.15260) motivates robust objectives using distributionally robust Bayesian optimization. A simple minimum-over-groups score would be our adaptation, not a reproduction of that algorithm. |
| 4 | Allocate evaluation effort adaptively | Gives promising candidates more of a limited budget instead of repeatedly scoring every candidate on every example. | [TRIPLE, NeurIPS 2024](https://github.com/ShenGroup/TRIPLE), implements fixed-budget best-arm identification. This is an established efficiency technique, rather than a newly released optimizer. |
| 5 | GEPA ConfidenceAdapter, after a multiclass task is available | Adds informative feedback about ambiguous and confidently wrong predictions. | A [March 2026 contributor benchmark](https://gepa-ai.github.io/gepa/blog/2026/03/17/confidence-adapter-benchmark/) reports +2.10 pp on AG News and +1.80 pp on Emotion versus the default adapter, but a tie on binary Rotten Tomatoes. It requires logprob support and is not evidence of OOD gains. |

The official [GEPA repository](https://github.com/gepa-ai/gepa) supports standalone adapters as well as DSPy. Its [February 2026 optimize_anything API](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/) allows a custom evaluator, separate training and validation data, and cached evaluations. This offers a practical integration route without rewriting the entire harness around DSPy.

For runtime, [July 2026 parallel proposals](https://gepa-ai.github.io/gepa/blog/2026/07/30/parallel-proposals/) are particularly relevant: the maintainers report roughly 3-4x lower wall time in their experiments. Benchmark this separately from quality because a local GPU may saturate, and parallel proposals also alter search. Start with exact caching and modest bounded evaluation concurrency; measure throughput instead of assuming a speedup.

**What the current code and results reveal**

The latest saved real Ollama multiseed file contains 25 runs, across five optimizers and five seeds. Twenty-four return the original sentiment instruction. Reflective has only one recorded candidate in all five runs, while OPRO has two. This shows that the saved experiments provide little evidence of substantial search; it does not establish that prompt optimization cannot help. See the [saved sweep](C:/Users/alber/NLPcapstone/prompt_opt_harness/logs/compare_real_ollama_multiseed_20260915T203141Z.json).

These logs predate some current working-tree changes. Findings about the present source are therefore distinguished from observations about historical runs.

| Issue | Verified location | Consequence and concrete change |
| --- | --- | --- |
| Final target examples are available during optimization | [harness.py:67](C:/Users/alber/NLPcapstone/prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/harness.py:67), [opro.py:13](C:/Users/alber/NLPcapstone/prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/optimizers/opro.py:13) | The harness passes `dataset.ood`; OPRO scores those examples and uses their scores for candidate ranking and final selection. Remove final-test data from optimizer inputs. Introduce distinct source training, source selection, source test, and target test containers. |
| GEPA is currently a simplified approximation | [gepa.py:10](C:/Users/alber/NLPcapstone/prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/optimizers/gepa.py:10) | It reflects on one random failure and keeps the best scalar-scoring candidate; it has no Pareto population or complementary candidate merging. Preserve it under a descriptive name and add an official GEPA adapter. The local TextGrad class is also a custom approximation; verify fidelity before using published method names in comparisons. |
| Evaluation can reuse a different input's prediction | [base.py:17](C:/Users/alber/NLPcapstone/prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/optimizers/base.py:17) | If sentence-transformers loads, cosine similarity above 0.92 causes prediction reuse. Similar sentiment/NLI inputs can have opposite labels. Use exact request caching only. Embeddings may help candidate diversity or error clustering, but should not substitute for predictions on distinct test examples. |
| Budgets are checked outside full evaluation batches | [base.py:79](C:/Users/alber/NLPcapstone/prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/optimizers/base.py:79), [gepa.py:14](C:/Users/alber/NLPcapstone/prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/optimizers/gepa.py:14) | A 30-example evaluation consumes about 30 model calls before another budget check. A budget of 40 permits very little search. Enforce the cap at each backend request and reserve enough budget to compare candidates on a common set. Do not rank partial-batch and full-batch scores as equivalent. |
| Search cost and final evaluation cost are mixed | [harness.py:74](C:/Users/alber/NLPcapstone/prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/harness.py:74), [llm_backends.py:91](C:/Users/alber/NLPcapstone/prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/llm_backends.py:91) | Counters are read after evaluation; a separate evaluation backend's costs are omitted. Retried slow successful requests are also excluded. Snapshot each phase, record all paid attempts, and separate proposer/scorer/final-test tokens and wall time. Total logged calls alone do not measure budget overshoot because they include evaluation. |
| Entropy is not being measured through controlled sampling | [entropy.py:9](C:/Users/alber/NLPcapstone/prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/optimizers/entropy.py:9) | Appending spaces changes the request; it does not set temperature or ensure independent samples. The real backend defaults are temperature zero. Use explicit sampling configuration and canonical parsed labels, or actual class probabilities. Low entropy alone can favor confidently wrong prompts. |
| The current DRO wrapper does not model domain shift | [dro.py:9](C:/Users/alber/NLPcapstone/prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/optimizers/dro.py:9) | It takes the minimum over random subsets, with subsets dependent on Python's process-dependent string hash. Fix groups and resamples across candidates, use stable seeds, and use actual source domain metadata for domain-robust experiments. |
| Thirty-example evaluation is too coarse for small gains | [real_datasets.py:27](C:/Users/alber/NLPcapstone/prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/real_datasets.py:27) | One changed prediction is 3.33 pp. Increase final held-out samples and use paired uncertainty estimates. The loader is not stratified, truncates Amazon text to 500 characters, and draws both SST-2 source splits from the 872-example validation set; increasing `n_per_split` alone is insufficient. |
| Task identity is inferred from mutable prompt text | [prompts.py:4](C:/Users/alber/NLPcapstone/prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/prompts.py:4) | If an NLI instruction loses the word `entailment`, the formatter requests sentiment labels. Move task labels and output parsing into an explicit task specification. Preserve the three-label schema when adding real MultiNLI. |

The oracle also needs separate target dev and target test splits: the present implementation combines source and OOD examples into optimization data. A target-trained reference is not a mathematical upper bound, even when the split is corrected.

**The extension I would build**

Use the official GEPA adapter as the base, then add a separate experimental mode with the following protocol. These are proposed design choices for this project, not claims of a reproduced published method.

1. Freeze source training and source selection IDs before search. Keep source and target final tests inaccessible to the optimizer, reflection model, and any hyperparameter search.
2. Cache a matrix of exact candidate-by-example predictions and correctness. Include model revision, decoding settings, full input, and prompt version in the key. For deliberate stochastic repeats, include a sample identifier and do not collapse samples.
3. Diagnose recurring source-training failure patterns and propose revisions, simplifications, and removal of harmful rules. For sentiment, investigate phenomena such as negation, mixed sentiment, and sarcasm when actually present in the training errors. Treat source-specific factual additions as a separate ablation.
4. Evaluate finalists on the same source selection examples. Resample the cached correctness matrix to estimate how stable candidate rankings are. This resampling adds no model calls for already collected predictions. It does not create new independent examples or prove robustness to unseen domains.
5. In a separately labeled multi-source experiment, use a fixed source-group objective such as `J(p) = (1-lambda) * mean_g accuracy_g(p) + lambda * min_g accuracy_g(p)`. Choose lambda using source-only validation and hold a whole target domain out. Start with a small predeclared comparison of mean selection and mean/minimum selection. Do not tune lambda on the reported target tests.
6. Prefer shorter prompts only when the selection scores are practically tied under a predeclared tolerance. The existing penalty of 0.005 per word subtracts 0.50 for a 100-word prompt, which can overwhelm useful accuracy improvements. Keep raw accuracy separate from any selection objective.
7. Freeze one prompt, then evaluate it unchanged on every final target and on the second model. Record actual OOD improvement, not just a smaller gap caused by worse source accuracy.

The [ESPO full text](https://arxiv.org/html/2609.04197v1) describes clustering errors, varied proposal strategies, and bootstrap selection. Its cross-model section reruns optimization for each student; it does not establish transfer of one frozen prompt between models. I did not verify an official downloadable ESPO implementation, so budget for a clearly labeled reimplementation or component ablation. Its claimed gains are motivation for an experiment, not a forecast.

**Evaluation that can establish an improvement**

Keep the primary experiment faithful to the original question: optimize on one source, evaluate on unseen targets. Report multi-source optimization as a different regime with the same source data available to every compared method. Artificial groups within SST-2 can test stability but are not equivalent to independent source domains.

Begin with sentiment, then add real MultiNLI genre transfer. [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) was designed to support cross-genre generalization. This is a cleaner second task than pooling HotpotQA-to-AIME results with same-task domain shift: the latter changes task requirements and answer format, and should be reported as a separate task-transfer stress test.

| Experiment | Comparison | What it answers |
| --- | --- | --- |
| A | Minimal task instruction, fixed human prompt, random candidate search, corrected OPRO, official GEPA | Does intelligent search beat simple controls under equal resources? |
| B | GEPA versus GEPA with stable final selection | Does selection stability alone help? |
| C | GEPA with structured diagnosis versus B, followed by the combined method | Does better error feedback help, and do the two changes interact? |
| D | Mean source-group objective versus mean/minimum objective, using the same permitted domains | Does explicit source-domain robustness transfer to an unseen domain? |
| E | Freeze each prompt optimized for model A and evaluate on A and B | Does the learned instruction transfer between models? |
| F | Serial versus bounded parallel execution with comparable search settings | How much wall time can be saved, and does search behavior change? |

Use five optimizer seeds after an inexpensive integration pilot. Keep data-split seeds separate from optimizer seeds so variability has an interpretable cause. Within each split, all methods must use identical examples, task models, parsing, truncation policy, and proposer resources. Start with a predeclared call cap such as 2,000 optimization requests per run, counting both proposal and scoring calls, then size it from pilot costs. Report token usage by model alongside calls; equal calls do not imply equal compute or dollars. Freeze the real budget before looking at final target results.

Aim for 500-1,000 final examples per target where available, and preserve source training/selection/test separation. Treat this as a planning range, not a power guarantee for a 1-2 pp improvement. Report per-domain accuracy and macro-F1, mean OOD accuracy, minimum OOD accuracy, source accuracy, prompt token count, optimization tokens, and inference latency. Use paired resampling of the same example IDs for method differences, with seed variability reported separately; repeated runs on the same test examples do not multiply the number of independent test examples.

Retain the original OOD gap, but add a baseline-adjusted transfer measure:

`excess_source_gain = [acc_source(p) - acc_source(p0)] - mean_d[acc_target_d(p) - acc_target_d(p0)]`

Here `p0` is the fixed baseline instruction. This helps distinguish dataset difficulty from gains that remain confined to the source. Success should require increased OOD accuracy and acceptable source accuracy at the matched budget; minimizing the raw OOD gap by itself is insufficient. Predeclare the allowable source regression and desired practical OOD improvement before final evaluation.

**Implementation order**

First change the data contract, exact evaluation cache, budget guard, and phase-specific accounting in the harness, optimizer base, dataset loader, and backend. Add focused checks that target examples cannot reach optimization, final tests are disjoint, the cap cannot be exceeded, distinct inputs are evaluated separately, and task labels survive prompt edits.

Next add the official GEPA adapter, pin its tested version, and log enough configuration to reproduce its settings. Keep reference implementations and project modifications under distinct method names. If adding MIPROv2 or SIMBA from [DSPy](https://dspy.ai/3.2.0/learn/optimization/optimizers/), match the search space: compare instruction-only modes first, or report few-shot demonstration optimization separately. Exporting only the instruction would discard learned demonstrations and change the evaluated program.

Then add cached stable selection and structured error diagnosis as independent switches, followed by the source-group objective. Complete the sentiment ablations before expanding to additional tasks or many optimizer variants.

Attention concentration remains a useful later white-box experiment. [Concentrate Attention, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/061d5d1b7d97117764f205d4e038f9eb-Abstract-Conference.html) reports improvements to prompt generalization, but needs access to internal attention information that the current text-only API abstraction does not expose. It requires a different instrumentation path and is lower priority than a reliable black-box comparison.

The deliverable from this review is a ranked, implementable research plan. Establishing an actual performance increase requires the corrected experiments above.
