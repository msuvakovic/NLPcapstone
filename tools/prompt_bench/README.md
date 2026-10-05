**Controlled prompt-optimization pilot**

**Gemma output-cap experiment (October 4):** `token_caps.py` tested classification caps 1/2/4/8/16 on 24 balanced source-training examples, then confirmed caps 1/4/16 on 64 separate source-validation examples. Cap 1 matched every cap-16 prediction with no invalid labels, using one output token instead of two. Accuracy was 96.88% and macro F1 0.9687 for all three confirmation caps. Cap 4 was the lowest tested cap with natural stopping; cap 3 was not tested. Cap 1 reported `finish_reason=length` but returned complete labels. This reduced total input-plus-output tokens by only 1.54%; no latency or billed-cost benefit was established.

See the [cap report](../../output/token_caps_2026-10-04/report.md) and [scoped settings](../../output/token_caps_2026-10-04/recommended_settings.json). For the exact tested base-prompt classifier, the call is `client.complete(task_prompt(BASE, text), max_tokens=1)` followed by strict label validation. Proposal/reflection caps and frozen benchmark defaults remain unchanged. These small source-only results do not establish OOD or rare-failure behavior.

Run `python -B tools/prompt_bench/test_token_caps.py` for offline checks. `python -u -B tools/prompt_bench/token_caps.py` resumes the fixed experiment, making model requests only for missing work. The fresh run used 312 requests with no failures. It has a 400-attempt safety cap per execution; completed rows and exact responses are checkpointed.

This separate runner tests repository OPRO with exact source-only evaluation, official GEPA 0.1.4, and an explicitly named ESPO-inspired adaptation. Mean, bootstrap-stability and worst-source-group selectors are separate candidate re-ranking ablations. The original harness's target-feedback path was also fixed on October 4, 2026; see the [root README](../../README.md).

The later harness reliability fixes add exact scoring, visible retry accounting and a hard OPRO attempt cap. This benchmark's OPRO adapter delegates batch reservation to `Evaluator`, retaining the frozen protocol's separate evaluation and proposal limits. An offline replay after those changes matched all three saved searches and final predictions. Use `python -B tools/prompt_bench/replay_opro.py --output output/harness_reliability_2026-10-04/replay` to keep that verification separate from the earlier leakage-fix artifacts.

**Current status:** all 12 historical searches are complete; 5 of 8 distinct final prompts have complete saved evaluations. ESPO-inspired final results are incomplete. Full `verify.py`, `analyze.py` and plotting require the remaining predictions; `verify.py --search-only` checks the completed searches. The new runner rejects automatic resumption of the historical unversioned experiment. Its completion needs an explicitly reviewed migration or its original implementation. No unfinished experiment was resumed for these fixes.

**Version-2 protocol guard:** prepare/search writes a protocol once, with model, endpoint, decoding, budgets, methods, seeds, workers, source-code hashes and a validated split fingerprint. All execution stages compare the complete configuration before constructing the client; existing search/final result files must carry its `protocol_hash`. Different configurations require a new directory. Legacy manifests are preserved without automatic upgrade; read-only verification and zero-request replay remain available. `latency.py` checks the same execution context, and `workflow.py` validates preparation before later steps. Cloud model aliases are not immutable weight revisions; the configuration lock cannot prevent a provider from changing the model behind an alias.

Run `python -B tools/prompt_bench/test_protocol.py` for eleven offline tests, including a fake-client search/final/resume cycle. Start future model runs in a new directory such as `output/prompt_benchmark_v2`; keep the September pilot as historical evidence.

`python -B tools/prompt_bench/replay_opro.py` is an offline alternative: it replays all three repaired OPRO searches and their 900-example final evaluations from the local exact response cache, enforces zero physical calls, and checks equality with the earlier clean results. Outputs are in `output/leakage_fix_2026-10-04/`. This is a real-response replay, not fresh inference; identical selected prompts reuse the same final responses.

Use Python 3.13 with NumPy, PyArrow and Matplotlib. Install `gepa==0.1.4` in the workspace `.benchmark_deps` directory, or make it available on the Python path. There are no API keys in this runner: it uses the already authenticated Ollama service at `http://localhost:11434/v1` and the configured project model `gemma4:31b-cloud`.

On this computer the working interpreter is `C:\Python313\python.exe`; the existing `.venv` references an unavailable interpreter. For a future API-enabled run, use `& 'C:\Python313\python.exe' -X utf8 -u -B tools/prompt_bench/workflow.py --output output/prompt_benchmark_v2` in PowerShell. This new full workflow has not been run against the model. The workflow enables UTF-8 for its child processes, and GEPA logs use explicit UTF-8 encoding.

From the repository root:

```powershell
python -B tools/prompt_bench/test_benchmark.py
python -B tools/prompt_bench/test_analysis.py
python -B tools/prompt_bench/test_protocol.py
python -B tools/prompt_bench/run.py --stage prepare --output output/prompt_benchmark_v2
python -u -B tools/prompt_bench/run.py --stage search --workers 4 --output output/prompt_benchmark_v2
python -u -B tools/prompt_bench/run.py --stage evaluate --workers 4 --output output/prompt_benchmark_v2
python -u -B tools/prompt_bench/latency.py --workers 1 2 3 4 --output output/prompt_benchmark_v2
python -B tools/prompt_bench/verify.py --output output/prompt_benchmark_v2
python -B tools/prompt_bench/analyze.py --output output/prompt_benchmark_v2
python -B tools/prompt_bench/plot.py --output output/prompt_benchmark_v2
```

Alternatively, `python -u -B tools/prompt_bench/workflow.py --output output/prompt_benchmark_v2` runs these stages in order, audits the artifacts and produces the report and figure once evaluation is complete. It resumes only a matching version-2 protocol. `test_benchmark.py` includes an offline regression that checks OPRO rejects target feedback. Its artificial scores verify control flow, not accuracy. The original pre-fix reproduction remains preserved in the September 29 output directory; current regression output goes to the October 4 fix directory.

Run the latency stage by itself, without concurrent search or evaluation. Completed search runs and final predictions are resumable. Exact request caching also preserves completed calls if a batch is interrupted. Cached responses are snapshots, not fresh model replications.

The latency matrix uses 24 source requests per block, worker counts 1/2/3/4, and two repeats in forward/reverse order. It checkpoints completed blocks. Intermediate worker counts were added before timing measurements after cloud timeouts during final evaluation. Accuracy requests were stopped during timing, and all accuracy comparisons retain their frozen prompts and decoding settings.

Application caching is disabled for those timing blocks. Provider-side prompt caching and service load are not controlled. Reported input token counts include provider-cached prompt tokens; they do not establish billed cost.

Default protocol: three optimization seeds, 64 source training examples, 64 source validation examples, 300 final examples per domain. Each optimizer receives at most 512 distinct candidate/example evaluations and 16 proposal calls. Count cached evaluations against each run's logical budget so one method does not gain a search advantage from execution order. Record tokens separately because equal call caps do not imply equal cost. GEPA has an additional conservative stop condition that reserves space for a complete validation evaluation.

The single-source regime holds both Amazon and tweets out of optimization. The separate multi-source regime gives GEPA balanced SST-2/Amazon training and validation examples; tweets remain unseen. Never pool these regimes as though they used the same information.

The dependency folder and response cache are local execution artifacts. The scripts, frozen split manifest, configurations and JSON/Markdown results are the reproducibility artifacts. The benchmark does not use embedding similarity to reuse predictions, alter label schemas based on instruction wording, or treat final target labels as optimization feedback.

The report also measures overfitting: validation improvement minus improvement on untouched source data, and source improvement minus OOD improvement, each relative to the base instruction. These descriptive gaps help distinguish selection optimism from baseline split difficulty. Three optimizer seeds share one data partition; they are not independent dataset replications. A raw source/OOD accuracy gap alone is not proof of overfitting.
