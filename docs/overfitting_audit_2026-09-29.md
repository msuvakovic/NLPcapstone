**Overfitting audit: confirmed issues and pending real-model evidence**

**October 4 update:** the original harness's target-feedback path described below has now been repaired. OPRO rejects target inputs; the harness enforces split separation; the old oracle is disabled. Current regression results and cached real-response replay are documented in [the root README](../README.md) and [log.txt](../log.txt). The remainder records the September 29 audit; the complete ESPO-inspired final evaluation is still unfinished.

The repository has two confirmed reasons to be cautious about apparent overfitting. Neither alone establishes how much a real optimized prompt overfits.

1. **The original evaluation exposes final target labels during search.** `harness.py` passes `dataset.ood` into `optimizer.optimize`, then reports final accuracy on those same examples. OPRO evaluates candidates on the supplied targets and uses their scores for ranking. The offline regression in `tools/prompt_bench/leakage_check.py` holds source scores fixed and changes only target feedback: the selected prompt changes. With target feedback removed, the selection stays fixed. This is a control-flow test with artificial scores, not a model-accuracy experiment.
2. **The September 21 individual logs reproduce simulated behavior.** All 1,186 classification traces across 20 saved runs exactly match the current `MockBackend` on the toy dataset. The backend deliberately gives movie-domain instruction fragments an accuracy bonus only in the source domain. The repository README identifies these as plumbing checks. Their source/target gaps cannot establish real-model overfitting. This finding does not apply automatically to the separate September 15 real Ollama sweep.

The separate September 15 real Ollama sweep returned the unchanged base instruction in 24 of 25 runs. Source/OOD gaps for those unchanged prompts cannot be attributed to new rules learned by prompt optimization. Those results provide little evidence of optimization-induced overfitting, given the limited search and small evaluation samples.

The new real-model pilot uses disjoint source training, source validation, held-out source, and held-out target examples. Single-source optimization receives SST-2 only. A separately labeled multi-source experiment receives SST-2 and Amazon, with tweets held out. Every selected prompt is frozen before final scores are examined. Exact normalized-text duplicates are removed across all splits.

Three diagnostics will be reported for each selected prompt:

| Diagnostic | Calculation | Interpretation |
| --- | --- | --- |
| Validation gain | Selected-prompt validation accuracy minus base-prompt validation accuracy | What candidate selection appears to achieve on its reused selection set |
| Selection optimism | Validation gain minus gain on untouched examples from the same source distribution | A positive value indicates validation improvement did not fully survive; baseline adjustment helps account for split difficulty |
| Excess source gain | Gain on untouched source examples minus gain on untouched OOD examples | A positive value suggests the benefit is concentrated in the source domain |

These are descriptive warning signs. Raw validation/source or source/OOD accuracy differences can arise from sampling, label conventions or domain difficulty. The 64-example validation set moves by 1.56 percentage points per changed answer. Three optimization seeds reuse one data split and are not independent dataset replications. Final-test paired bootstrap intervals are conditional on the selected prompts; they do not quantify uncertainty over new search seeds or partitions.

The benchmark additionally checks whether instructions copy eight-token spans from allowed source examples. Absence of such matches does not exclude semantic memorization, and model pretraining contamination cannot be determined from these experiments.

During integration, independent rescoring caught a separate GEPA cache-ID collision: default list loaders numbered training and validation examples from zero, while the library cache keyed scores by candidate and example ID. Globally unique hashed IDs now separate the splits, and every official validation vector must equal independently checked predictions. A regression with opposite training and validation outcomes tests this boundary. The affected exploratory runs were archived before final testing.

The isolated runner corrects these evaluation issues for the new experiment. The original project harness remains unchanged, so its existing entry points still need the target-feedback fix before they can support unseen-domain claims.

Evidence artifacts are in `output/prompt_benchmark_2026-09-29/leakage_regression.json`, `historical_log_audit.json`, and `execution_notes.json`. Once the real-model workflow finishes, `report.md` and `overfitting.json` in that directory will contain the measured findings.
