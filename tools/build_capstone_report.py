from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "capstone_results" / "controlled_sentiment"
SOURCE_DIR = RESULT_DIR / "source"
HARNESS_DIR = ROOT / "prompt_opt_harness" / "prompt_opt_harness"
LOG_DIR = HARNESS_DIR / "logs"

ALBERT_COMMIT = "98a911810a5defcbd772af22035ed46b08274560"
APINERO_BASE_COMMIT = "0af8f3eee7b7ed952486c9e7ab63287becd7090f"
REPORT_DATE = "2026-10-18"

SOURCE_ARTIFACTS = {
    "output__prompt_benchmark_2026-09-29__protocol.json": "output/prompt_benchmark_2026-09-29/protocol.json",
    "output__prompt_benchmark_2026-09-29__single_opro_0__search.json": "output/prompt_benchmark_2026-09-29/single_opro_0/search.json",
    "output__prompt_benchmark_2026-09-29__single_opro_1__search.json": "output/prompt_benchmark_2026-09-29/single_opro_1/search.json",
    "output__prompt_benchmark_2026-09-29__single_opro_2__search.json": "output/prompt_benchmark_2026-09-29/single_opro_2/search.json",
    "output__leakage_fix_2026-10-04__replay_summary.json": "output/leakage_fix_2026-10-04/replay_summary.json",
    "output__leakage_fix_2026-10-04__validation.json": "output/leakage_fix_2026-10-04/validation.json",
    "output__improvement_pilot_2026-10-04__summary.json": "output/improvement_pilot_2026-10-04/summary.json",
    "output__improvement_pilot_2026-10-04__confirmation.json": "output/improvement_pilot_2026-10-04/confirmation.json",
    "output__improvement_pilot_2026-10-04__execution.json": "output/improvement_pilot_2026-10-04/execution.json",
    "output__improvement_pilot_2026-10-04__protocol.json": "output/improvement_pilot_2026-10-04/protocol.json",
    "output__improvement_pilot_2026-10-04__verification.json": "output/improvement_pilot_2026-10-04/verification.json",
    "output__round_sweep_2026-10-13__summary.json": "output/round_sweep_2026-10-13/summary.json",
    "output__round_sweep_2026-10-13__execution.json": "output/round_sweep_2026-10-13/execution.json",
    "output__round_sweep_2026-10-13__protocol.json": "output/round_sweep_2026-10-13/protocol.json",
    "output__round_sweep_2026-10-13__verification.json": "output/round_sweep_2026-10-13/verification.json",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def source(name: str) -> dict:
    return load_json(SOURCE_DIR / name)


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def signed_pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def population_variance(values: list[float]) -> float:
    if not values:
        return 0.0
    center = mean(values)
    return mean((value - center) ** 2 for value in values)


def qwen_reasoning_rows() -> list[dict]:
    files = [
        "zero-shot_20261009T004302Z.json",
        "human-written_20261009T004415Z.json",
        "opro_20261009T004949Z.json",
        "evoprompt-lite_20261009T005506Z.json",
        "gepa_20261009T010718Z.json",
        "miprov2_20261009T010937Z.json",
    ]
    rows = []
    for filename in files:
        run = load_json(LOG_DIR / filename)
        ood = list(run["ood_accs"].values())
        rows.append(
            {
                "artifact": f"prompt_opt_harness/prompt_opt_harness/logs/{filename}",
                "method": run["name"],
                "instruction": run["best_instruction"],
                "dev_accuracy": run["dev_score"],
                "source_test_accuracy": run["source_test_acc"],
                "ood_accuracy": mean(ood),
                "worst_ood_accuracy": min(ood),
                "ood_variance": population_variance(ood),
                "ood_gap": run["ood_gap"],
                "optimization_calls": run["optimization_calls"],
                "optimization_budget": run["optimization_budget"],
                "evaluation_calls": run["evaluation_calls"],
                "total_calls": run["api_calls"],
                "wall_time_sec": run["wall_time_sec"],
            }
        )
    return rows


def controlled_opro() -> dict:
    replay = source("output__leakage_fix_2026-10-04__replay_summary.json")
    if replay.get("result") != "PASS":
        raise ValueError("controlled OPRO replay did not pass")
    runs = replay["runs"]
    prompts = {run["selected_prompt"] for run in runs}
    accuracies = {json.dumps(run["accuracy"], sort_keys=True) for run in runs}
    if len(prompts) != 1 or len(accuracies) != 1:
        raise ValueError("controlled OPRO seeds did not freeze the same final result")

    accuracy = runs[0]["accuracy"]
    ood_values = [accuracy["amazon"], accuracy["tweets"]]
    search_rows = []
    for seed in range(3):
        search = source(
            f"output__prompt_benchmark_2026-09-29__single_opro_{seed}__search.json"
        )
        if search["original_selection"] != runs[seed]["selected_prompt"]:
            raise ValueError(f"seed {seed} replay/search selection mismatch")
        search_rows.append(
            {
                "seed": seed,
                "selected_prompt": search["original_selection"],
                "source_validation_accuracy": runs[seed]["source_validation_accuracy"],
                "logical_example_evaluations": search["resources"][
                    "unique_example_evaluations"
                ],
                "proposal_calls": search["resources"]["proposal_calls"],
                "logical_input_tokens": search["resources"]["logical_input_tokens"],
                "logical_output_tokens": search["resources"]["logical_output_tokens"],
                "physical_calls": search["physical_resources"]["actual_calls"],
                "failed_attempts": search["physical_resources"]["failed_attempts"],
                "cache_hits": search["physical_resources"]["cache_hits"],
                "physical_input_tokens": search["physical_resources"]["input_tokens"],
                "physical_output_tokens": search["physical_resources"]["output_tokens"],
                "wall_time_sec": search["wall_seconds"],
            }
        )

    return {
        "model": replay["model"],
        "prompt": runs[0]["selected_prompt"],
        "search_seeds": [run["seed"] for run in runs],
        "source_validation_accuracy": runs[0]["source_validation_accuracy"],
        "source_test_accuracy": accuracy["sst2"],
        "amazon_accuracy": accuracy["amazon"],
        "tweets_accuracy": accuracy["tweets"],
        "mean_ood_accuracy": mean(ood_values),
        "worst_ood_accuracy": min(ood_values),
        "ood_variance": population_variance(ood_values),
        "ood_gap": accuracy["sst2"] - mean(ood_values),
        "same_prompt_all_seeds": True,
        "shared_final_predictions": True,
        "searches": search_rows,
        "replay_transport": replay["transport"],
    }


def improvement_pilot() -> dict:
    summary = source("output__improvement_pilot_2026-10-04__summary.json")
    confirmation = source("output__improvement_pilot_2026-10-04__confirmation.json")
    execution = source("output__improvement_pilot_2026-10-04__execution.json")
    verification = source("output__improvement_pilot_2026-10-04__verification.json")
    if verification.get("status") != "PASS":
        raise ValueError("improvement pilot verification did not pass")

    methods = []
    for row in summary["methods"]:
        ood_values = [row["amazon"]["accuracy"], row["tweets"]["accuracy"]]
        methods.append(
            {
                "method": row["method"],
                "source_accuracy": row["sst2"]["accuracy"],
                "amazon_accuracy": row["amazon"]["accuracy"],
                "tweets_accuracy": row["tweets"]["accuracy"],
                "mean_ood_accuracy": row["ood_accuracy"],
                "worst_ood_accuracy": row["worst_ood_accuracy"],
                "ood_variance": population_variance(ood_values),
                "ood_gap": row["sst2"]["accuracy"] - row["ood_accuracy"],
                "ood_delta": row["ood_delta"],
                "ood_delta_ci95": row["ood_delta_ci95"],
                "invalid_predictions": row["invalid"],
                "mean_input_tokens": row["mean_input_tokens"],
                "prompt_words": row["prompt_words"],
            }
        )

    searches = {}
    for name, row in summary["searches"].items():
        searches[name] = {
            **row["resources"],
            "physical_resources": row["physical_resources"],
            "wall_seconds": row["wall_seconds"],
        }

    return {
        "protocol_hash": summary["protocol_hash"],
        "methods": methods,
        "confirmation": confirmation["decisions"],
        "searches": searches,
        "execution": execution,
        "verification": verification,
    }


def round_sweep() -> dict:
    summary = source("output__round_sweep_2026-10-13__summary.json")
    execution = source("output__round_sweep_2026-10-13__execution.json")
    verification = source("output__round_sweep_2026-10-13__verification.json")
    if not verification.get("passed"):
        raise ValueError("round-sweep verification did not pass")
    return {
        "protocol_hash": summary["protocol_hash"],
        "selected_cap": summary["selected_cap"],
        "curve": summary["curve"],
        "final": summary["final"],
        "default_prompt": summary["default"],
        "conclusion_scope": summary["conclusion_scope"],
        "execution": execution,
        "verification": verification,
    }


def requirements() -> list[dict]:
    return [
        {
            "requirement": "Unified API for prompt optimizers",
            "status": "met",
            "evidence": "Shared Optimizer/Candidate interface and common harness across fixed prompts, Random Search, OPRO, EvoPrompt-lite, and DSPy adapters.",
        },
        {
            "requirement": "Optimize on a source development split under a fixed budget",
            "status": "met",
            "evidence": "BudgetedBackend enforces request caps; controlled sentiment searches use frozen source-only train/validation data and explicit evaluation/proposal caps.",
        },
        {
            "requirement": "Evaluate the same selected prompt on source test and OOD targets",
            "status": "met",
            "evidence": "Sentiment evaluates SST-2, Amazon, and tweets; reasoning evaluates GSM8K and SVAMP.",
        },
        {
            "requirement": "Reproducible logs and costs",
            "status": "met",
            "evidence": "Committed protocols, seeds, frozen selections, request/token accounting, verification records, and selected run logs.",
        },
        {
            "requirement": "Q1: quantify OOD gaps across optimizer settings",
            "status": "met_with_pilot_scope",
            "evidence": "Controlled OPRO, simple-procedure sentiment pilot, and six-method Qwen reasoning pilot report source/OOD scores and gaps.",
        },
        {
            "requirement": "Q2: compare variance and worst-case OOD robustness",
            "status": "met_for_sentiment",
            "evidence": "Two sentiment OOD domains support worst-case and cross-domain variance; the one-domain reasoning pilot cannot inform variance.",
        },
        {
            "requirement": "Q3: identify factors affecting generalization",
            "status": "partially_met",
            "evidence": "Optimizer/search procedure, prompt length, and optimization-round cost are analyzed; broad causal conclusions require more models and partitions.",
        },
        {
            "requirement": "Concrete recommendations and report",
            "status": "met",
            "evidence": "Generated capstone report and README summary provide scoped recommendations, limitations, and reproducible provenance.",
        },
        {
            "requirement": "Cross-model transfer",
            "status": "stretch_not_required",
            "evidence": "Exploratory branch work exists, but it is excluded from the main conclusions.",
        },
        {
            "requirement": "Code/formal tasks",
            "status": "stretch_not_required",
            "evidence": "Not implemented; the deck explicitly labels this stretch work.",
        },
    ]


def source_manifest() -> list[dict]:
    rows = []
    for local_name, original_path in SOURCE_ARTIFACTS.items():
        path = SOURCE_DIR / local_name
        if not path.exists():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        rows.append(
            {
                "local_path": str(path.relative_to(ROOT)),
                "source_branch": "origin/albert",
                "source_commit": ALBERT_COMMIT,
                "source_path": original_path,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return rows


def consolidated_protocol() -> dict:
    return {
        "version": "capstone-evidence-consolidation-v1",
        "generated_date": REPORT_DATE,
        "research_questions": {
            "q1": "How large is the OOD gap across optimizer settings?",
            "q2": "Which prompts look robust by worst-case accuracy and cross-domain variance?",
            "q3": "Which measured factors affect generalization?",
            "q4": "Stretch: can simple source-only procedures improve OOD performance?",
        },
        "primary_evidence": {
            "controlled_sentiment_opro": {
                "role": "main Q1/Q2 evidence",
                "model": "gemma4:31b-cloud",
                "search_seeds": [0, 1, 2],
                "source_train_n": 64,
                "source_validation_n": 64,
                "final_examples_per_domain": 300,
                "domains": ["sst2", "amazon", "tweets"],
                "search_evaluation_cap": 512,
                "proposal_cap": 16,
            },
            "simple_procedures": {
                "role": "Q3/Q4 component-ablation evidence",
                "model": "gemma4:31b-cloud",
                "search_seed_count": 1,
                "source_train_n": 64,
                "source_selection_n": 64,
                "source_confirmation_n": 80,
                "source_test_n": 64,
                "ood_examples_per_domain": 160,
            },
            "round_sweep": {
                "role": "Q3 cost-versus-round evidence",
                "model": "gemma4:31b-cloud",
                "search_seeds": [0, 1, 2],
                "round_caps": [0, 1, 2, 4, 8, 16],
            },
            "reasoning_pilot": {
                "role": "secondary cross-task consistency check",
                "model": "qwen2.5:7b",
                "seed": 42,
                "examples_per_split": 30,
                "domains": ["gsm8k", "svamp"],
            },
        },
        "inclusion_rules": [
            "Use only measured real-model artifacts with documented provenance.",
            "Exclude mock-backend accuracy from research conclusions.",
            "Exclude incomplete final evaluations and invalid/fallback runs.",
            "Separate controlled experiments rather than pool incompatible percentages.",
            "State when identical selected prompts share final predictions across seeds.",
        ],
        "consolidation_environment": {
            "new_model_calls": 0,
            "reason": "No local model server or weights and no provider API credentials were available.",
        },
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(result: dict) -> str:
    opro = result["controlled_sentiment"]["opro"]
    improvement = result["controlled_sentiment"]["simple_procedures"]
    rounds = result["controlled_sentiment"]["round_sweep"]
    qwen = result["reasoning_pilot"]["methods"]

    requirement_rows = [
        [row["requirement"], row["status"], row["evidence"]]
        for row in result["requirements"]
    ]
    qwen_rows = [
        [
            row["method"],
            pct(row["dev_accuracy"]),
            pct(row["source_test_accuracy"]),
            pct(row["ood_accuracy"]),
            signed_pp(row["ood_gap"]),
            f'{row["optimization_calls"]}/{row["optimization_budget"]}',
            str(row["total_calls"]),
        ]
        for row in qwen
    ]
    procedure_rows = [
        [
            row["method"],
            pct(row["source_accuracy"]),
            pct(row["amazon_accuracy"]),
            pct(row["tweets_accuracy"]),
            pct(row["mean_ood_accuracy"]),
            pct(row["worst_ood_accuracy"]),
            signed_pp(row["ood_delta"]),
            str(row["prompt_words"]),
        ]
        for row in improvement["methods"]
    ]
    search_rows = [
        [
            str(row["seed"]),
            str(row["logical_example_evaluations"]),
            str(row["proposal_calls"]),
            f'{row["logical_input_tokens"] + row["logical_output_tokens"]:,}',
            str(row["physical_calls"]),
            f'{row["wall_time_sec"]:.1f}',
        ]
        for row in opro["searches"]
    ]
    round_rows = [
        [
            str(row["rounds"]),
            pct(row["confirmation_accuracy"]),
            f'{row["mean_search_tokens"]:,.0f}',
        ]
        for row in rounds["curve"]
    ]

    return f"""# Capstone final evidence report

Generated on **{REPORT_DATE}** from committed real-model artifacts. No mock
backend scores are used as research results. The source PDF defines three main
deliverables: a unified optimizer API, reproducible logs/costs, and concrete
recommendations. It asks Q1–Q3 about OOD gaps, robustness, and factors affecting
generalization; Q4 and cross-model transfer are stretch goals.

## Executive conclusion

The engineering deliverables are complete. Across the strongest controlled
sentiment evidence, OPRO selected the unchanged base prompt in all three search
seeds, producing **{pct(opro['mean_ood_accuracy'])} mean OOD accuracy** and a
**{signed_pp(opro['ood_gap'])} source-minus-OOD gap**. A fresh simple-procedure
pilot found at most a **+0.31 pp** observed OOD gain for an unguarded rewrite,
while its independent confirmation guard rejected every replacement. A separate
round sweep increased mean logical search tokens from **0** at zero rounds to
**62,776** at 16 rounds without changing the selected prompt or final accuracy.

The six-method Qwen reasoning pilot likewise did not establish a reliable
optimizer advantage: the human-written prompt had the highest observed SVAMP
score, while OPRO and EvoPrompt-lite selected the original instruction. These
results support a scoped recommendation: **retain a strong base or human prompt
unless source-only confirmation establishes a gain, and always report target
transfer and search cost separately from development accuracy.**

## Slide-deck requirement status

{markdown_table(['Requirement', 'Status', 'Evidence'], requirement_rows)}

## Main controlled sentiment result: OPRO

Protocol: Gemma 4 31B cloud alias; three source-only OPRO search seeds; 64 source
training and 64 source-validation examples; 300 frozen final examples each from
SST-2, Amazon, and binary tweet sentiment; 512 candidate/example evaluations and
at most 16 proposals per seed. Final target data was not used during search.

All three seeds retained this prompt:

> {opro['prompt']}

| Source test | Amazon | Tweets | Mean OOD | Worst OOD | OOD gap | OOD variance |
| --- | --- | --- | --- | --- | --- | --- |
| {pct(opro['source_test_accuracy'])} | {pct(opro['amazon_accuracy'])} | {pct(opro['tweets_accuracy'])} | {pct(opro['mean_ood_accuracy'])} | {pct(opro['worst_ood_accuracy'])} | {signed_pp(opro['ood_gap'])} | {opro['ood_variance']:.6f} |

The identical selected prompt reuses the same frozen final predictions across
seeds; these are three search replications, not three independent final-test
replications.

### OPRO search costs

{markdown_table(['Seed', 'Logical evals', 'Proposals', 'Logical tokens', 'Physical calls', 'Wall sec'], search_rows)}

## Simple OOD procedures

This fresh one-seed Gemma 4 pilot compared the base and human prompts with three
custom history/feedback search procedures. These are component ablations, not
canonical OPRO, GEPA, or Random Search implementations.

{markdown_table(['Method', 'Source', 'Amazon', 'Tweets', 'Mean OOD', 'Worst OOD', 'OOD delta', 'Prompt words'], procedure_rows)}

The raw `history_full` rewrite had the largest observed OOD delta, **+0.31 pp**
with a paired 95% interval of **[0.00, +0.94] pp**. However, none of the three
raw winners improved the independent 80-example source confirmation set, so the
predeclared guard retained the base prompt in every arm. The full pilot used
**{improvement['execution']['last_session']['actual_calls']:,} requests** and
**{improvement['execution']['last_session']['input_tokens'] + improvement['execution']['last_session']['output_tokens']:,} returned tokens**.

## Optimization rounds as a generalization factor

{markdown_table(['Round cap', 'Confirmation accuracy', 'Mean logical search tokens/seed'], round_rows)}

All 0/1/2/4/8/16-round checkpoints across three seeds selected the same base
prompt. The frozen prompt scored **{pct(rounds['final']['base']['ood_accuracy'])}
mean OOD accuracy** and **{pct(rounds['final']['base']['worst_ood_accuracy'])}
worst-domain accuracy**. This supports a cost conclusion for this pilot—not a
universal round-count rule: extra rounds increased search cost without changing
the selected instruction.

## Qwen 2.5 7B reasoning pilot

Protocol: one seed; 30 GSM8K development, 30 GSM8K test, and 30 SVAMP examples;
temperature zero; 512-token cap. GEPA/MIPROv2 rows are instruction-transfer
adapters rather than full compiled-program evaluations.

{markdown_table(['Method', 'Dev', 'GSM8K', 'SVAMP', 'OOD gap', 'Opt calls/cap', 'Total calls'], qwen_rows)}

Because this pilot has one OOD domain, OOD variance is necessarily zero and
worst-case OOD equals SVAMP accuracy. It supports a cross-task consistency check,
not a broad robustness ranking.

## Answers to the research questions

### Q1: How large is the OOD gap across optimizer settings?

- Controlled sentiment OPRO/base: **{signed_pp(opro['ood_gap'])}**.
- Fresh simple-procedure rows range from **{signed_pp(min(row['ood_gap'] for row in improvement['methods']))}** to **{signed_pp(max(row['ood_gap'] for row in improvement['methods']))}** source-minus-mean-OOD.
- Qwen reasoning rows range from **{signed_pp(min(row['ood_gap'] for row in qwen))}** to **{signed_pp(max(row['ood_gap'] for row in qwen))}**, but the 30-example, one-seed pilot is too small for a stable method ranking.

### Q2: Which prompts look robust?

In the controlled sentiment replay, the unchanged base prompt achieved
**{pct(opro['worst_ood_accuracy'])} worst-domain accuracy** with very low
cross-domain variance. In the fresh procedure pilot, `history_full` had the best
observed worst-domain score, but failed independent source confirmation; the
guarded choice therefore remained the base prompt. Robustness claims should use
multiple target domains—SVAMP alone cannot estimate cross-domain variance.

### Q3: Which factors affect generalization?

The measured factors with the clearest evidence are:

1. **Search procedure and confirmation:** target gains from raw rewrites were
   small and were not supported by independent source confirmation.
2. **Optimization rounds:** more rounds sharply increased tokens without
   changing the selected prompt in the round-sweep pilot.
3. **Prompt specificity/length:** longer prompts did not consistently improve
   mean or worst-case OOD accuracy in the fresh procedure pilot.
4. **Task/domain choice:** sentiment supports variance/worst-case analysis with
   two targets; the one-target reasoning pilot does not.

These are pilot-level associations, not general causal laws across models.

## Recommendations

1. Use a strong fixed prompt and human-written prompt as mandatory controls.
2. Add Random Search to future live matrices so search intelligence is separated
   from simply evaluating more candidates; it is implemented on `apinero` but no
   fresh real-model run was possible in the current credential-free environment.
3. Select replacements using source-only confirmation before any target test is
   opened; retain the base prompt on ties or unconfirmed gains.
4. Report development, source-test, every target-domain score, worst-domain
   accuracy, cross-domain variance, requests, tokens, and elapsed time.
5. Do not assume more optimization rounds are beneficial; justify them with an
   independent confirmation gain.
6. Treat DSPy instruction extraction and full compiled-program evaluation as
   different experiments.

## Scope and limitations

- The strongest sentiment evidence uses one model alias and one frozen partition.
- Identical final prompts share exact predictions across search seeds.
- The simple-procedure pilot uses custom component ablations, not every optimizer
  in the unified harness.
- The Qwen reasoning comparison uses one seed and 30 examples per split.
- No new real-model calls were made while consolidating this report: the current
  environment has GPUs but no local model weights/server, no provider API keys,
  and no installed real-data/DSPy packages. Existing measured artifacts were
  consolidated rather than replacing them with mock results.
- Cross-model transfer and code/formal tasks remain stretch work.

## Provenance

- `apinero` base revision: `{APINERO_BASE_COMMIT}`.
- Controlled sentiment source revision: `origin/albert` at `{ALBERT_COMMIT}`.
- Compact source artifacts are committed under `capstone_results/controlled_sentiment/source/`.
- Qwen pilot artifacts remain under `prompt_opt_harness/prompt_opt_harness/logs/`.
- Rebuild with `python3 tools/build_capstone_report.py`.
"""


def build_result() -> dict:
    return {
        "generated_date": REPORT_DATE,
        "provenance": {
            "apinero_base_commit": APINERO_BASE_COMMIT,
            "controlled_sentiment_source_branch": "origin/albert",
            "controlled_sentiment_source_commit": ALBERT_COMMIT,
            "new_model_calls_during_consolidation": 0,
        },
        "requirements": requirements(),
        "controlled_sentiment": {
            "opro": controlled_opro(),
            "simple_procedures": improvement_pilot(),
            "round_sweep": round_sweep(),
        },
        "reasoning_pilot": {
            "model": "qwen2.5:7b",
            "seed": 42,
            "source_domain": "GSM8K",
            "ood_domains": ["SVAMP"],
            "examples_per_split": 30,
            "methods": qwen_reasoning_rows(),
        },
    }


def main() -> None:
    result = build_result()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (RESULT_DIR / "requirements.json").write_text(
        json.dumps(result["requirements"], indent=2) + "\n", encoding="utf-8"
    )
    (RESULT_DIR / "protocol.json").write_text(
        json.dumps(consolidated_protocol(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (RESULT_DIR / "source_manifest.json").write_text(
        json.dumps(source_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (RESULT_DIR / "report.md").write_text(build_markdown(result), encoding="utf-8")
    print(f"wrote {RESULT_DIR / 'summary.json'}")
    print(f"wrote {RESULT_DIR / 'requirements.json'}")
    print(f"wrote {RESULT_DIR / 'protocol.json'}")
    print(f"wrote {RESULT_DIR / 'source_manifest.json'}")
    print(f"wrote {RESULT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()