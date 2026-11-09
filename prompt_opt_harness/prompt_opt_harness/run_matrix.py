"""Run every optimizer on every dataset (and seed), then build the results matrix.

Small first run:
    python run_matrix.py --backend groq --n-per-split 10 --budget 50 --seeds 0

Outputs (in results/):
    matrix_rows.csv  one row per (dataset, method, seed) run, written as runs finish
    matrix.md        method x dataset tables: mean OOD accuracy, OOD gap, API calls
"""
from __future__ import annotations

import argparse
import csv
import os
from statistics import mean, pstdev

from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import GroqBackend, OllamaBackend
from prompt_opt_harness.logger import save_run
from prompt_opt_harness.optimizers import (
    GEPA, OPRO, EvoPromptLite, HumanWrittenBaseline, HumanWrittenNLIBaseline,
    HumanWrittenReasoningBaseline, TextGrad, ZeroShotBaseline,
)
from prompt_opt_harness.real_datasets import load_real_dataset, load_real_nli_dataset
from prompt_opt_harness.real_datasets_reasoning import load_real_reasoning_dataset
from prompt_opt_harness.tasks import NLI_TASK, REASONING_TASK, SENTIMENT_TASK

# One entry per matrix column: loader, task, and that task's human-written baseline.
DATASETS = {
    "sentiment": (load_real_dataset, SENTIMENT_TASK, HumanWrittenBaseline),
    "reasoning": (load_real_reasoning_dataset, REASONING_TASK, HumanWrittenReasoningBaseline),
    "nli": (load_real_nli_dataset, NLI_TASK, HumanWrittenNLIBaseline),
}

# One entry per matrix row (baselines are added automatically).
OPTIMIZERS = {
    "OPRO": OPRO,
    "EvoPrompt": EvoPromptLite,
    "GEPA": GEPA,
    "TextGrad": TextGrad,
}

DEFAULT_MODEL = {"groq": "openai/gpt-oss-20b", "ollama": "llama3.1"}

# Which numbers become tables in matrix.md: (column in the CSV, table title, format).
MATRIX_METRICS = [
    ("mean_ood_acc", "Mean OOD accuracy (higher is better)", "{:.2f}"),
    ("ood_gap", "OOD gap = source test acc - mean OOD acc (lower is better)", "{:+.2f}"),
    ("worst_case_acc", "Worst-case OOD accuracy (higher is better)", "{:.2f}"),
    ("api_calls", "API calls (cost)", "{:.0f}"),
]

CSV_FIELDS = [
    "dataset", "method", "seed", "source_test_acc", "mean_ood_acc", "ood_gap",
    "worst_case_acc", "variance", "api_calls", "input_tokens", "output_tokens",
    "wall_time", "ood_accs", "best_instruction",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["groq", "ollama"], default="groq")
    p.add_argument("--model", default=None)
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--n-per-split", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--optimizers", nargs="+", default=list(OPTIMIZERS), choices=list(OPTIMIZERS))
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def make_backend_factory(args):
    model = args.model or DEFAULT_MODEL[args.backend]
    if args.backend == "ollama":
        return lambda: OllamaBackend(model=model)
    return lambda: GroqBackend(model=model)


def to_row(result, dataset: str, seed: int) -> dict:
    ood_values = list(result.ood_accs.values())
    return {
        "dataset": dataset,
        "method": result.name,
        "seed": seed,
        "source_test_acc": result.source_test_acc,
        "mean_ood_acc": mean(ood_values) if ood_values else 0.0,
        "ood_gap": result.ood_gap,
        "worst_case_acc": result.worst_case_acc,
        "variance": result.variance,
        "api_calls": result.api_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "wall_time": round(result.wall_time, 2),
        "ood_accs": ";".join(f"{d}={a:.2f}" for d, a in result.ood_accs.items()),
        "best_instruction": result.best_instruction,
    }


def run_all(args, backend_factory) -> list[dict]:
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "matrix_rows.csv")
    rows = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for ds_name in args.datasets:
            loader, task, human_baseline = DATASETS[ds_name]
            # baselines spend 1 "search" call; optimizers get the full budget
            methods = [("Zero-shot", ZeroShotBaseline, 1), ("Human-written", human_baseline, 1)]
            methods += [(name, OPTIMIZERS[name], args.budget) for name in args.optimizers]
            for seed in args.seeds:
                dataset = loader(n_per_split=args.n_per_split, seed=seed)
                for name, cls, budget in methods:
                    print(f"[{ds_name} | seed {seed}] {name} ...", flush=True)
                    try:
                        result = run_experiment(name, cls, backend_factory, dataset,
                                                budget=budget, seed=seed, task=task)
                    except Exception as e:  # one failed run shouldn't kill the whole matrix
                        print(f"  FAILED: {type(e).__name__}: {e}")
                        continue
                    save_run(result, log_dir=os.path.join("logs", ds_name))
                    row = to_row(result, ds_name, seed)
                    rows.append(row)
                    writer.writerow(row)
                    f.flush()  # keep finished runs on disk even if a later one crashes
                    print(f"  src={row['source_test_acc']:.2f} ood={row['mean_ood_acc']:.2f} calls={row['api_calls']}")
    print(f"\nPer-run rows: {csv_path}")
    return rows


def build_matrix_md(rows: list[dict], datasets: list[str]) -> str:
    methods = list(dict.fromkeys(r["method"] for r in rows))  # keep run order
    n_seeds = len({r["seed"] for r in rows})
    lines = [f"# Results matrix\n\n{n_seeds} seed(s); cells are the mean over seeds"
             + (" ± std." if n_seeds > 1 else ".") + "\n"]
    for metric, title, fmt in MATRIX_METRICS:
        lines.append(f"## {title}\n")
        lines.append("| Method | " + " | ".join(datasets) + " |")
        lines.append("|---|" + "---|" * len(datasets))
        for m in methods:
            cells = []
            for d in datasets:
                vals = [float(r[metric]) for r in rows if r["method"] == m and r["dataset"] == d]
                if not vals:
                    cells.append("–")
                elif len(vals) == 1:
                    cells.append(fmt.format(vals[0]))
                else:
                    cells.append(f"{fmt.format(mean(vals))} ± {pstdev(vals):.2f}")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = run_all(args, make_backend_factory(args))
    md = build_matrix_md(rows, args.datasets)
    md_path = os.path.join(args.out_dir, "matrix.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(md)
    print(f"Matrix: {md_path}")


if __name__ == "__main__":
    main()
