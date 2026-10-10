"""Which prompt traits predict OOD robustness? (research questions Q2/Q3)

Reads harness run logs, measures each selected prompt, and prints Spearman
correlations between traits and OOD gap / worst-case accuracy / variance.

    python -B analyze_traits.py logs/                       # traits from text only
    python -B analyze_traits.py logs/ --dataset demo        # + domain-vocabulary traits
    python -B analyze_traits.py logs/ --dataset real --out trait_report.md

``--dataset`` must be the dataset the logged runs used. It supplies the source
dev texts the optimizer saw and the OOD target texts, used only to measure
source-specific vocabulary and copied spans, never to score prompts.
"""
from __future__ import annotations

import argparse
import csv
import sys

from prompt_opt_harness.traits import correlations, format_markdown, load_runs, trait_table


def load_texts(name: str, n_per_split: int, seed: int):
    if name == "none":
        return None, None
    if name == "demo":
        from prompt_opt_harness.datasets import load_demo_dataset
        data = load_demo_dataset()
    elif name == "nli-demo":
        from prompt_opt_harness.datasets import load_nli_demo_dataset
        data = load_nli_demo_dataset()
    else:
        from prompt_opt_harness.real_datasets import load_real_dataset
        data = load_real_dataset(n_per_split=n_per_split, seed=seed)
    source = [ex.text for ex in data.source.dev]
    targets = [ex.text for examples in data.ood.values() for ex in examples]
    return source, targets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help="run log files or directories")
    parser.add_argument("--dataset", choices=("none", "demo", "nli-demo", "real"), default="none")
    parser.add_argument("--n-per-split", type=int, default=30, help="for --dataset real")
    parser.add_argument("--data-seed", type=int, default=0, help="for --dataset real")
    parser.add_argument("--exclude", nargs="*", default=["Zero-shot"],
                        help="run names to leave out (zero-shot has an empty prompt)")
    parser.add_argument("--out", help="write the markdown report here")
    parser.add_argument("--csv", help="write the per-run trait table here")
    args = parser.parse_args(argv)

    runs = [r for r in load_runs(args.logs) if r["name"] not in set(args.exclude)]
    if not runs:
        print("No run logs with OOD results found.", file=sys.stderr)
        return 1
    source, targets = load_texts(args.dataset, args.n_per_split, args.data_seed)
    rows = trait_table(runs, source_texts=source, target_texts=targets)
    report = format_markdown(rows, correlations(rows))
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
