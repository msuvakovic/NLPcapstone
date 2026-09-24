from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def find_latest_logs(logs_dir: Path):
    detailed = sorted(logs_dir.glob("compare_detailed_*.json"))
    summary = sorted(logs_dir.glob("compare_summary_*.json"))
    return (detailed[-1] if detailed else None, summary[-1] if summary else None)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def df_from_detailed(detailed):
    records = []
    for r in detailed:
        records.append({
            "name": r.get("name"),
            "optimizer": r.get("optimizer_class"),
            "budget": r.get("budget"),
            "seed": r.get("seed"),
            "src_acc": r.get("source_test_acc", 0.0),
            "ood_gap": r.get("ood_gap", 0.0),
            "cost": r.get("estimated_cost_usd", 0.0),
        })
    return pd.DataFrame.from_records(records)


def plot_mean_with_errorbars(df: pd.DataFrame, x_col: str, y_col: str, hue: list, facet_col: str | None, out_path: Path, ylabel: str, title: str):
    if facet_col:
        unique_facets = sorted(df[facet_col].unique())
        n = len(unique_facets)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, facet in zip(axes, unique_facets):
            sub = df[df[facet_col] == facet]
            _plot_by_optimizer(sub, x_col, y_col, hue, ax)
            ax.set_title(f"{facet_col}={facet}")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_by_optimizer(df, x_col, y_col, hue, ax)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_by_optimizer(df: pd.DataFrame, x_col: str, y_col: str, hue: list, ax):
    # group by optimizer and budget
    grp = df.groupby(["optimizer", "budget"])
    for (opt, budget), g in grp:
        x = g[x_col].values
        y = g[y_col].values
        # compute mean and std
        xm = np.mean(x)
        ym = np.mean(y)
        ys = np.std(y)
        label = f"{opt}@{budget}"
        ax.errorbar(xm, ym, yerr=ys, fmt='o', label=label)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')


def main():
    # Try a few likely logs locations (package-level and repo-level)
    candidates = [
        Path(__file__).resolve().parents[1] / 'logs',
        Path(__file__).resolve().parents[2] / 'logs',
        Path.cwd() / 'prompt_opt_harness' / 'logs',
    ]
    logs_dir = None
    for c in candidates:
        if c.exists():
            logs_dir = c
            break
    if logs_dir is None:
        logs_dir = candidates[0]

    detailed_path, summary_path = find_latest_logs(logs_dir)
    if not detailed_path:
        print('No detailed log found in', logs_dir)
        return

    detailed = load_json(detailed_path)
    df = df_from_detailed(detailed)

    out_csv = logs_dir / (detailed_path.stem + '.csv')
    df.to_csv(out_csv, index=False)
    print('Saved CSV to', out_csv)

    # Plot mean ± std of src_acc vs cost, faceted by budget
    plot_mean_with_errorbars(df, 'cost', 'src_acc', ['optimizer'], 'budget', logs_dir / 'src_acc_vs_cost_mean.png', 'Source accuracy vs cost (mean±std)', 'Source accuracy vs cost')
    print('Saved src_acc_vs_cost_mean.png')

    # Plot mean ± std of ood_gap vs cost
    plot_mean_with_errorbars(df, 'cost', 'ood_gap', ['optimizer'], 'budget', logs_dir / 'ood_gap_vs_cost_mean.png', 'OOD gap vs cost (mean±std)', 'OOD gap vs cost')
    print('Saved ood_gap_vs_cost_mean.png')


if __name__ == '__main__':
    main()
