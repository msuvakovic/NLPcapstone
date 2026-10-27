from __future__ import annotations

import statistics
from typing import Dict, List

from .harness import RunResult, cost_normalized_gains


def _table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    lines = [
        " | ".join(h.ljust(w) for h, w in zip(headers, widths)),
        "-+-".join("-" * w for w in widths),
    ]
    for row in rows:
        lines.append(" | ".join(c.ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)


def build_report(results: List[RunResult]) -> str:
    if not results:
        return "(no results)"

    ood_domains = sorted({domain for r in results for domain in r.ood_accs})
    reference = next((r for r in results if r.name.lower().startswith("zero")), results[0])
    gains = cost_normalized_gains(results, reference)

    headers = ["Method", "Dev", "Src Test", "Dev-Test"] + [f"OOD:{d}" for d in ood_domains] + [
        "OOD Gap", "Worst-Case", "Variance", "API Calls", "Gain/Call",
    ]
    rows = []
    for r in results:
        row = [
            r.name,
            f"{r.dev_score:.2f}",
            f"{r.source_test_acc:.2f}",
            f"{r.dev_minus_test:+.2f}",
        ] + [f"{r.ood_accs.get(d, float('nan')):.2f}" for d in ood_domains] + [
            f"{r.ood_gap:+.2f}",
            f"{r.worst_case_acc:.2f}",
            f"{r.variance:.3f}",
            str(r.api_calls),
            f"{gains[r.name]:+.4f}",
        ]
        rows.append(row)
    return _table(headers, rows)


def _mean_sd(values: List[float], signed: bool = False) -> str:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:+.2f} +/- {sd:.2f}" if signed else f"{mean:.2f} +/- {sd:.2f}"


def build_multiseed_report(runs: Dict[str, List[RunResult]]) -> str:
    if not runs:
        return "(no results)"

    ood_domains = sorted({d for rs in runs.values() for r in rs for d in r.ood_accs})
    headers = ["Method", "Seeds", "Dev", "Src Test", "Dev-Test"] + [f"OOD:{d}" for d in ood_domains] + [
        "Worst-Case", "Avg Calls",
    ]
    rows = []
    for name, rs in runs.items():
        rows.append([
            name,
            str(len(rs)),
            _mean_sd([r.dev_score for r in rs]),
            _mean_sd([r.source_test_acc for r in rs]),
            _mean_sd([r.dev_minus_test for r in rs], signed=True),
        ] + [_mean_sd([r.ood_accs[d] for r in rs if d in r.ood_accs]) for d in ood_domains] + [
            _mean_sd([r.worst_case_acc for r in rs]),
            f"{statistics.mean(r.api_calls for r in rs):.0f}",
        ])
    return _table(headers, rows)
