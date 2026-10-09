from __future__ import annotations

from typing import List

from .harness import RunResult, cost_normalized_gains


def build_report(results: List[RunResult]) -> str:
    if not results:
        return "(no results)"

    ood_domains = sorted({domain for r in results for domain in r.ood_accs})
    reference = next((r for r in results if r.name.lower().startswith("zero")), results[0])
    gains = cost_normalized_gains(results, reference)

    headers = ["Method", "Dev", "Src Test"] + [f"OOD:{d}" for d in ood_domains] + [
        "OOD Gap", "Worst-Case", "Variance", "Opt Calls/Budget", "Eval Calls", "Budget Stop", "API Calls", "Gain/Call",
    ]
    rows = []
    for r in results:
        row = [
            r.name,
            f"{r.dev_score:.2f}",
            f"{r.source_test_acc:.2f}",
        ] + [f"{r.ood_accs.get(d, float('nan')):.2f}" for d in ood_domains] + [
            f"{r.ood_gap:+.2f}",
            f"{r.worst_case_acc:.2f}",
            f"{r.variance:.3f}",
            f"{r.optimization_calls}/{r.optimization_budget}",
            str(r.api_calls - r.optimization_calls),
            "yes" if r.optimizer_budget_exhausted else "no",
            str(r.api_calls),
            f"{gains[r.name]:+.4f}",
        ]
        rows.append(row)

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    lines = [
        " | ".join(h.ljust(w) for h, w in zip(headers, widths)),
        "-+-".join("-" * w for w in widths),
    ]
    for row in rows:
        lines.append(" | ".join(c.ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)
