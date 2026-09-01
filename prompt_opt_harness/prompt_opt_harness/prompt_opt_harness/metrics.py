from __future__ import annotations

import statistics
from typing import Iterable


def ood_gap(source_test_acc: float, ood_accs: Iterable[float]) -> float:
    ood_accs = list(ood_accs)
    if not ood_accs:
        return 0.0
    return source_test_acc - (sum(ood_accs) / len(ood_accs))


def cross_domain_variance(ood_accs: Iterable[float]) -> float:
    ood_accs = list(ood_accs)
    if len(ood_accs) < 2:
        return 0.0
    return statistics.pvariance(ood_accs)


def worst_case_accuracy(ood_accs: Iterable[float]) -> float:
    ood_accs = list(ood_accs)
    return min(ood_accs) if ood_accs else 0.0


def cost_normalized_gain(gain: float, api_calls: int) -> float:
    return gain / api_calls if api_calls else 0.0
