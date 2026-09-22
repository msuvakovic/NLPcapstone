from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Type

from .datasets import Dataset
from .metrics import cost_normalized_gain, cross_domain_variance, ood_gap, worst_case_accuracy
from .optimizers.base import Optimizer, evaluate
from .tasks import SENTIMENT_TASK, Task


@dataclass
class RunResult:
    name: str
    best_instruction: str
    dev_score: float
    source_test_acc: float
    ood_accs: Dict[str, float]
    ood_gap: float
    worst_case_acc: float
    variance: float
    api_calls: int
    wall_time: float
    candidate_pool: List[Tuple[str, float]] = field(default_factory=list)


def run_experiment(
    name: str,
    optimizer_cls: Type[Optimizer],
    backend_factory: Callable[[], object],
    dataset: Dataset,
    budget: int,
    seed: int = 0,
    task: Task = SENTIMENT_TASK,
) -> RunResult:
    backend = backend_factory()
    optimizer = optimizer_cls(backend, budget=budget, seed=seed, task=task)

    best = optimizer.optimize(dataset.base_instruction, dataset.source.dev)

    source_test_acc = evaluate(backend, best.instruction, dataset.source.test, task=task)
    ood_accs = {domain: evaluate(backend, best.instruction, examples, task=task) for domain, examples in dataset.ood.items()}
    ood_values = list(ood_accs.values())

    return RunResult(
        name=name,
        best_instruction=best.instruction,
        dev_score=best.dev_score,
        source_test_acc=source_test_acc,
        ood_accs=ood_accs,
        ood_gap=ood_gap(source_test_acc, ood_values),
        worst_case_acc=worst_case_accuracy(ood_values),
        variance=cross_domain_variance(ood_values),
        api_calls=backend.stats.calls,
        wall_time=backend.stats.wall_time,
        candidate_pool=[(c.instruction, c.dev_score) for c in optimizer.history],
    )


def cost_normalized_gains(results: List[RunResult], reference: RunResult) -> Dict[str, float]:
    return {
        r.name: cost_normalized_gain(r.source_test_acc - reference.source_test_acc, r.api_calls)
        for r in results
    }
