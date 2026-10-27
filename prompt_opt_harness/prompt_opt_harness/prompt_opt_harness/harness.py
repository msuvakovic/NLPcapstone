from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Type

from .datasets import Dataset
from .metrics import cost_normalized_gain, cross_domain_variance, ood_gap, worst_case_accuracy
from .optimizers.base import Optimizer, evaluate_detailed
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
    candidate_stats: List[dict] = field(default_factory=list)
    seed: int = 0
    config: dict = field(default_factory=dict)
    search_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    dev_records: List[dict] = field(default_factory=list)
    eval_records: Dict[str, List[dict]] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)  # e.g. the confirmation guard's decision

    @property
    def dev_minus_test(self) -> float:
        return self.dev_score - self.source_test_acc

    @property
    def eval_summary(self) -> dict:
        records = [r for rs in self.eval_records.values() for r in rs]
        if not records:
            return {}
        return {
            "n": len(records),
            "truncated": sum(r["truncated"] for r in records),
            "unformatted": sum(not r["well_formed"] for r in records),
        }


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def check_splits(dataset: Dataset) -> None:
    """Fail before any model call if a question used for tuning also appears in a reported test set."""
    tuning = {"source tuning set": dataset.source.dev, "source confirmation set": dataset.source.confirm}
    tuning.update({f"{d} tuning sample": rows for d, rows in dataset.ood_val.items()})
    reported = {"source test set": dataset.source.test}
    reported.update({f"{d} test set": rows for d, rows in dataset.ood.items()})
    seen = {}
    for name, rows in tuning.items():
        for ex in rows:
            seen.setdefault(_norm(ex.text), name)
    for name, rows in reported.items():
        for ex in rows:
            where = seen.get(_norm(ex.text))
            if where:
                raise ValueError(
                    f"Data leak: question {ex.id!r} is in both the {where} and the {name}. "
                    "Tuning and test questions must not overlap."
                )


def run_experiment(
    name: str,
    optimizer_cls: Type[Optimizer],
    backend_factory: Callable[[], object],
    dataset: Dataset,
    budget: int,
    seed: int = 0,
    task: Task = SENTIMENT_TASK,
    proposal_temperature: Optional[float] = None,
    config: Optional[dict] = None,
) -> RunResult:
    check_splits(dataset)
    backend = backend_factory()
    optimizer = optimizer_cls(backend, budget=budget, seed=seed, task=task, proposal_temperature=proposal_temperature)

    if optimizer.tune_on_target:
        tuning_examples = [ex for rows in dataset.ood_val.values() for ex in rows]
        if not tuning_examples:
            raise ValueError(f"{optimizer.name} tunes on the new dataset, but this dataset has no tuning sample for it (ood_val).")
    else:
        tuning_examples = dataset.source.dev

    best = optimizer.optimize(dataset.base_instruction, tuning_examples)
    search_calls = backend.stats.calls

    extra = {}
    if optimizer.needs_confirm:
        # Confirmation guard: keep the search's winner only if it also beats the starting prompt on separate
        # source questions the search never saw. Ties and losses keep the starting prompt.
        confirm = dataset.source.confirm
        if not confirm:
            raise ValueError(f"{optimizer.name} needs a source confirmation sample, but this dataset has none (source.confirm).")
        start = next((c for c in optimizer.history if c.instruction == dataset.base_instruction), optimizer.history[0])
        if best.instruction == start.instruction:
            extra["guard"] = {"checked": False, "kept": "starting prompt", "reason": "the search kept the starting prompt", "n_confirm": len(confirm)}
        else:
            start_acc, _ = evaluate_detailed(backend, start.instruction, confirm, task=task)
            win_acc, _ = evaluate_detailed(backend, best.instruction, confirm, task=task)
            keep_winner = win_acc > start_acc
            extra["guard"] = {"checked": True, "n_confirm": len(confirm), "start_confirm_acc": start_acc, "winner_confirm_acc": win_acc,
                              "winner_instruction": best.instruction, "kept": "search winner" if keep_winner else "starting prompt"}
            if not keep_winner:
                best = start

    source_test_acc, test_records = evaluate_detailed(backend, best.instruction, dataset.source.test, task=task)
    eval_records = {"source_test": test_records}
    ood_accs = {}
    for domain, examples in dataset.ood.items():
        ood_accs[domain], eval_records[f"ood:{domain}"] = evaluate_detailed(backend, best.instruction, examples, task=task)
    ood_values = list(ood_accs.values())

    candidate_stats = []
    for step in range(len(optimizer.history)):
        stats = dict(optimizer.dev_stats[step]) if step < len(optimizer.dev_stats) else {}
        stats["step"] = step
        candidate_stats.append(stats)
    best_idx = next(i for i, c in enumerate(optimizer.history) if c is best)

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
        candidate_stats=candidate_stats,
        seed=seed,
        config={
            **backend.info(),
            "task": task.name,
            "budget": budget,
            "proposal_temperature": proposal_temperature,
            "n_dev": len(dataset.source.dev),
            "n_tuning": len(tuning_examples),
            "n_confirm": len(dataset.source.confirm),
            "tuned_on": "new dataset sample (oracle)" if optimizer.tune_on_target else "source tuning set",
            "n_test": len(dataset.source.test),
            "n_ood": {d: len(e) for d, e in dataset.ood.items()},
            "base_instruction": dataset.base_instruction,
            **(config or {}),
        },
        search_calls=search_calls,
        input_tokens=backend.stats.input_tokens,
        output_tokens=backend.stats.output_tokens,
        dev_records=optimizer.dev_records[best_idx] if best_idx < len(optimizer.dev_records) else [],
        eval_records=eval_records,
        extra=extra,
    )


def cost_normalized_gains(results: List[RunResult], reference: RunResult) -> Dict[str, float]:
    return {
        r.name: cost_normalized_gain(r.source_test_acc - reference.source_test_acc, r.api_calls)
        for r in results
    }
