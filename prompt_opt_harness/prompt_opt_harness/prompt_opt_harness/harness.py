from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Tuple, Type, Optional

from .datasets import Dataset
from .metrics import cost_normalized_gain, cross_domain_variance, ood_gap, worst_case_accuracy
from .optimizers.base import Optimizer, evaluate


_COUNTERS = ('calls', 'wall_time', 'input_tokens', 'output_tokens',
             'failed_attempts', 'unknown_usage_attempts', 'backoff_time')


def _snapshot_stats(stats):
    return {key: getattr(stats, key, 0) for key in _COUNTERS} | {
        'trace_count': len(getattr(stats, 'per_call', None) or [])}


def validate_splits(dataset: Dataset) -> None:
    """Reject reused IDs or normalized texts before any model calls."""
    splits = [("source.dev", dataset.source.dev), ("source.test", dataset.source.test)]
    splits.extend((f"ood.{domain}", rows) for domain, rows in dataset.ood.items())
    seen_ids, seen_texts = {}, {}
    allowed_labels = {label.casefold() for label in dataset.task.labels}
    for split, examples in splits:
        for example in examples:
            if example.label.strip().casefold() not in allowed_labels:
                raise ValueError(f'Label outside task {dataset.task.name} in {split}: {example.id}')
            text = " ".join(example.text.casefold().split())
            for value, seen, kind in ((example.id, seen_ids, "ID"), (text, seen_texts, "text")):
                if value in seen:
                    raise ValueError(f"Data leakage/duplicate {kind}: {seen[value]} overlaps {split} "
                                     f"(example {example.id}). Use disjoint splits.")
                seen[value] = split


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
    prompt_chars: int = 0
    prompt_words: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    candidate_pool: List[Tuple[str, float]] = field(default_factory=list)
    per_call_traces: List[Dict] = field(default_factory=list)
    latency: Dict[str, float] = field(default_factory=dict)
    tokenizer: str | None = None
    evaluation_protocol: str = "source-dev-only-v1"
    scoring_protocol: str = "exact-label-v2-explicit-task"
    task_spec: Dict = field(default_factory=dict)
    optimizer_metadata: Dict = field(default_factory=dict)
    search_api_calls: int = 0
    evaluation_api_calls: int = 0
    failed_attempts: int = 0
    unknown_usage_attempts: int = 0
    backoff_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "best_instruction": self.best_instruction,
            "dev_score": self.dev_score,
            "source_test_acc": self.source_test_acc,
            "ood_accs": self.ood_accs,
            "ood_gap": self.ood_gap,
            "worst_case_acc": self.worst_case_acc,
            "variance": self.variance,
            "api_calls": self.api_calls,
            "wall_time": self.wall_time,
            "prompt_chars": self.prompt_chars,
            "prompt_words": self.prompt_words,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "candidate_pool": [{"instruction": i, "dev_score": s} for i, s in self.candidate_pool],
            "per_call_traces": self.per_call_traces,
            "latency": self.latency,
            "tokenizer": self.tokenizer,
            "evaluation_protocol": self.evaluation_protocol,
            "scoring_protocol": self.scoring_protocol,
            "task_spec": self.task_spec,
            "optimizer_metadata": self.optimizer_metadata,
            "search_api_calls": self.search_api_calls,
            "evaluation_api_calls": self.evaluation_api_calls,
            "failed_attempts": self.failed_attempts,
            "unknown_usage_attempts": self.unknown_usage_attempts,
            "backoff_time": self.backoff_time,
        }


def run_experiment(
    name: str,
    optimizer_cls: Type[Optimizer],
    backend_factory: Callable[[], object],
    dataset: Dataset,
    budget: int,
    seed: int = 0,
    eval_backend_factory: Optional[Callable[[], object]] = None,
) -> RunResult:
    validate_splits(dataset)
    backend = backend_factory()
    search_start = _snapshot_stats(backend.stats)
    optimizer = optimizer_cls(backend, budget=budget, seed=seed)
    optimizer.task = dataset.task

    best = optimizer.optimize(
        dataset.base_instruction, 
        list(dataset.source.dev),
    )
    search_calls = backend.stats.calls - search_start['calls']
    
    # Use eval_backend if provided (cross-model transfer testing)
    eval_backend = eval_backend_factory() if eval_backend_factory else backend
    eval_start = _snapshot_stats(eval_backend.stats)

    source_test_acc = evaluate(eval_backend, best.instruction, dataset.source.test, dataset.task)
    ood_accs = {domain: evaluate(eval_backend, best.instruction, examples, dataset.task) for domain, examples in dataset.ood.items()}
    ood_values = list(ood_accs.values())
    # Count a separate evaluation backend too; do not double-count a shared one.
    stats = [(backend.stats, search_start)]
    if eval_backend is not backend:
        stats.append((eval_backend.stats, eval_start))
    totals = {key: sum(getattr(stat, key, 0) - start[key] for stat, start in stats)
              for key in _COUNTERS}
    traces = [row for stat, start in stats
              for row in (getattr(stat, 'per_call', None) or [])[start['trace_count']:]]
    latencies = sorted(row['elapsed'] for row in traces if 'elapsed' in row)
    latency = ({name: latencies[min(len(latencies)-1, int(p * len(latencies)))]
                for name, p in (('p50', .5), ('p90', .9), ('p99', .99))} if latencies else {})

    return RunResult(
        name=name,
        best_instruction=best.instruction,
        dev_score=best.dev_score,
        source_test_acc=source_test_acc,
        ood_accs=ood_accs,
        ood_gap=ood_gap(source_test_acc, ood_values),
        worst_case_acc=worst_case_accuracy(ood_values),
        variance=cross_domain_variance(ood_values),
        api_calls=totals['calls'],
        wall_time=totals['wall_time'],
        prompt_chars=len(best.instruction),
        prompt_words=len(best.instruction.split()),
        input_tokens=totals['input_tokens'],
        output_tokens=totals['output_tokens'],
        candidate_pool=[(c.instruction, c.dev_score) for c in optimizer.history],
        per_call_traces=traces,
        latency=latency,
        tokenizer=getattr(backend.stats, "tokenizer_name", None),
        search_api_calls=search_calls,
        evaluation_api_calls=eval_backend.stats.calls-eval_start['calls'],
        failed_attempts=totals['failed_attempts'],
        unknown_usage_attempts=totals['unknown_usage_attempts'],
        backoff_time=totals['backoff_time'],
        task_spec=asdict(dataset.task),
        optimizer_metadata=({'source_subset_ids': optimizer.source_subset_ids}
                            if hasattr(optimizer, 'source_subset_ids') else {}),
    )


def cost_normalized_gains(results: List[RunResult], reference: RunResult) -> Dict[str, float]:
    return {
        r.name: cost_normalized_gain(r.source_test_acc - reference.source_test_acc, r.api_calls)
        for r in results
    }
