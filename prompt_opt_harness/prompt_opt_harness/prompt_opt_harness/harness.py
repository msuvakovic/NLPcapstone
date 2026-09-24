from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Type, Optional

from .datasets import Dataset
from .metrics import cost_normalized_gain, cross_domain_variance, ood_gap, worst_case_accuracy
from .optimizers.base import Optimizer, evaluate


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
    backend = backend_factory()
    optimizer = optimizer_cls(backend, budget=budget, seed=seed)

    best = optimizer.optimize(
        dataset.base_instruction, 
        dataset.source.dev,
        ood_examples=dataset.ood
    )
    
    # Use eval_backend if provided (cross-model transfer testing)
    eval_backend = eval_backend_factory() if eval_backend_factory else backend

    source_test_acc = evaluate(eval_backend, best.instruction, dataset.source.test)
    ood_accs = {domain: evaluate(eval_backend, best.instruction, examples) for domain, examples in dataset.ood.items()}
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
        prompt_chars=len(best.instruction),
        prompt_words=len(best.instruction.split()),
        input_tokens=getattr(backend.stats, "input_tokens", 0),
        output_tokens=getattr(backend.stats, "output_tokens", 0),
        candidate_pool=[(c.instruction, c.dev_score) for c in optimizer.history],
        per_call_traces=getattr(backend.stats, "per_call", []),
        latency=getattr(backend.stats, "latency_percentiles", lambda: {})(),
        tokenizer=getattr(backend.stats, "tokenizer_name", None),
    )


def cost_normalized_gains(results: List[RunResult], reference: RunResult) -> Dict[str, float]:
    return {
        r.name: cost_normalized_gain(r.source_test_acc - reference.source_test_acc, r.api_calls)
        for r in results
    }
