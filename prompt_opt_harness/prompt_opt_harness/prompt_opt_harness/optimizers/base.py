from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from ..prompts import build_classification_prompt
from ..tasks import SENTIMENT, TaskSpec


@dataclass
class Candidate:
    instruction: str
    dev_score: Optional[float] = None
    ood_accs: Optional[Dict[str, float]] = None
    ood_score: Optional[float] = None


def parse_label(raw: str, task: TaskSpec = SENTIMENT) -> str:
    """Accept a complete label, optionally quoted or followed by one ./!.

    Keep lowercase predictions for existing reflection consumers. Explanations,
    partial words and multiple labels are invalid, even if their prefix matches.
    """
    labels = '|'.join(re.escape(label.casefold()) for label in task.labels)
    match = re.fullmatch(rf'''(?:({labels})|(["'])({labels})\2)[.!]?''', raw.strip().casefold())
    return (match[1] or match[3]) if match else 'INVALID'


def evaluate_per_example(backend, instruction: str, examples, task: TaskSpec = SENTIMENT) -> List[Tuple[object, str, bool]]:
    """Score exact prompts once per batch; never reuse a similar text's answer."""
    examples = list(examples)
    allowed = {label.casefold() for label in task.labels}
    if any(ex.label.strip().casefold() not in allowed for ex in examples):
        raise ValueError(f'Example label outside task {task.name}; pass the explicit TaskSpec')
    prompts = [build_classification_prompt(instruction, ex.text, task) for ex in examples]
    # Budgeted optimizers must afford the whole batch before sending anything.
    if hasattr(backend, 'require_calls'):
        backend.require_calls(len(set(prompts)))
    cache = {}
    results = []
    for ex, prompt in zip(examples, prompts):
        if prompt not in cache:
            cache[prompt] = parse_label(backend.generate(prompt), task)
        prediction = cache[prompt]
        correct = prediction != 'INVALID' and prediction == ex.label.strip().casefold()
        results.append((ex, prediction, correct))
    return results


def evaluate(backend, instruction: str, examples, task: TaskSpec = SENTIMENT) -> float:
    rows = evaluate_per_example(backend, instruction, examples, task)
    return sum(correct for _, _, correct in rows) / len(rows) if rows else 0.0


class Optimizer:
    name = "base"

    def __init__(self, backend, budget: int, seed: int = 0, task: TaskSpec = SENTIMENT):
        self.backend = backend
        self.task = task
        self.budget = budget  # OPRO caps search attempts; final evaluation is separate.
        self.seed = seed
        self.history: List[Candidate] = []

    def ood_rank_score(self, dev_score: float, ood_accs: dict | list | tuple) -> float:
        if isinstance(ood_accs, dict):
            ood_values = list(ood_accs.values())
        else:
            ood_values = list(ood_accs)
        if not ood_values:
            return dev_score
        mean_ood = sum(ood_values) / len(ood_values)
        worst_ood = min(ood_values)
        gap = abs(dev_score - mean_ood)
        # score = 0.4 * source_acc + 0.6 * mean_ood_acc - ? * |ood_gap|
        return 0.4 * dev_score + 0.6 * mean_ood - 0.5 * gap

    def progressive_score(self, dev_score: float, ood_accs: dict | list | tuple, progress: float) -> float:
        ood_score = self.ood_rank_score(dev_score, ood_accs)
        return (1.0 - progress) * dev_score + progress * ood_score

    def _evaluation_calls(self, dev_examples) -> int:
        """Requests for a full score, excluding unpredictable retries.

        Sampling/paraphrase wrappers must include their extra requests. Adapters
        with a separate evaluator budget can override this reservation.
        """
        return len({ex.text for ex in dev_examples})

    def _dev_score(self, instruction: str, dev_examples) -> float:
        return evaluate(self.backend, instruction, dev_examples, self.task)

    def optimize(self, task_desc: str, dev_examples, ood_examples: dict = None) -> Candidate:
        raise NotImplementedError
