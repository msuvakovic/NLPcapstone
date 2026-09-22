from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..tasks import SENTIMENT_TASK, Task


@dataclass
class Candidate:
    instruction: str
    dev_score: Optional[float] = None


def evaluate(backend, instruction: str, examples, task: Task = SENTIMENT_TASK) -> float:
    if not examples:
        return 0.0
    correct = 0
    for ex in examples:
        prompt = task.build_prompt(instruction, ex.text)
        response = backend.generate(prompt)
        if task.is_correct(response, ex.label):
            correct += 1
    return correct / len(examples)


class Optimizer:
    name = "base"

    def __init__(self, backend, budget: int, seed: int = 0, task: Task = SENTIMENT_TASK):
        self.backend = backend
        self.budget = budget  # max LLM calls for this run
        self.seed = seed
        self.task = task
        self.history: List[Candidate] = []

    def _dev_score(self, instruction: str, dev_examples) -> float:
        return evaluate(self.backend, instruction, dev_examples, task=self.task)

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        raise NotImplementedError
