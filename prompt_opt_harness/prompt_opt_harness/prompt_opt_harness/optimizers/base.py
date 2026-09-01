from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..prompts import build_classification_prompt


@dataclass
class Candidate:
    instruction: str
    dev_score: Optional[float] = None


def evaluate(backend, instruction: str, examples) -> float:
    if not examples:
        return 0.0
    correct = 0
    for ex in examples:
        prompt = build_classification_prompt(instruction, ex.text)
        prediction = backend.generate(prompt).strip().lower()
        if prediction.startswith(ex.label.lower()[:3]):
            correct += 1
    return correct / len(examples)


class Optimizer:
    name = "base"

    def __init__(self, backend, budget: int, seed: int = 0):
        self.backend = backend
        self.budget = budget  # max LLM calls for this run
        self.seed = seed
        self.history: List[Candidate] = []

    def _dev_score(self, instruction: str, dev_examples) -> float:
        return evaluate(self.backend, instruction, dev_examples)

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        raise NotImplementedError
