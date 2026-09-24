from __future__ import annotations

from typing import List
import random

from ..prompts import REFLECTION_TEMPLATE, REFLECTIVE_MUTATION_TEMPLATE
from .base import Candidate, Optimizer, evaluate_per_example


class ReflectiveOptimizer(Optimizer):
    """Use per-example failures to generate short reflections and targeted mutations.

    Workflow:
    - Start from seed instruction
    - Evaluate per-example to find a failing example
    - Ask for a brief reflection on the failure
    - Use the reflection to create a focused mutation
    - Repeat until budget exhausted
    """

    name = "Reflective"

    def optimize(self, task_desc: str, dev_examples, ood_examples: dict = None) -> Candidate:
        rnd = random.Random(self.seed)
        current = Candidate(instruction=task_desc, dev_score=self._dev_score(task_desc, dev_examples))
        self.history.append(current)

        while self.backend.stats.calls < self.budget:
            # get per-example results and find a failing example
            per_example = evaluate_per_example(self.backend, current.instruction, dev_examples)
            failures = [t for t in per_example if not t[2]]
            if not failures:
                break

            ex, prediction, is_correct = rnd.choice(failures)
            reflection_prompt = REFLECTION_TEMPLATE.format(instruction=current.instruction, text=ex.text, label=ex.label, prediction=prediction)
            reflection = self.backend.generate(reflection_prompt).strip()
            if self.backend.stats.calls >= self.budget:
                break

            child_prompt = REFLECTIVE_MUTATION_TEMPLATE.format(parent=current.instruction, reflection=reflection)
            new_instruction = self.backend.generate(child_prompt).strip()
            if self.backend.stats.calls >= self.budget:
                break

            new_score = self._dev_score(new_instruction, dev_examples)
            new_candidate = Candidate(instruction=new_instruction, dev_score=new_score)
            self.history.append(new_candidate)

            # if improvement, adopt as current
            if new_score >= current.dev_score:
                current = new_candidate

        return max(self.history, key=lambda c: c.dev_score)
