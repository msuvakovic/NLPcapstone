from __future__ import annotations

import random

from ..prompts import REFLECTION_TEMPLATE, REFLECTIVE_MUTATION_TEMPLATE
from .base import Candidate, Optimizer


class ReflectiveOptimizer(Optimizer):
    """Use per-example failures to generate short reflections and targeted mutations.

    Workflow:
    - Start from seed instruction
    - Pick a failing dev example of the current prompt
    - Ask for a brief reflection on the failure
    - Use the reflection to create a focused mutation
    - Adopt the mutation if it scores at least as well; repeat within budget
    """

    name = "Reflective"

    def _search(self, task_desc: str, dev_examples) -> Candidate:
        rnd = random.Random(self.seed)
        current = self._score(task_desc, dev_examples)

        while self._can_afford(2, 1, dev_examples):
            failures = self._failures(current.instruction, dev_examples)
            if not failures:
                break
            ex, prediction, _ = rnd.choice(failures)
            reflection = self._generate(REFLECTION_TEMPLATE.format(
                instruction=current.instruction, text=ex.text, label=ex.label, prediction=prediction))
            candidate = self._score(self._generate(REFLECTIVE_MUTATION_TEMPLATE.format(
                parent=current.instruction, reflection=reflection)), dev_examples)
            if candidate.dev_score >= current.dev_score:
                current = candidate

        return self._best()
