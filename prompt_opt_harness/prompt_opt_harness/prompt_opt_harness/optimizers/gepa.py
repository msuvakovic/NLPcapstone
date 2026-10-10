from __future__ import annotations
import random

from ..prompts import REFLECTION_TEMPLATE, REFLECTIVE_MUTATION_TEMPLATE
from .base import Candidate, Optimizer


class GEPA(Optimizer):
    """GEPA-style reflective search: reflect on one failure, mutate, keep if better."""
    name = "GEPA"

    def _search(self, task_desc: str, dev_examples) -> Candidate:
        rng = random.Random(self.seed)
        best = self._score(task_desc, dev_examples)

        # Two proposal calls (reflection + mutation) and one full score per step.
        while self._can_afford(2, 1, dev_examples):
            failures = self._failures(best.instruction, dev_examples)
            if not failures:
                break  # Perfect on dev; nothing left to reflect on.
            failed_ex, prediction, _ = rng.choice(failures)
            reflection = self._generate(REFLECTION_TEMPLATE.format(
                instruction=best.instruction, text=failed_ex.text,
                label=failed_ex.label, prediction=prediction,
            ))
            child = self._score(self._generate(REFLECTIVE_MUTATION_TEMPLATE.format(
                parent=best.instruction, reflection=reflection,
            )), dev_examples)
            if child.dev_score > best.dev_score:
                best = child

        return self._best()
