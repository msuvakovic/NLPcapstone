from __future__ import annotations

import random

from ..prompts import REFLECTION_TEMPLATE, REFLECTIVE_MUTATION_TEMPLATE
from .base import Candidate, Optimizer, format_failure


class GEPALite(Optimizer):
    """Reflective search in the spirit of GEPA: read one wrong answer, write what the
    instruction missed, rewrite the instruction, keep it only if it scores better.

    This is a small greedy version. It has no Pareto frontier and no merging of
    candidates, so call it GEPA-lite, not GEPA.
    """

    name = "GEPA-lite"

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        rng = random.Random(self.seed)
        best = self._score(task_desc, dev_examples)
        seen = {task_desc}

        # two proposal calls (reflection, then rewrite) and one full pass per step
        while self._can_afford(2, 1, dev_examples):
            failures = self._failures(best)
            if not failures:
                break  # perfect on the tuning questions, nothing to reflect on
            reflection = self._propose(REFLECTION_TEMPLATE.format(
                task_name=self.task.name, instruction=best.instruction,
                failure=format_failure(rng.choice(failures)),
            ))
            rewrite = self._propose(REFLECTIVE_MUTATION_TEMPLATE.format(
                task_name=self.task.name, parent=best.instruction, reflection=reflection,
            ))
            if not rewrite or rewrite in seen:
                continue  # nothing new to score, and scoring costs a full pass
            seen.add(rewrite)
            child = self._score(rewrite, dev_examples)
            if child.dev_score > best.dev_score:
                best = child

        return self._best()
