from __future__ import annotations

from ..prompts import PARAPHRASE_TEMPLATE
from .base import Candidate, Optimizer


class EnsembleOptimizer(Optimizer):
    """Paraphrase the prompt a few times, then score one combined prompt made of the best few (Kanika's branch)."""

    name = "Ensemble"
    population_size = 6
    top_k = 3

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        population = [self._score(task_desc, dev_examples)]

        # keep one full score in reserve for the combined prompt
        while len(population) < self.population_size and self._can_afford(1, 2, dev_examples):
            prompt = PARAPHRASE_TEMPLATE.format(task_name=self.task.name, instruction=population[-1].instruction)
            population.append(self._score(self._propose(prompt), dev_examples))

        top = sorted(population, key=lambda c: c.dev_score, reverse=True)[: self.top_k]
        if len(top) >= 2 and self._can_afford(0, 1, dev_examples):
            combined = ("Consider the following different instructions and then answer accordingly:\n"
                        + "\n---\n".join(c.instruction for c in top))
            self._score(combined, dev_examples)

        return self._best()
