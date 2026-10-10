from __future__ import annotations

from .base import Candidate, Optimizer
from ..prompts import PARAPHRASE_TEMPLATE


class EnsembleOptimizer(Optimizer):
    """Create an ensemble by generating diverse candidates and synthesizing
    a single instruction that asks the model to consider multiple views.

    This is a simple, black-box ensemble that concatenates top-k prompts
    into a single instruction template.
    """

    name = "Ensemble"
    population_size = 6
    top_k = 3

    def _search(self, task_desc: str, dev_examples) -> Candidate:
        population = [self._score(task_desc, dev_examples)]

        # Keep one full score in reserve for the synthesized ensemble prompt.
        while len(population) < self.population_size and self._can_afford(1, 2, dev_examples):
            inst = self._generate(PARAPHRASE_TEMPLATE.format(instruction=population[-1].instruction))
            population.append(self._score(inst, dev_examples))

        topk = sorted(population, key=lambda c: c.dev_score, reverse=True)[: self.top_k]
        if len(topk) >= 2 and self._can_afford(0, 1, dev_examples):
            ensemble_instruction = ("Consider the following different instructions and then answer accordingly:\n"
                                    + "\n---\n".join(c.instruction for c in topk))
            self._score(ensemble_instruction, dev_examples)

        return self._best()
