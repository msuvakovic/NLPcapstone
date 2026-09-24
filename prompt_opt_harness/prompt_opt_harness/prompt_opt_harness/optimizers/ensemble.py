from __future__ import annotations

from typing import List
import random

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

    def optimize(self, task_desc: str, dev_examples, ood_examples: dict = None) -> Candidate:
        rnd = random.Random(self.seed)
        population: List[Candidate] = [Candidate(instruction=task_desc, dev_score=self._dev_score(task_desc, dev_examples))]

        # generate diverse paraphrases/mutations
        while len(population) < self.population_size and self.backend.stats.calls < self.budget:
            prompt = PARAPHRASE_TEMPLATE.format(instruction=population[-1].instruction)
            inst = self.backend.generate(prompt).strip()
            population.append(Candidate(instruction=inst))

        for c in population:
            if c.dev_score is None:
                c.dev_score = self._dev_score(c.instruction, dev_examples)
        self.history.extend(population)

        # synthesize ensemble instruction from top-k
        topk = sorted(population, key=lambda c: c.dev_score, reverse=True)[: self.top_k]
        fragments = [c.instruction for c in topk]
        ensemble_instruction = "Consider the following different instructions and then answer accordingly:\n" + "\n---\n".join(fragments)
        ensemble_score = self._dev_score(ensemble_instruction, dev_examples)
        ensemble_candidate = Candidate(instruction=ensemble_instruction, dev_score=ensemble_score)
        self.history.append(ensemble_candidate)

        return max(self.history, key=lambda c: c.dev_score)
