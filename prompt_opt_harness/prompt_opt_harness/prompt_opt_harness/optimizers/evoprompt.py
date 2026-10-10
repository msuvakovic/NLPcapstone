from __future__ import annotations

from ..prompts import CROSSOVER_TEMPLATE, MUTATION_TEMPLATE
from .base import Candidate, Optimizer


class EvoPromptLite(Optimizer):
    name = "EvoPrompt-lite"
    population_size = 4

    def _search(self, task_desc: str, dev_examples) -> Candidate:
        population = [self._score(task_desc, dev_examples)]

        # Seed the population with mutations; each child is scored before the
        # next mutation is requested, so the budget never strands an unscored child.
        while len(population) < self.population_size and self._can_afford(1, 1, dev_examples):
            child = self._generate(MUTATION_TEMPLATE.format(parent=population[-1].instruction))
            population.append(self._score(child, dev_examples))

        while self._can_afford(1, 1, dev_examples):
            parents = sorted(population, key=lambda c: c.dev_score, reverse=True)[:2]
            if len(parents) < 2:
                break
            meta_prompt = CROSSOVER_TEMPLATE.format(
                parent_a=parents[0].instruction, score_a=parents[0].dev_score,
                parent_b=parents[1].instruction, score_b=parents[1].dev_score,
            )
            child = self._score(self._generate(meta_prompt), dev_examples)
            population.append(child)
            population = sorted(population, key=lambda c: c.dev_score, reverse=True)[: self.population_size]

        return self._best()
