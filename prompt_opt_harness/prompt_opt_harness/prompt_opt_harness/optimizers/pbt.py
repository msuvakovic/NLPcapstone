from __future__ import annotations

import random

from ..prompts import MUTATION_TEMPLATE, CROSSOVER_TEMPLATE
from .base import Candidate, Optimizer


class PBT(Optimizer):
    """Population-Based Training style optimizer for prompt strings.

    Maintain a population of candidate instructions and periodically replace
    weaker candidates with mutated/crossover children of stronger parents
    until the search budget is exhausted.
    """

    name = "PBT"
    population_size = 6
    exploit_interval = 4  # how many child evaluations between exploit steps

    def _search(self, task_desc: str, dev_examples) -> Candidate:
        rnd = random.Random(self.seed)
        population = [self._score(task_desc, dev_examples)]

        def mutate(parent: Candidate) -> Candidate:
            return self._score(self._generate(MUTATION_TEMPLATE.format(parent=parent.instruction)), dev_examples)

        while len(population) < self.population_size and self._can_afford(1, 1, dev_examples):
            population.append(mutate(rnd.choice(population)))

        evaluations_since_exploit = 0
        while self._can_afford(1, 1, dev_examples):
            parents = sorted(population, key=lambda c: c.dev_score, reverse=True)
            if len(parents) >= 2 and rnd.random() < 0.6:
                a, b = parents[0], parents[1]
                prompt = CROSSOVER_TEMPLATE.format(parent_a=a.instruction, score_a=a.dev_score,
                                                   parent_b=b.instruction, score_b=b.dev_score)
                child = self._score(self._generate(prompt), dev_examples)
            else:
                child = mutate(rnd.choice(parents))
            population.append(child)
            evaluations_since_exploit += 1

            # Periodically exploit: keep the top half, refill with mutations of it.
            if evaluations_since_exploit >= self.exploit_interval:
                population = sorted(population, key=lambda c: c.dev_score, reverse=True)[: self.population_size]
                top_half = population[: max(1, len(population) // 2)]
                population = top_half.copy()
                while len(population) < self.population_size and self._can_afford(1, 1, dev_examples):
                    population.append(mutate(rnd.choice(top_half)))
                evaluations_since_exploit = 0

        return self._best()
