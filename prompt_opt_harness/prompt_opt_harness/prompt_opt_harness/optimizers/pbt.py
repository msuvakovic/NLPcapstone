from __future__ import annotations

import random

from ..prompts import CROSSOVER_TEMPLATE, MUTATION_TEMPLATE
from .base import Candidate, Optimizer


class PBT(Optimizer):
    """Population-based training for prompts (Kanika's branch, rewritten for this harness).

    Keeps a population of instructions. Children come from crossover of the two best or from a mutation of a
    random member; every few children the weaker half is dropped and refilled from mutations of the top half.
    """

    name = "PBT"
    population_size = 6
    exploit_interval = 4  # children between exploit steps

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        rnd = random.Random(self.seed)
        population = [self._score(task_desc, dev_examples)]

        def mutate(parent: Candidate) -> Candidate:
            prompt = MUTATION_TEMPLATE.format(task_name=self.task.name, parent=parent.instruction)
            return self._score(self._propose(prompt), dev_examples)

        while len(population) < self.population_size and self._can_afford(1, 1, dev_examples):
            population.append(mutate(rnd.choice(population)))

        since_exploit = 0
        while self._can_afford(1, 1, dev_examples):
            parents = sorted(population, key=lambda c: c.dev_score, reverse=True)
            if len(parents) >= 2 and rnd.random() < 0.6:
                a, b = parents[0], parents[1]
                prompt = CROSSOVER_TEMPLATE.format(task_name=self.task.name, parent_a=a.instruction, score_a=a.dev_score,
                                                   parent_b=b.instruction, score_b=b.dev_score)
                child = self._score(self._propose(prompt), dev_examples)
            else:
                child = mutate(rnd.choice(parents))
            population.append(child)
            since_exploit += 1

            if since_exploit >= self.exploit_interval:
                population = sorted(population, key=lambda c: c.dev_score, reverse=True)[: self.population_size]
                top_half = population[: max(1, len(population) // 2)]
                population = top_half.copy()
                while len(population) < self.population_size and self._can_afford(1, 1, dev_examples):
                    population.append(mutate(rnd.choice(top_half)))
                since_exploit = 0

        return self._best()
