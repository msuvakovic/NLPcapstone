from __future__ import annotations

from typing import List
import random

from ..prompts import MUTATION_TEMPLATE, CROSSOVER_TEMPLATE
from .base import Candidate, Optimizer


class PBT(Optimizer):
    """Population-Based Training style optimizer for prompt strings.

    Simple implementation: maintain a population of candidate instructions,
    periodically replace weaker candidates with mutated/crossover children
    from stronger parents until the LLM call budget is exhausted.
    """

    name = "PBT"
    population_size = 6
    exploit_interval = 4  # how many child evaluations between exploit steps

    def optimize(self, task_desc: str, dev_examples, ood_examples: dict = None) -> Candidate:
        rnd = random.Random(self.seed)

        # Initialize population with the seed instruction plus a few mutations
        population: List[Candidate] = [Candidate(instruction=task_desc, dev_score=self._dev_score(task_desc, dev_examples))]

        # Fill population
        while len(population) < self.population_size and self.backend.stats.calls < self.budget:
            parent = rnd.choice(population)
            mutation_prompt = MUTATION_TEMPLATE.format(parent=parent.instruction)
            child_instruction = self.backend.generate(mutation_prompt).strip()
            population.append(Candidate(instruction=child_instruction))

        # Score any unscored candidates
        for c in population:
            if c.dev_score is None:
                c.dev_score = self._dev_score(c.instruction, dev_examples)
        self.history.extend(population)

        evaluations_since_exploit = 0
        while self.backend.stats.calls < self.budget:
            # Create a child by crossover of two good parents or a mutation
            parents = sorted(population, key=lambda c: c.dev_score, reverse=True)
            if len(parents) >= 2 and rnd.random() < 0.6:
                a, b = parents[0], parents[1]
                prompt = CROSSOVER_TEMPLATE.format(parent_a=a.instruction, score_a=a.dev_score, parent_b=b.instruction, score_b=b.dev_score)
            else:
                parent = rnd.choice(parents)
                prompt = MUTATION_TEMPLATE.format(parent=parent.instruction)

            child_inst = self.backend.generate(prompt).strip()
            if self.backend.stats.calls >= self.budget:
                break

            child = Candidate(instruction=child_inst, dev_score=self._dev_score(child_inst, dev_examples))
            self.history.append(child)
            population.append(child)

            evaluations_since_exploit += 1

            # Periodically exploit: replace worst half with mutated copies of top half
            if evaluations_since_exploit >= self.exploit_interval:
                population = sorted(population, key=lambda c: c.dev_score, reverse=True)[: self.population_size]
                top_half = population[: max(1, len(population) // 2)]
                # Replace bottom half
                new_pop = top_half.copy()
                while len(new_pop) < self.population_size and self.backend.stats.calls < self.budget:
                    parent = rnd.choice(top_half)
                    mutation_prompt = MUTATION_TEMPLATE.format(parent=parent.instruction)
                    child_instruction = self.backend.generate(mutation_prompt).strip()
                    if self.backend.stats.calls >= self.budget:
                        break
                    child = Candidate(instruction=child_instruction, dev_score=self._dev_score(child_instruction, dev_examples))
                    new_pop.append(child)
                    self.history.append(child)

                population = new_pop
                evaluations_since_exploit = 0

        return max(self.history, key=lambda c: c.dev_score)
