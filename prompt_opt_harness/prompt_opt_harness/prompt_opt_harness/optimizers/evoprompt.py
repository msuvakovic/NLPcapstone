from __future__ import annotations

from ..prompts import CROSSOVER_TEMPLATE, MUTATION_TEMPLATE
from .base import Candidate, Optimizer


class EvoPromptLite(Optimizer):
    name = "EvoPrompt-lite"
    population_size = 4

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        population = [Candidate(instruction=task_desc)]

        while len(population) < self.population_size and self.backend.stats.calls < self.budget:
            mutation_prompt = MUTATION_TEMPLATE.format(task_name=self.task.name, parent=population[-1].instruction)
            child_instruction = self.backend.generate(mutation_prompt).strip()
            population.append(Candidate(instruction=child_instruction))

        for candidate in population:
            if candidate.dev_score is None:
                candidate.dev_score = self._dev_score(candidate.instruction, dev_examples)
        self.history.extend(population)

        while self.backend.stats.calls < self.budget:
            parents = sorted(population, key=lambda c: c.dev_score, reverse=True)[:2]
            if len(parents) < 2:
                break

            meta_prompt = CROSSOVER_TEMPLATE.format(
                task_name=self.task.name,
                parent_a=parents[0].instruction, score_a=parents[0].dev_score,
                parent_b=parents[1].instruction, score_b=parents[1].dev_score,
            )
            child_instruction = self.backend.generate(meta_prompt).strip()
            if self.backend.stats.calls >= self.budget:
                break

            child = Candidate(child_instruction, self._dev_score(child_instruction, dev_examples))
            self.history.append(child)
            population.append(child)
            population = sorted(population, key=lambda c: c.dev_score, reverse=True)[: self.population_size]

        return max(self.history, key=lambda c: c.dev_score)
