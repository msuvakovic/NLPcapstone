from __future__ import annotations
import random

from ..prompts import REFLECTION_TEMPLATE, REFLECTIVE_MUTATION_TEMPLATE
from .base import Candidate, Optimizer, evaluate_per_example

class GEPA(Optimizer):
    name = "GEPA"

    def optimize(self, task_desc: str, dev_examples, ood_examples: dict = None) -> Candidate:
        best_candidate = Candidate(instruction=task_desc, dev_score=self._dev_score(task_desc, dev_examples))
        self.history.append(best_candidate)

        while self.backend.stats.calls < self.budget:
            # Re-evaluate the best instruction per-example to find failures
            per_example_results = evaluate_per_example(self.backend, best_candidate.instruction, dev_examples)
            if self.backend.stats.calls >= self.budget:
                break

            failures = [res for res in per_example_results if not res[2]]
            if not failures:
                # Perfect score, no need to optimize further
                break

            # Pick a random failure to reflect on
            failed_ex, prediction, _ = random.choice(failures)

            reflection_prompt = REFLECTION_TEMPLATE.format(
                instruction=best_candidate.instruction,
                text=failed_ex.text,
                label=failed_ex.label,
                prediction=prediction,
            )
            reflection = self.backend.generate(reflection_prompt).strip()
            if self.backend.stats.calls >= self.budget:
                break

            mutation_prompt = REFLECTIVE_MUTATION_TEMPLATE.format(
                parent=best_candidate.instruction,
                reflection=reflection,
            )
            child_instruction = self.backend.generate(mutation_prompt).strip()
            if self.backend.stats.calls >= self.budget:
                break

            score = self._dev_score(child_instruction, dev_examples)
            child = Candidate(child_instruction, score)
            self.history.append(child)

            if score > best_candidate.dev_score:
                best_candidate = child

        return max(self.history, key=lambda c: c.dev_score)
