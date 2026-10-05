from __future__ import annotations

from ..prompts import TEXTGRAD_CRITIQUE_TEMPLATE, TEXTGRAD_UPDATE_TEMPLATE
from .base import Candidate, Optimizer, evaluate_per_example

class TextGrad(Optimizer):
    name = "TextGrad"
    batch_size = 5

    def optimize(self, task_desc: str, dev_examples, ood_examples: dict = None) -> Candidate:
        best_candidate = Candidate(instruction=task_desc, dev_score=self._dev_score(task_desc, dev_examples))
        self.history.append(best_candidate)

        while self.backend.stats.calls < self.budget:
            # Re-evaluate to find failures
            per_example_results = evaluate_per_example(self.backend, best_candidate.instruction, dev_examples, self.task)
            if self.backend.stats.calls >= self.budget:
                break

            failures = [res for res in per_example_results if not res[2]]
            if not failures:
                break

            # Aggregate a batch of failures
            batch = failures[:self.batch_size]
            failures_block = "\n".join(
                f"Text: {ex.text}\nTrue: {ex.label}\nPred: {pred}\n" 
                for ex, pred, _ in batch
            )

            # Compute Textual Gradient (Critique)
            critique_prompt = TEXTGRAD_CRITIQUE_TEMPLATE.format(
                instruction=best_candidate.instruction,
                failures_block=failures_block
            )
            gradient = self.backend.generate(critique_prompt).strip()
            if self.backend.stats.calls >= self.budget:
                break

            # Apply Gradient to update the prompt
            update_prompt = TEXTGRAD_UPDATE_TEMPLATE.format(
                instruction=best_candidate.instruction,
                gradient=gradient
            )
            child_instruction = self.backend.generate(update_prompt).strip()
            if self.backend.stats.calls >= self.budget:
                break

            score = self._dev_score(child_instruction, dev_examples)
            child = Candidate(child_instruction, score)
            self.history.append(child)

            if score > best_candidate.dev_score:
                best_candidate = child

        return max(self.history, key=lambda c: c.dev_score)
