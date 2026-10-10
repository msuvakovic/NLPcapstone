from __future__ import annotations

from ..prompts import TEXTGRAD_CRITIQUE_TEMPLATE, TEXTGRAD_UPDATE_TEMPLATE
from .base import Candidate, Optimizer


class TextGrad(Optimizer):
    """TextGrad-style search: critique a batch of failures, then apply the critique."""
    name = "TextGrad"
    batch_size = 5

    def _search(self, task_desc: str, dev_examples) -> Candidate:
        best = self._score(task_desc, dev_examples)

        # Two proposal calls (critique + update) and one full score per step.
        while self._can_afford(2, 1, dev_examples):
            failures = self._failures(best.instruction, dev_examples)
            if not failures:
                break
            failures_block = "\n".join(
                f"Text: {ex.text}\nTrue: {ex.label}\nPred: {pred}\n"
                for ex, pred, _ in failures[: self.batch_size]
            )
            gradient = self._generate(TEXTGRAD_CRITIQUE_TEMPLATE.format(
                instruction=best.instruction, failures_block=failures_block,
            ))
            child = self._score(self._generate(TEXTGRAD_UPDATE_TEMPLATE.format(
                instruction=best.instruction, gradient=gradient,
            )), dev_examples)
            if child.dev_score > best.dev_score:
                best = child

        return self._best()
