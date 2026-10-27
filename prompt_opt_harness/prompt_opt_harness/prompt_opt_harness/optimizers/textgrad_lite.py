from __future__ import annotations

from ..prompts import TEXTGRAD_CRITIQUE_TEMPLATE, TEXTGRAD_UPDATE_TEMPLATE
from .base import Candidate, Optimizer, format_failure


class TextGradLite(Optimizer):
    """Critique-then-update search in the spirit of TextGrad: write a critique of a batch of
    wrong answers (the text version of a gradient), then rewrite the instruction using it.

    This is not the TextGrad library, so call it TextGrad-lite.
    """

    name = "TextGrad-lite"
    batch_size = 5

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        best = self._score(task_desc, dev_examples)
        seen = {task_desc}

        # two proposal calls (critique, then update) and one full pass per step
        while self._can_afford(2, 1, dev_examples):
            failures = self._failures(best)
            if not failures:
                break
            failures_block = "\n\n".join(format_failure(r, limit=250) for r in failures[: self.batch_size])
            critique = self._propose(TEXTGRAD_CRITIQUE_TEMPLATE.format(
                task_name=self.task.name, instruction=best.instruction, failures_block=failures_block,
            ))
            update = self._propose(TEXTGRAD_UPDATE_TEMPLATE.format(
                task_name=self.task.name, instruction=best.instruction, gradient=critique,
            ))
            if not update or update in seen:
                continue
            seen.add(update)
            child = self._score(update, dev_examples)
            if child.dev_score > best.dev_score:
                best = child

        return self._best()
