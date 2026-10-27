from __future__ import annotations

from ..prompts import META_TEMPLATE
from .base import Candidate, Optimizer

HISTORY_KEPT = 5


class OPRO(Optimizer):
    name = "OPRO"

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        self._score(task_desc, dev_examples)

        # one proposal plus one full pass over the tuning questions per step,
        # and only start a step the budget can finish
        while self._can_afford(1, 1, dev_examples):
            top = sorted(self.history, key=lambda c: c.dev_score, reverse=True)[:HISTORY_KEPT]
            history_block = "\n".join(f"Instruction: {c.instruction}\nScore: {c.dev_score:.2f}\n" for c in top)
            meta_prompt = META_TEMPLATE.format(task_name=self.task.name, history_block=history_block)

            new_instruction = self._propose(meta_prompt)
            self._score(new_instruction, dev_examples)

        return self._best()
