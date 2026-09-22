from __future__ import annotations

from ..prompts import META_TEMPLATE
from .base import Candidate, Optimizer

HISTORY_KEPT = 5


class OPRO(Optimizer):
    name = "OPRO"

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        seed = Candidate(instruction=task_desc, dev_score=self._dev_score(task_desc, dev_examples))
        self.history.append(seed)

        while self.backend.stats.calls < self.budget:
            top = sorted(self.history, key=lambda c: c.dev_score, reverse=True)[:HISTORY_KEPT]
            history_block = "\n".join(f"Instruction: {c.instruction}\nScore: {c.dev_score:.2f}\n" for c in top)
            meta_prompt = META_TEMPLATE.format(task_name=self.task.name, history_block=history_block)

            new_instruction = self.backend.generate(meta_prompt).strip()
            if self.backend.stats.calls >= self.budget:
                break

            score = self._dev_score(new_instruction, dev_examples)
            self.history.append(Candidate(new_instruction, score))

        return max(self.history, key=lambda c: c.dev_score)
