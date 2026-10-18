from __future__ import annotations

import random

from ..prompts import RANDOM_SEARCH_TEMPLATES
from .base import Candidate, Optimizer


class RandomSearch(Optimizer):
    """Score independent, uninformed rewrites of the original instruction."""

    name = "Random-search"

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        if len(dev_examples) > self.budget:
            raise ValueError(
                f"budget ({self.budget}) must cover the initial dev evaluation "
                f"({len(dev_examples)} calls)"
            )

        seed_candidate = Candidate(task_desc, self._dev_score(task_desc, dev_examples))
        self.history.append(seed_candidate)
        rng = random.Random(self.seed)
        variant = 0

        while self.backend.remaining_calls >= len(dev_examples) + 1:
            variant += 1
            template = rng.choice(RANDOM_SEARCH_TEMPLATES)
            proposal_prompt = template.format(
                task_name=self.task.name,
                instruction=task_desc,
                variant=variant,
            )
            instruction = self.backend.generate(proposal_prompt).strip()
            if not instruction:
                continue
            score = self._dev_score(instruction, dev_examples)
            self.history.append(Candidate(instruction, score))

        return max(self.history, key=lambda candidate: candidate.dev_score)