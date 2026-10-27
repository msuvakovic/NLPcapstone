from __future__ import annotations

import random

from ..prompts import RANDOM_SEARCH_TEMPLATES
from .base import Candidate, Optimizer


class RandomSearch(Optimizer):
    """APE-style baseline: independent rewrites of the starting instruction, each scored on the
    tuning questions, best one kept. No proposal sees any score, wrong answer or earlier
    candidate, so the gap between this and a real optimizer shows what the optimizer's search
    signal adds beyond spending the same budget on more candidates.
    """

    name = "Random-search"

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        rng = random.Random(self.seed)
        self._score(task_desc, dev_examples)
        seen = {task_desc}

        variant = 0
        while self._can_afford(1, 1, dev_examples):
            variant += 1
            template = rng.choice(RANDOM_SEARCH_TEMPLATES)
            rewrite = self._propose(template.format(task_name=self.task.name, instruction=task_desc, variant=variant))
            if rewrite and rewrite not in seen:
                seen.add(rewrite)
                self._score(rewrite, dev_examples)

        return self._best()
