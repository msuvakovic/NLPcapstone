from __future__ import annotations

import random

from ..prompts import RANDOM_SEARCH_TEMPLATES
from .base import Candidate, Optimizer


class RandomSearch(Optimizer):
    """APE-style random-search baseline from the capstone proposal.

    Samples independent rewrites of the *seed* instruction, scores each on
    source dev, and keeps the best. Unlike OPRO/EvoPrompt/GEPA, no proposal
    sees any score, failure or earlier candidate, so the gap between this and
    a real optimizer isolates the value of the optimizer's search signal from
    simply spending the same compute on more candidates.

    Each candidate costs one proposal request plus one full dev score, under
    the same shared budget guard as every other optimizer.
    """

    name = "Random-search"

    def _search(self, task_desc: str, dev_examples) -> Candidate:
        rng = random.Random(self.seed)
        self._score(task_desc, dev_examples)

        variant = 0
        while self._can_afford(1, 1, dev_examples):
            variant += 1
            template = rng.choice(RANDOM_SEARCH_TEMPLATES)
            rewrite = self._generate(template.format(instruction=task_desc, variant=variant))
            if rewrite:
                self._score(rewrite, dev_examples)

        return self._best()
