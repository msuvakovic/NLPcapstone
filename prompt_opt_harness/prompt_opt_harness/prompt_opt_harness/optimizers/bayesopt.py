from __future__ import annotations

import random
import warnings

from ..prompts import CROSSOVER_TEMPLATE, MUTATION_TEMPLATE
from .base import Candidate, Optimizer


class CostBayesOpt(Optimizer):
    """Cost-aware Bayesian optimization over prompt strings (Kanika's branch).

    Surrogate: TF-IDF features + a Gaussian process. Acquisition: upper confidence bound divided by the
    prompt's length, so shorter prompts are preferred at equal promise. Needs scikit-learn and numpy
    (`pip3 install scikit-learn`); the other optimizers need nothing extra.
    """

    name = "CostBayesOpt"
    init_random = 4
    kappa = 1.96
    candidate_pool = 12

    def _surrogate(self):
        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
        except ImportError as exc:
            raise ImportError("CostBayesOpt needs scikit-learn: pip3 install scikit-learn") from exc
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
        return np, TfidfVectorizer(ngram_range=(1, 2), max_features=1024), GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    def _draw(self, rnd) -> str:
        if rnd.random() < 0.5 and len(self.history) >= 2:
            a, b = rnd.sample(self.history, 2)
            prompt = CROSSOVER_TEMPLATE.format(task_name=self.task.name, parent_a=a.instruction, score_a=a.dev_score,
                                               parent_b=b.instruction, score_b=b.dev_score)
        else:
            prompt = MUTATION_TEMPLATE.format(task_name=self.task.name, parent=rnd.choice(self.history).instruction)
        return self._propose(prompt)

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        np, vectorizer, gpr = self._surrogate()
        warnings.filterwarnings("ignore", module="sklearn")  # kernel-bound notices; harmless here
        rnd = random.Random(self.seed)
        self._score(task_desc, dev_examples)

        while len(self.history) < self.init_random and self._can_afford(1, 1, dev_examples):
            self._score(self._draw(rnd), dev_examples)

        while self._can_afford(1, 1, dev_examples):
            try:
                X = vectorizer.fit_transform([c.instruction for c in self.history]).toarray()
                gpr.fit(X, np.array([c.dev_score for c in self.history], dtype=float))
            except Exception:
                self._score(self._draw(rnd), dev_examples)  # surrogate failed: one random step
                continue
            # proposals cost one call each; leave room to score the winner
            room = self.budget - self.backend.stats.calls - self._score_cost(dev_examples)
            n = max(1, min(self.candidate_pool, room))
            candidates = list(dict.fromkeys(self._draw(rnd) for _ in range(n)))
            mu, sigma = gpr.predict(vectorizer.transform(candidates).toarray(), return_std=True)
            cost = np.array([max(1, len(c.split())) for c in candidates], dtype=float)
            self._score(candidates[int(np.argmax((mu + self.kappa * sigma) / cost))], dev_examples)

        return self._best()
