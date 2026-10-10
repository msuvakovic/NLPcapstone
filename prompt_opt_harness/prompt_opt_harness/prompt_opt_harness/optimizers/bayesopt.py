from __future__ import annotations

from typing import List, Optional
import random

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.feature_extraction.text import TfidfVectorizer

from ..prompts import MUTATION_TEMPLATE, CROSSOVER_TEMPLATE, build_classification_prompt
from ..tasks import SENTIMENT, TaskSpec
from .base import Candidate, Optimizer


class CostAwareBayesOpt(Optimizer):
    """Simple cost-aware Bayesian optimizer over prompt strings.

    Surrogate: TF-IDF features + Gaussian Process.
    Acquisition: UCB (mu + kappa*sigma) divided by estimated cost.
    """

    name = "CostBayesOpt"

    def __init__(self, backend, budget: int, seed: int = 0, init_random: int = 4, kappa: float = 1.96, candidate_pool: int = 20, task: TaskSpec = SENTIMENT):
        super().__init__(backend, budget, seed, task)
        self.candidate_pool = candidate_pool
        self.init_random = init_random
        self.kappa = kappa
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1024)
        # Kernel with noise term
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
        self.gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    def _estimate_cost(self, instruction: str, sample_text: Optional[str]) -> float:
        # Estimate cost by tokenizing a representative prompt with the sample_text if available
        try:
            if sample_text:
                prompt = build_classification_prompt(instruction, sample_text, self.task)
            else:
                prompt = instruction
            # Use backend.tokenizer if available (some backends expose stats/tokenizer), else crude wc
            if hasattr(self.backend, 'tokenize'):
                tokens = len(self.backend.tokenize(prompt))
            else:
                tokens = len(prompt.split())
            # cost in abstract units (tokens); downstream compare_runs maps tokens -> USD
            return float(tokens)
        except Exception:
            return float(max(1, len(instruction.split())))

    def _propose(self, rnd) -> str:
        if rnd.random() < 0.5 and len(self.history) >= 2:
            a, b = rnd.sample(self.history, 2)
            prompt = CROSSOVER_TEMPLATE.format(parent_a=a.instruction, score_a=a.dev_score,
                                               parent_b=b.instruction, score_b=b.dev_score)
        else:
            prompt = MUTATION_TEMPLATE.format(parent=rnd.choice(self.history).instruction)
        return self._generate(prompt)

    def _search(self, task_desc: str, dev_examples) -> Candidate:
        rnd = random.Random(self.seed)
        sample_text = dev_examples[0].text
        score_calls = self._evaluation_calls(dev_examples)
        self._score(task_desc, dev_examples)

        # Initial design: random mutations, each scored as soon as it exists.
        while len(self.history) < self.init_random and self._can_afford(1, 1, dev_examples):
            parent = rnd.choice(self.history)
            self._score(self._generate(MUTATION_TEMPLATE.format(parent=parent.instruction)), dev_examples)

        while self._can_afford(1, 1, dev_examples):
            instrs = [c.instruction for c in self.history]
            ys = np.array([c.dev_score for c in self.history], dtype=float)
            try:
                X = self.vectorizer.fit_transform(instrs).toarray()
                self.gpr.fit(X, ys)
            except Exception:
                # Surrogate failed: fall back to one random exploration step.
                self._score(self._propose(rnd), dev_examples)
                continue

            # Proposals cost one request each, so only draw as many as still
            # leave room to score the winner.
            n_proposals = min(self.candidate_pool, self._remaining() - score_calls)
            candidates = list(dict.fromkeys(self._propose(rnd) for _ in range(n_proposals)))

            X_cand = self.vectorizer.transform(candidates).toarray()
            mu, sigma = self.gpr.predict(X_cand, return_std=True)
            acq = mu + self.kappa * sigma
            costs = np.array([self._estimate_cost(c, sample_text) for c in candidates])
            selected = candidates[int(np.argmax(acq / (costs + 1e-6)))]
            self._score(selected, dev_examples)

        return self._best()
