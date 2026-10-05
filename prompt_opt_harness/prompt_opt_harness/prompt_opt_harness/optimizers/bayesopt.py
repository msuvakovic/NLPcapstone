from __future__ import annotations

from typing import List, Optional
import random

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.feature_extraction.text import TfidfVectorizer

from ..prompts import MUTATION_TEMPLATE, CROSSOVER_TEMPLATE, build_classification_prompt
from .base import Candidate, Optimizer


class CostAwareBayesOpt(Optimizer):
    """Simple cost-aware Bayesian optimizer over prompt strings.

    Surrogate: TF-IDF features + Gaussian Process.
    Acquisition: UCB (mu + kappa*sigma) divided by estimated cost.
    """

    name = "CostBayesOpt"

    def __init__(self, backend, budget: int, seed: int = 0, init_random: int = 4, kappa: float = 1.96, candidate_pool: int = 20):
        super().__init__(backend, budget, seed)
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

    def optimize(self, task_desc: str, dev_examples, ood_examples: dict = None) -> Candidate:
        rnd = random.Random(self.seed)
        sample_text = dev_examples[0].text if dev_examples else None

        # Initial pool
        pool: List[Candidate] = [Candidate(instruction=task_desc, dev_score=self._dev_score(task_desc, dev_examples))]

        # Seed with a few mutated candidates
        while len(pool) < self.init_random and self.backend.stats.calls < self.budget:
            parent = rnd.choice(pool)
            prompt = MUTATION_TEMPLATE.format(parent=parent.instruction)
            inst = self.backend.generate(prompt).strip()
            pool.append(Candidate(instruction=inst))

        # Score unscored
        for c in pool:
            if c.dev_score is None:
                c.dev_score = self._dev_score(c.instruction, dev_examples)
        self.history.extend(pool)

        # Main loop: fit surrogate and propose candidates until budget
        while self.backend.stats.calls < self.budget:
            instrs = [c.instruction for c in self.history]
            ys = np.array([c.dev_score for c in self.history], dtype=float)

            # Vectorize
            X = self.vectorizer.fit_transform(instrs).toarray()

            # Fit surrogate
            try:
                self.gpr.fit(X, ys)
            except Exception:
                # Fallback: random exploration
                parent = rnd.choice(self.history)
                prompt = MUTATION_TEMPLATE.format(parent=parent.instruction)
                candidate_inst = self.backend.generate(prompt).strip()
                cand = Candidate(instruction=candidate_inst, dev_score=self._dev_score(candidate_inst, dev_examples))
                self.history.append(cand)
                continue

            # Propose a set of candidates via mutation/crossover
            candidates = []
            for _ in range(getattr(self, 'candidate_pool', 20)):
                if rnd.random() < 0.5 and len(self.history) >= 2:
                    a, b = rnd.sample(self.history, 2)
                    prompt = CROSSOVER_TEMPLATE.format(parent_a=a.instruction, score_a=a.dev_score or 0.0, parent_b=b.instruction, score_b=b.dev_score or 0.0)
                else:
                    parent = rnd.choice(self.history)
                    prompt = MUTATION_TEMPLATE.format(parent=parent.instruction)
                inst = self.backend.generate(prompt).strip()
                candidates.append(inst)

            # Vectorize candidates using same vectorizer (transform)
            X_cand = self.vectorizer.transform(candidates).toarray()
            mu, sigma = self.gpr.predict(X_cand, return_std=True)

            best_y = max(ys)
            acq = mu + self.kappa * sigma

            # Compute cost estimates and score acquisition per cost
            costs = [self._estimate_cost(c, sample_text) for c in candidates]
            # Avoid divide by zero
            scores = acq / (np.array(costs) + 1e-6)

            # Select best candidate by score
            idx = int(np.argmax(scores))
            selected = candidates[idx]

            # Evaluate selected candidate
            if self.backend.stats.calls >= self.budget:
                break
            dev_score = self._dev_score(selected, dev_examples)
            cand = Candidate(instruction=selected, dev_score=dev_score)
            self.history.append(cand)

        return max(self.history, key=lambda c: c.dev_score)
