from __future__ import annotations

from typing import List, Optional
import random

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from .base import Candidate, Optimizer
from ..prompts import MUTATION_TEMPLATE, CROSSOVER_TEMPLATE, build_classification_prompt


class EmbeddingBayesOpt(Optimizer):
    """Bayesian optimizer using sentence-transformers embeddings as surrogate features."""

    name = "EmbeddingBayesOpt"

    def __init__(self, backend, budget: int, seed: int = 0, init_random: int = 6, kappa: float = 1.96, model_name: str = 'all-MiniLM-L6-v2', candidate_pool: int = 40):
        super().__init__(backend, budget, seed)
        self.init_random = init_random
        self.kappa = kappa
        self.model_name = model_name
        if SentenceTransformer is not None:
            self.embedder = SentenceTransformer(model_name)
        else:
            self.embedder = None
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
        self.gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self.embedder is not None:
            return np.array(self.embedder.encode(texts, convert_to_numpy=True))
        # fallback: simple char-level features
        return np.array([[len(t), sum(c.isalpha() for c in t)] for t in texts], dtype=float)

    def _estimate_cost(self, instruction: str, sample_text: Optional[str]) -> float:
        try:
            if sample_text:
                prompt = build_classification_prompt(instruction, sample_text, self.task)
            else:
                prompt = instruction
            if hasattr(self.backend, 'tokenize'):
                tokens = len(self.backend.tokenize(prompt))
            else:
                tokens = len(prompt.split())
            return float(tokens)
        except Exception:
            return float(max(1, len(instruction.split())))

    def optimize(self, task_desc: str, dev_examples, ood_examples: dict = None) -> Candidate:
        rnd = random.Random(self.seed)
        sample_text = dev_examples[0].text if dev_examples else None

        pool: List[Candidate] = [Candidate(instruction=task_desc, dev_score=self._dev_score(task_desc, dev_examples))]

        while len(pool) < self.init_random and self.backend.stats.calls < self.budget:
            parent = rnd.choice(pool)
            prompt = MUTATION_TEMPLATE.format(parent=parent.instruction)
            inst = self.backend.generate(prompt).strip()
            pool.append(Candidate(instruction=inst))

        for c in pool:
            if c.dev_score is None:
                c.dev_score = self._dev_score(c.instruction, dev_examples)
        self.history.extend(pool)

        while self.backend.stats.calls < self.budget:
            instrs = [c.instruction for c in self.history]
            ys = np.array([c.dev_score for c in self.history], dtype=float)

            X = self._embed(instrs)

            try:
                self.gpr.fit(X, ys)
            except Exception:
                parent = rnd.choice(self.history)
                prompt = MUTATION_TEMPLATE.format(parent=parent.instruction)
                candidate_inst = self.backend.generate(prompt).strip()
                cand = Candidate(instruction=candidate_inst, dev_score=self._dev_score(candidate_inst, dev_examples))
                self.history.append(cand)
                continue

            # propose candidates
            candidates = []
            for _ in range(getattr(self, 'candidate_pool', 40)):
                if rnd.random() < 0.5 and len(self.history) >= 2:
                    a, b = rnd.sample(self.history, 2)
                    prompt = CROSSOVER_TEMPLATE.format(parent_a=a.instruction, score_a=a.dev_score or 0.0, parent_b=b.instruction, score_b=b.dev_score or 0.0)
                else:
                    parent = rnd.choice(self.history)
                    prompt = MUTATION_TEMPLATE.format(parent=parent.instruction)
                inst = self.backend.generate(prompt).strip()
                candidates.append(inst)

            X_cand = self._embed(candidates)
            mu, sigma = self.gpr.predict(X_cand, return_std=True)

            acq = mu + self.kappa * sigma
            costs = [self._estimate_cost(c, sample_text) for c in candidates]
            scores = acq / (np.array(costs) + 1e-6)

            idx = int(np.argmax(scores))
            selected = candidates[idx]

            if self.backend.stats.calls >= self.budget:
                break
            dev_score = self._dev_score(selected, dev_examples)
            cand = Candidate(instruction=selected, dev_score=dev_score)
            self.history.append(cand)

        return max(self.history, key=lambda c: c.dev_score)
