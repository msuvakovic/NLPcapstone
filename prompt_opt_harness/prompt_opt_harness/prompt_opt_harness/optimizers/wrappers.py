"""Variants that change how a candidate prompt is scored. Each takes any optimizer class and returns a new
one, so "DRO-OPRO", "SAPO-OPRO", "Reg-OPRO", "Entropy-OPRO" and "Guard-OPRO" all come from the same recipe.
From Kanika's and Albert's branches, rewritten to share this harness's budget guard and per-question logs.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Type

from ..prompts import PARAPHRASE_TEMPLATE
from .base import Optimizer, evaluate_detailed


def create_regularized_optimizer(base: Type[Optimizer], penalty_per_word: float = 0.005) -> Type[Optimizer]:
    """Subtract a small penalty per word, so between equal prompts the shorter one wins."""

    class Regularized(base):
        name = f"Reg-{base.name}"

        def _dev_score(self, instruction, dev_examples):
            return max(0.0, super()._dev_score(instruction, dev_examples) - penalty_per_word * len(instruction.split()))

    return Regularized


def create_dro_optimizer(base: Type[Optimizer], num_splits: int = 3, split_ratio: float = 0.5) -> Type[Optimizer]:
    """Score a prompt by its worst accuracy over fixed random subsets of the tuning questions.

    Worst case over subsets of one dataset, not real domain-group DRO.
    """
    if num_splits < 1 or not 0 < split_ratio <= 1:
        raise ValueError("num_splits must be at least 1 and split_ratio in (0, 1]")

    class DRO(base):
        name = f"DRO-{base.name}"

        def _dev_score(self, instruction, dev_examples):
            if not dev_examples:
                return 0.0
            ids = sorted(ex.id for ex in dev_examples)
            if len(set(ids)) != len(ids):
                raise ValueError("DRO needs unique question ids")
            rng = random.Random(self.seed)  # the same subsets face every candidate
            size = max(1, int(len(ids) * split_ratio))
            subsets = [rng.sample(ids, size) for _ in range(num_splits)]
            _, records = evaluate_detailed(self.backend, instruction, dev_examples, task=self.task)
            self._record(records)
            correct = {r["id"]: r["correct"] for r in records}
            return min(sum(correct[i] for i in group) / len(group) for group in subsets)

    return DRO


def create_sapo_optimizer(base: Type[Optimizer], num_perturbations: int = 2) -> Type[Optimizer]:
    """Sharpness-aware: score a prompt and some paraphrases of it, and keep the worst score (a flat optimum)."""

    class SAPO(base):
        name = f"SAPO-{base.name}"

        def _score_cost(self, dev_examples):
            return (num_perturbations + 1) * len(dev_examples) + num_perturbations

        def _dev_score(self, instruction, dev_examples):
            if not dev_examples:
                return 0.0
            own = super()._dev_score(instruction, dev_examples)  # logged as this candidate's record
            scores = [own]
            for _ in range(num_perturbations):
                variant = self._propose(PARAPHRASE_TEMPLATE.format(task_name=self.task.name, instruction=instruction))
                scores.append(evaluate_detailed(self.backend, variant, dev_examples, task=self.task)[0])
            return min(scores)

    return SAPO


def create_entropy_regularized_optimizer(base: Type[Optimizer], num_samples: int = 5, entropy_weight: float = 0.2,
                                         temperature: float = 0.7) -> Type[Optimizer]:
    """Penalize prompts whose answers change from sample to sample.

    Each question is answered `num_samples` times at a sampling temperature; correctness is the majority
    answer, and the score is reduced by the average normalized entropy of the answers.
    """

    class Entropy(base):
        name = f"Entropy-{base.name}"

        def _score_cost(self, dev_examples):
            return num_samples * len(dev_examples)

        def _dev_score(self, instruction, dev_examples):
            if not dev_examples:
                return 0.0
            records, total_entropy = [], 0.0
            for ex in dev_examples:
                prompt = self.task.build_prompt(instruction, ex.text)
                responses, answers, last_meta = [], [], {}
                for i in range(num_samples):
                    responses.append(self.backend.generate(prompt, temperature=temperature, seed=self.seed * 1000 + i))
                    last_meta = dict(self.backend.last_meta)
                    answers.append(self.task.extract(responses[-1]))
                counts = Counter("INVALID" if a is None else str(a) for a in answers)
                majority = counts.most_common(1)[0][0]
                majority_response = next(r for r, a in zip(responses, answers) if ("INVALID" if a is None else str(a)) == majority)
                entropy = -sum((c / num_samples) * math.log(c / num_samples) for c in counts.values())
                total_entropy += entropy / (math.log(num_samples) if num_samples > 1 else 1.0)
                records.append({
                    "id": ex.id, "domain": ex.domain, "question": ex.text, "gold": ex.label,
                    "response": majority_response, "predicted": None if majority == "INVALID" else majority,
                    "correct": self.task.is_correct(majority_response, ex.label),
                    "well_formed": self.task.well_formed(majority_response),
                    "finish_reason": last_meta.get("finish_reason"), "truncated": last_meta.get("finish_reason") == "length",
                    "output_tokens": last_meta.get("output_tokens"),
                })
            accuracy = self._record(records)
            return max(0.0, accuracy - entropy_weight * total_entropy / len(dev_examples))

    return Entropy


def create_guarded_optimizer(base: Type[Optimizer]) -> Type[Optimizer]:
    """After the search, the harness compares the winner with the starting prompt on a separate source sample
    and keeps the winner only if it is strictly better there (Albert's confirmation guard). The search itself
    is unchanged."""

    class Guarded(base):
        name = f"Guard-{base.name}"
        needs_confirm = True

    return Guarded
