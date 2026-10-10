"""Budget and isolation guarantees for every search optimizer, not just OPRO."""
import dataclasses
import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompt_opt_harness.budget import CallBudgetExceeded
from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.llm_backends import MockBackend
from prompt_opt_harness.optimizers import (
    OPRO, EvoPromptLite, GEPA, TextGrad, PBT, ReflectiveOptimizer, EnsembleOptimizer,
    CostAwareBayesOpt, EmbeddingBayesOpt, create_dro_optimizer, create_sapo_optimizer,
    create_entropy_regularized_optimizer, create_regularized_optimizer,
)

SEARCH_OPTIMIZERS = (OPRO, EvoPromptLite, GEPA, TextGrad, PBT, ReflectiveOptimizer,
                     EnsembleOptimizer, CostAwareBayesOpt, EmbeddingBayesOpt)
WRAPPED = (create_dro_optimizer(GEPA), create_sapo_optimizer(EvoPromptLite),
           create_entropy_regularized_optimizer(PBT), create_regularized_optimizer(TextGrad))
SEED = "Classify sentiment as Positive or Negative."


def make_dev(copies=3):
    data = load_demo_dataset()
    dev = [dataclasses.replace(ex, id=f"{ex.id}-{i}", text=f"{ex.text} ({i})")
           for i in range(copies) for ex in data.source.dev]
    lookup = data.lookup()
    lookup.update({ex.text: (ex.label, ex.domain) for ex in dev})
    return dev, data.ood, lookup


class OptimizerBudgetTests(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")  # sklearn GP convergence noise

    def run_search(self, cls, budget, prior_calls=0, seed=0):
        dev, _, lookup = make_dev()
        backend = MockBackend(lookup)
        for _ in range(prior_calls):
            backend.generate("unrelated earlier request")
        start = backend.stats.calls
        optimizer = cls(backend, budget=budget, seed=seed)
        best = optimizer.optimize(SEED, dev)
        return optimizer, best, backend.stats.calls - start, backend

    def test_search_never_exceeds_budget(self):
        for cls in SEARCH_OPTIMIZERS + WRAPPED:
            for budget in (60, 97, 200):
                with self.subTest(optimizer=cls.name, budget=budget):
                    try:
                        _, best, used, _ = self.run_search(cls, budget)
                    except CallBudgetExceeded:
                        continue  # Wrapper's seed score alone exceeds this budget.
                    self.assertLessEqual(used, budget)
                    self.assertIsNotNone(best.dev_score)

    def test_budget_is_relative_to_search_start(self):
        # Earlier requests on a shared backend must not eat this search's budget.
        for cls in SEARCH_OPTIMIZERS:
            with self.subTest(optimizer=cls.name):
                fresh = self.run_search(cls, 150)[2]
                reused = self.run_search(cls, 150, prior_calls=500)[2]
                self.assertEqual(fresh, reused)

    def test_every_history_entry_is_fully_scored(self):
        for cls in SEARCH_OPTIMIZERS:
            with self.subTest(optimizer=cls.name):
                optimizer, best, _, _ = self.run_search(cls, 130)
                self.assertTrue(all(c.dev_score is not None for c in optimizer.history))
                self.assertIn(best, optimizer.history)
                self.assertEqual(best.dev_score, max(c.dev_score for c in optimizer.history))

    def test_budget_below_one_score_fails_before_any_request(self):
        dev, _, lookup = make_dev()
        for cls in SEARCH_OPTIMIZERS:
            with self.subTest(optimizer=cls.name):
                backend = MockBackend(lookup)
                with self.assertRaises(CallBudgetExceeded):
                    cls(backend, budget=len(dev) - 1).optimize(SEED, dev)
                self.assertEqual(backend.stats.calls, 0)

    def test_target_data_is_rejected_before_any_request(self):
        dev, ood, lookup = make_dev()
        for cls in SEARCH_OPTIMIZERS + WRAPPED:
            with self.subTest(optimizer=cls.name):
                backend = MockBackend(lookup)
                with self.assertRaises(ValueError):
                    cls(backend, budget=200).optimize(SEED, dev, ood_examples=ood)
                self.assertEqual(backend.stats.calls, 0)

    def test_failure_driven_optimizers_do_not_rescore_incumbent(self):
        # Previously each step re-ran the whole dev set on the current best
        # prompt just to find its failures, burning a full score per step.
        class Recording(MockBackend):
            def _generate(self, prompt):
                self.sent.append(prompt)
                return super()._generate(prompt)

        dev, _, lookup = make_dev()
        for cls in (GEPA, TextGrad, ReflectiveOptimizer):
            with self.subTest(optimizer=cls.name):
                backend = Recording(lookup)
                backend.sent = []
                cls(backend, budget=200).optimize(SEED, dev)
                seed_scoring = [p for p in backend.sent if p.startswith(SEED) and "Text:" in p]
                self.assertEqual(len(seed_scoring), len(dev))

    def test_same_seed_same_result(self):
        for cls in (GEPA, PBT, ReflectiveOptimizer, CostAwareBayesOpt):
            with self.subTest(optimizer=cls.name):
                a = self.run_search(cls, 150, seed=7)[1]
                b = self.run_search(cls, 150, seed=7)[1]
                self.assertEqual((a.instruction, a.dev_score), (b.instruction, b.dev_score))


if __name__ == "__main__":
    unittest.main()
