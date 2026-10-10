"""Random-search baseline and prompt-trait analysis (offline)."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompt_opt_harness.budget import CallBudgetExceeded
from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import MockBackend
from prompt_opt_harness.optimizers import RandomSearch
from prompt_opt_harness.traits import (correlations, load_runs, prompt_traits, spearman,
                                       trait_table)

SEED = "Classify sentiment as Positive or Negative."


class Recording(MockBackend):
    def _generate(self, prompt):
        self.sent.append(prompt)
        return super()._generate(prompt)


def recording_backend(data):
    backend = Recording(data.lookup())
    backend.sent = []
    return backend


class RandomSearchTests(unittest.TestCase):
    def test_stays_in_budget_and_scores_several_candidates(self):
        data = load_demo_dataset()
        dev = data.source.dev
        for budget in (len(dev), 40, 100):
            backend = MockBackend(data.lookup())
            opt = RandomSearch(backend, budget=budget, seed=0)
            best = opt.optimize(SEED, dev)
            self.assertLessEqual(backend.stats.calls, budget)
            self.assertEqual(best.dev_score, max(c.dev_score for c in opt.history))
        self.assertGreater(len(opt.history), 3)

    def test_proposals_never_see_scores_or_other_candidates(self):
        # This is what makes it a control: every proposal rewrites the seed only.
        data = load_demo_dataset()
        backend = recording_backend(data)
        RandomSearch(backend, budget=100, seed=0).optimize(SEED, data.source.dev)
        proposals = [p for p in backend.sent if "Variant #" in p]
        self.assertGreater(len(proposals), 3)
        for p in proposals:
            self.assertIn(f"Instruction: {SEED}\n", p)
            self.assertNotRegex(p, r"[Ss]core|[Aa]ccuracy|0\.\d")
        self.assertEqual(len(set(proposals)), len(proposals))  # distinct variants

    def test_seeded_and_rejects_target_data(self):
        data = load_demo_dataset()
        a = RandomSearch(MockBackend(data.lookup()), 80, seed=3).optimize(SEED, data.source.dev)
        b = RandomSearch(MockBackend(data.lookup()), 80, seed=3).optimize(SEED, data.source.dev)
        self.assertEqual(a.instruction, b.instruction)
        backend = MockBackend(data.lookup())
        with self.assertRaises(ValueError):
            RandomSearch(backend, 80).optimize(SEED, data.source.dev, ood_examples=data.ood)
        with self.assertRaises(CallBudgetExceeded):
            RandomSearch(backend, len(data.source.dev) - 1).optimize(SEED, data.source.dev)
        self.assertEqual(backend.stats.calls, 0)

    def test_runs_through_harness(self):
        data = load_demo_dataset()
        result = run_experiment("Random-search", RandomSearch, lambda: MockBackend(data.lookup()), data, 60)
        self.assertLessEqual(result.search_api_calls, 60)
        self.assertEqual(set(result.ood_accs), set(data.ood))


class TraitTests(unittest.TestCase):
    def test_text_traits(self):
        t = prompt_traits('Label the review Positive or Negative. Ignore sarcasm, e.g. "great, broke again".',
                          labels=("Positive", "Negative"))
        self.assertEqual(t["words"], 12)
        self.assertEqual(t["sentences"], 2)
        self.assertEqual(t["names_labels"], 1.0)
        self.assertEqual(t["has_examples"], 1.0)
        self.assertGreaterEqual(t["guidance_clauses"], 1)
        self.assertIsNone(t["source_vocab_rate"])
        self.assertEqual(prompt_traits("")["words"], 0)

    def test_domain_vocabulary_and_copied_spans(self):
        source = ["the battery on this phone died after two days", "great screen and fast shipping"]
        target = ["the movie was great", "worst film this year"]
        generic = prompt_traits("Decide whether the text is positive or negative.", (), source, target)
        leaky = prompt_traits("Negative if the battery died after two days or shipping was slow.", (), source, target)
        self.assertEqual(generic["source_vocab_rate"], 0.0)
        self.assertGreater(leaky["source_vocab_rate"], 0.3)
        self.assertEqual(generic["copied_4grams"], 0)
        self.assertGreaterEqual(leaky["copied_4grams"], 1)  # "battery died after two days"

    def test_spearman(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertAlmostEqual(spearman([1, 1, 2, 3], [1, 2, 3, 4]), 0.9486832980505138)  # ties
        self.assertIsNone(spearman([1, 2], [1, 2]))
        self.assertIsNone(spearman([5, 5, 5], [1, 2, 3]))
        self.assertAlmostEqual(spearman([1, None, 2, 3, math.nan], [1, 9, 2, 3, 4]), 1.0)

    def test_end_to_end_from_run_logs(self):
        data = load_demo_dataset()
        runs = []
        for seed in range(4):
            result = run_experiment(f"RS-{seed}", RandomSearch, lambda: MockBackend(data.lookup()), data, 60, seed=seed)
            runs.append(result.to_dict())
        with tempfile.TemporaryDirectory() as tmp:
            for i, run in enumerate(runs):
                Path(tmp, f"run{i}.json").write_text(json.dumps(run))
            Path(tmp, "not_a_run.json").write_text("{}")
            loaded = load_runs([tmp])
        self.assertEqual(len(loaded), 4)
        rows = trait_table(loaded, source_texts=[e.text for e in data.source.dev],
                           target_texts=[e.text for v in data.ood.values() for e in v])
        self.assertTrue(all(r["names_labels"] is not None for r in rows))  # labels from task_spec
        corr = {c["trait"]: c for c in correlations(rows)}
        self.assertIn("source_vocab_rate", corr)
        self.assertEqual(corr["words"]["n"], 4)


if __name__ == "__main__":
    unittest.main()
