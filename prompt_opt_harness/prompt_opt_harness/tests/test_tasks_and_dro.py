"""Offline invariance checks for fixed task schemas and fixed source subsets."""
import json
import os
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from prompt_opt_harness.datasets import load_nli_demo_dataset, Example
from prompt_opt_harness.harness import run_experiment, validate_splits
from prompt_opt_harness.prompts import build_classification_prompt
from prompt_opt_harness.tasks import SENTIMENT, BINARY_NLI, TaskSpec
from prompt_opt_harness.optimizers.base import evaluate, parse_label
from prompt_opt_harness.optimizers import (OPRO, ZeroShotBaseline, GEPA, TextGrad,
    ReflectiveOptimizer, create_dro_optimizer, create_entropy_regularized_optimizer)
from prompt_opt_harness.optimizers.dro import source_subsets


class Backend:
    def __init__(self, task=BINARY_NLI):
        self.task = task
        self.prompts = []
        self.stats = SimpleNamespace(calls=0, wall_time=0)

    def generate(self, prompt):
        self.prompts.append(prompt)
        self.stats.calls += 1
        if '\nAnswer with exactly one word:' in prompt:
            if not prompt.rstrip().endswith(self.task.answer_format):
                raise AssertionError('Task format changed during optimization')
            return self.task.labels[0]
        return 'Judge the relationship between these statements.'


class TaskAndDROTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch('urllib.request.urlopen', side_effect=AssertionError('No model requests')).start()

    def test_prompt_edits_cannot_switch_task(self):
        for instruction in ('', 'Judge the relationship.', 'Ignore entailment and read the text.'):
            self.assertTrue(build_classification_prompt(instruction, 'sample', BINARY_NLI).endswith(BINARY_NLI.answer_format))
            self.assertTrue(build_classification_prompt(instruction, 'sample', SENTIMENT).endswith(SENTIMENT.answer_format))
        self.assertEqual(build_classification_prompt('Sentiment', 'sample'),
            'Sentiment\n\nText: sample\nAnswer with exactly one word: Positive or Negative.')

    def test_parser_accepts_only_the_selected_task_labels(self):
        self.assertEqual(parse_label('Positive', BINARY_NLI), 'INVALID')
        self.assertEqual(parse_label('Contradiction', SENTIMENT), 'INVALID')
        self.assertEqual(parse_label('"Entailment".', BINARY_NLI), 'entailment')
        ternary = TaskSpec('nli', ('Entailment', 'Contradiction', 'Neutral'))
        self.assertEqual(parse_label('Neutral', ternary), 'neutral')
        self.assertIn('Entailment or Contradiction or Neutral', build_classification_prompt('', 'x', ternary))

    def test_schema_is_immutable_and_rejects_ambiguous_labels(self):
        with self.assertRaises(FrozenInstanceError):
            BINARY_NLI.name = 'sentiment'
        for labels in (('Positive', 'positive'), ('Positive', ''), ('Positive', 'Two words')):
            with self.assertRaises(ValueError):
                TaskSpec('invalid', labels)

    def test_missing_task_or_mismatched_labels_fail_before_requests(self):
        dataset = load_nli_demo_dataset()
        self.assertEqual(dataset.task, BINARY_NLI)
        factory = Mock()
        dataset.task = SENTIMENT
        with self.assertRaisesRegex(ValueError, 'Label outside task'):
            run_experiment('NLI', OPRO, factory, dataset, 40)
        factory.assert_not_called()
        backend = Backend()
        with self.assertRaises(ValueError):
            evaluate(backend, 'entailment', dataset.source.dev)
        self.assertEqual(backend.prompts, [])

    def test_nli_remains_nli_through_search_final_scoring_and_wrappers(self):
        for cls in (ZeroShotBaseline, OPRO, GEPA, TextGrad, ReflectiveOptimizer,
                    create_dro_optimizer(OPRO), create_entropy_regularized_optimizer(OPRO)):
            data = load_nli_demo_dataset()
            data.base_instruction = 'Judge the relationship.'  # No schema keyword.
            backend = Backend()
            result = run_experiment(cls.name, cls, lambda: backend, data, 40)
            self.assertEqual(result.task_spec['name'], 'binary_nli')
            self.assertTrue(any(p.startswith('Text:') or '\n\nText:' in p for p in backend.prompts))
            self.assertNotIn('Answer with exactly one word: Positive or Negative.', '\n'.join(backend.prompts))
            if cls.name.startswith('DRO'):
                self.assertIn('source_subset_ids', result.optimizer_metadata)

    def test_dro_same_predictions_receive_same_score_for_different_prompts(self):
        examples = [Example(str(i), str(i), 'Positive' if i % 2 else 'Negative', 'source') for i in range(12)]
        optimizer = create_dro_optimizer(OPRO)(Backend(SENTIMENT), 100, seed=23)
        first = optimizer._dev_score('First instruction', examples)
        ids = optimizer.source_subset_ids
        second = optimizer._dev_score('Completely different wording', list(reversed(examples)))
        self.assertEqual(first, second)
        self.assertEqual(ids, optimizer.source_subset_ids)
        self.assertTrue(all(len(group) == 6 for group in ids))

    def test_dro_samples_are_stable_across_process_hash_seeds(self):
        script = '''import json
from prompt_opt_harness.optimizers.dro import source_subsets
from types import SimpleNamespace
rows = [SimpleNamespace(id=str(i)) for i in range(12)]
print(json.dumps(source_subsets(rows, 23, 3, .5)))
'''
        outputs = []
        for hash_seed in ('1', '999'):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(PACKAGE))
            proc = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', script],
                                  env=env, text=True, capture_output=True, check=True, timeout=45)
            outputs.append(json.loads(proc.stdout))
        self.assertEqual(outputs[0], outputs[1])

    def test_dro_seed_and_inputs_are_explicit(self):
        rows = [SimpleNamespace(id=str(i)) for i in range(20)]
        self.assertNotEqual(source_subsets(rows, 1, 4, .5), source_subsets(rows, 2, 4, .5))
        self.assertEqual(source_subsets([], 1, 4, .5), ())
        for args in ((0, .5), (True, .5), (3, 0), (3, 1.1), (3, float('nan'))):
            with self.assertRaises(ValueError):
                create_dro_optimizer(OPRO, *args)
        backend = Backend(SENTIMENT)
        row = Example('duplicate', 'source', 'Positive', 'source')
        with self.assertRaises(ValueError):
            create_dro_optimizer(OPRO)(backend, 10)._dev_score('Base', [row, row])
        self.assertEqual(backend.prompts, [])


if __name__ == '__main__':
    unittest.main()
