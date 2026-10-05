"""Offline regression tests: final data must never influence prompt selection."""
import random
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompt_opt_harness.datasets import Dataset, DomainSplit, Example
from prompt_opt_harness.harness import run_experiment, validate_splits
from prompt_opt_harness.optimizers import (OPRO, OracleOPRO, create_dro_optimizer,
    create_regularized_optimizer, create_sapo_optimizer, create_entropy_regularized_optimizer)
from prompt_opt_harness.optimizers.base import Candidate, Optimizer


def dataset(target_label='Positive', target_text='target secret text'):
    return Dataset('source', DomainSplit(
        [Example('dev', 'source development text', 'Positive', 'source')],
        [Example('test', 'source heldout text', 'Positive', 'source')]),
        {'target': [Example('target', target_text, target_label, 'target')]},
        base_instruction='Base instruction')


class RecordingBackend:
    def __init__(self):
        self.stats = SimpleNamespace(calls=0, wall_time=0)
        self.frozen = None
        self.search_prompts = []
        self.final_prompts = []

    def generate(self, prompt):
        self.stats.calls += 1
        if self.frozen is None:
            self.search_prompts.append(prompt)
            if '\n\nText:' in prompt:
                assert 'Text: source development text\n' in prompt, 'Final text reached search'
        else:
            self.final_prompts.append(prompt)
            assert prompt.startswith(self.frozen), 'Final prompt changed after selection'
        if '\n\nText:' not in prompt:
            return 'Revised source instruction'
        return 'Negative' if prompt.startswith('Base instruction') else 'Positive'


class TrackedOPRO(OPRO):
    def optimize(self, *args, **kwargs):
        selected = super().optimize(*args, **kwargs)
        self.backend.frozen = selected.instruction
        return selected


class IsolationTests(unittest.TestCase):
    def setUp(self):
        # Unit tests must never download embedding weights or send network requests.
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {'sentence_transformers': None}).start()
        patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected network call')).start()

    def test_target_labels_and_text_cannot_change_search(self):
        results, backends = [], []
        for label, text in [('Positive','target secret text'), ('Negative','target secret text'),
                            ('Negative','entirely different target content')]:
            backend = RecordingBackend()
            results.append(run_experiment('OPRO', TrackedOPRO, lambda: backend,
                                          dataset(label,text), budget=7, seed=42))
            backends.append(backend)
        self.assertEqual({r.best_instruction for r in results}, {'Revised source instruction'})
        self.assertTrue(all(b.search_prompts == backends[0].search_prompts for b in backends))
        self.assertTrue(all(r.candidate_pool == results[0].candidate_pool for r in results))
        self.assertEqual([r.ood_accs['target'] for r in results], [1.,0.,0.])
        self.assertEqual(len(backends[0].final_prompts), 2)
        self.assertEqual(results[0].to_dict()['evaluation_protocol'], 'source-dev-only-v1')

    def test_all_optimizers_receive_only_source_dev(self):
        class Probe(Optimizer):
            # Deliberately has no target-data parameter.
            def optimize(self, task_desc, dev_examples):
                self.assert_source = [e.id for e in dev_examples]
                assert self.assert_source == ['dev']
                dev_examples.clear()  # Cannot mutate the dataset's dev container.
                selected = Candidate(task_desc, 1.)
                self.history.append(selected)
                return selected
        data = dataset()
        with patch('prompt_opt_harness.harness.evaluate', return_value=1.):
            run_experiment('Probe', Probe, RecordingBackend, data, budget=1)
        self.assertEqual(len(data.source.dev), 1)

    def test_direct_and_wrapped_opro_reject_target_data_before_calls(self):
        classes = [OPRO, create_dro_optimizer(OPRO), create_regularized_optimizer(OPRO),
                   create_sapo_optimizer(OPRO), create_entropy_regularized_optimizer(OPRO)]
        for cls in classes:
            for targets in ({}, dataset().ood):
                with self.subTest(optimizer=cls.name, targets=bool(targets)):
                    backend = RecordingBackend()
                    optimizer = cls(backend, budget=7)
                    with self.assertRaisesRegex(ValueError, 'source dev data only'):
                        optimizer.optimize('Base instruction', dataset().source.dev, targets)
                    self.assertEqual(backend.stats.calls, 0)
                    self.assertEqual(optimizer.history, [])

    def test_candidate_helper_rejects_target_data(self):
        backend = RecordingBackend()
        with self.assertRaises(ValueError):
            OPRO(backend,7)._eval_candidate('Base', dataset().source.dev, dataset().ood)
        self.assertEqual(backend.stats.calls, 0)

    def test_oracle_cannot_reintroduce_test_training(self):
        for targets in (None, dataset().ood):
            backend = RecordingBackend()
            with self.assertRaisesRegex(ValueError, 'OracleOPRO is disabled'):
                OracleOPRO(backend,7).optimize('Base', dataset().source.dev, targets)
            self.assertEqual(backend.stats.calls, 0)

    def test_overlapping_ids_or_normalized_text_rejected_before_backend(self):
        for field, value in [('id','dev'), ('text','  SOURCE   development TEXT  ')]:
            for location in ('source_test','target'):
                with self.subTest(field=field, location=location):
                    data = dataset()
                    row = replace(data.ood['target'][0], **{field:value})
                    if location == 'target':
                        data.ood['target'] = [row]
                    else:
                        data.source = DomainSplit(data.source.dev, [row])
                    factory = Mock()
                    with self.assertRaisesRegex(ValueError, 'disjoint splits'):
                        run_experiment('OPRO', OPRO, factory, data, budget=7)
                    factory.assert_not_called()

    def test_final_domains_cannot_reuse_examples(self):
        data = dataset()
        data.ood['other'] = [replace(data.ood['target'][0], id='other')]
        with self.assertRaises(ValueError): validate_splits(data)

    def test_search_seed_is_independent_of_global_rng(self):
        traces = []
        for global_seed in (1, 999):
            random.seed(global_seed)
            backend = RecordingBackend()
            OPRO(backend,7,seed=42).optimize('Base instruction',dataset().source.dev)
            traces.append(backend.search_prompts)
        self.assertEqual(traces[0], traces[1])

    def test_old_history_cannot_affect_a_new_search(self):
        backend = RecordingBackend()
        optimizer = OPRO(backend,7,seed=42)
        optimizer.history = [Candidate('Stale target-trained winner', 100., {'target':1.},100.)]
        result = optimizer.optimize('Base instruction',dataset().source.dev)
        self.assertEqual(result.instruction, 'Revised source instruction')
        self.assertNotIn('Stale target-trained winner', '\n'.join(backend.search_prompts))

    def test_duplicate_proposals_do_not_bypass_batch_budget_stop(self):
        backend = RecordingBackend()
        OPRO(backend,7,seed=42).optimize('Base instruction',dataset().source.dev)
        self.assertLessEqual(backend.stats.calls, 7)


if __name__ == '__main__':
    unittest.main()
