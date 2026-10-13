"""DSPy integration regressions using the real library and an offline backend.

Install requirements-dspy.txt in an isolated environment. Network is blocked,
including during import. These test plumbing, not language-model performance.
"""
import copy
import json
import os
import re
import socket
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['DSPY_CACHEDIR'] = str(Path(__file__).resolve().parents[3] / 'tmp' / 'dspy-test-cache')
with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')):
    import dspy
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

from prompt_opt_harness.budget import BudgetedBackend, CallBudgetExceeded
from prompt_opt_harness.datasets import Dataset, DomainSplit, Example
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import LLMBackend, OpenAICompatibleBackend
from prompt_opt_harness.optimizers.base import evaluate
from prompt_opt_harness.optimizers.dspy_adapters import (
    DSPyGEPA, DSPyMIPROv2, DSPyCOPRO, split_source, _make_bridge, _make_task_adapter,
)
from prompt_opt_harness.prompts import BASE_INSTRUCTION, build_classification_prompt
from prompt_opt_harness.tasks import SENTIMENT, BINARY_NLI


def rows(prefix='source', task=SENTIMENT):
    return [Example(f'{prefix}-{i}', f'{prefix} text {i} {label}', label, prefix)
            for i, label in enumerate(task.labels * 4)]


class Backend(LLMBackend):
    def __init__(self):
        super().__init__()
        self.model = 'offline-model'
        self.temperature = 0.0
        self.max_tokens = 16
        self.requests = []
        self.failures = []

    def _generate(self, prompt):
        self.requests.append((prompt, self.temperature, self.max_tokens))
        if self.failures:
            raise self.failures.pop(0)
        if '\n\nText:' in prompt:
            if prompt.startswith('Improved sentiment'):
                text = prompt.split('\n\nText:', 1)[1].split('\nAnswer', 1)[0]
                return text.split()[-1]
            return 'Negative'
        if prompt.startswith('Complete the following conversation.'):
            messages = json.loads(prompt.split('\n', 1)[1])
            reminder = messages[-1]['content'].split('Respond with the corresponding output fields')[-1]
            fields = re.findall(r'\[\[ ## (\w+) ## \]\]', reminder)
            if fields:
                return '\n'.join(f'[[ ## {f} ## ]]\n' + (
                    'Improved sentiment instruction' if 'instruction' in f else 'Classification task')
                    for f in fields)
        return '```\nImproved sentiment instruction\n```'


class ControlledMIPRO(DSPyMIPROv2):
    def _compile(self, dspy, student, metric, lm, trainset, valset, remaining):
        self.seen_train = trainset
        self.seen_val = valset
        lm(prompt='proposal')
        student.prog.signature = student.prog.signature.with_instructions('Improved sentiment instruction')
        for ex in trainset + valset:
            prediction = student(**ex.inputs())
            assert metric(ex, prediction) == 1.0
        return student


class DSPyTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')).start()
        patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')).start()
        patch.dict(sys.modules, {'tiktoken': None}).start()
        patch.dict(os.environ, {'LLM_RETRIES': '2'}).start()
        patch('prompt_opt_harness.llm_backends.time.sleep').start()

    def test_split_is_disjoint_stratified_and_order_independent(self):
        a, b = split_source(rows(), 37, SENTIMENT)
        c, d = split_source(list(reversed(rows())), 37, SENTIMENT)
        self.assertEqual((a, b), (c, d))
        self.assertFalse({e.id for e in a} & {e.id for e in b})
        self.assertEqual({e.label for e in a}, set(SENTIMENT.labels))
        self.assertNotEqual((a, b), split_source(rows(), 38, SENTIMENT))

    def test_rejects_duplicates_labels_and_small_splits(self):
        for bad in [rows() + [rows()[0]], rows() + [Example('other', rows()[0].text.upper(), 'Positive', 's')],
                    rows() + [Example('other', 'new text', 'Neutral', 's')], rows()[:2]]:
            with self.subTest(bad=bad[-1]), self.assertRaises(ValueError):
                split_source(bad, 0, SENTIMENT)

    def test_rejects_target_argument_before_import_or_requests(self):
        backend = Backend()
        with patch('prompt_opt_harness.optimizers.dspy_adapters._load_dspy') as load:
            with self.assertRaises(ValueError):
                DSPyGEPA(backend, 20).optimize(BASE_INSTRUCTION, rows(), ood_examples={})
            load.assert_not_called()
        self.assertEqual(backend.stats.calls, 0)

    def test_initial_score_must_fit_before_requests(self):
        backend = Backend()
        with self.assertRaises(CallBudgetExceeded):
            DSPyGEPA(backend, 1).optimize(BASE_INSTRUCTION, rows())
        self.assertEqual(backend.requests, [])

    def test_calls_settings_seed_and_exact_final_predictions(self):
        backend = Backend()
        optimizer = ControlledMIPRO(backend, 30, seed=37, proposal_max_tokens=700)
        best = optimizer.optimize(BASE_INSTRUCTION, rows())
        self.assertEqual(best.dev_score, 1.0)
        self.assertEqual(backend.stats.calls, len(backend.requests))
        self.assertEqual(backend.stats.calls, 17)  # baseline4 + proposal1 + compile8 + validation4
        self.assertFalse({e.harness_text for e in optimizer.seen_train} &
                         {e.harness_text for e in optimizer.seen_val})
        proposal = [r for r in backend.requests if r[0] == 'proposal']
        self.assertEqual(proposal[0][1:], (0.7, 700))
        for prompt, temperature, cap in backend.requests:
            if '\n\nText:' in prompt:
                self.assertEqual((temperature, cap), (0.0, 16))
        self.assertEqual(evaluate(backend, best.instruction, rows()), 1.0)
        expected = build_classification_prompt(best.instruction, rows()[0].text)
        self.assertGreaterEqual(sum(p == expected for p, _, _ in backend.requests), 2)
        self.assertEqual((backend.temperature, backend.max_tokens), (0.0, 16))
        self.assertIsNone(backend._call_limit)

    def test_completed_validation_can_be_recovered_after_compile_budget_exhaustion(self):
        class RunsPastBudget(ControlledMIPRO):
            def _compile(self, dspy, student, metric, lm, *args):
                super()._compile(dspy, student, metric, lm, *args)
                for _ in range(5):
                    lm(prompt='another proposal')
                return student
        backend = Backend()
        optimizer = RunsPastBudget(backend, 18)
        best = optimizer.optimize(BASE_INSTRUCTION, rows())
        self.assertEqual(best.dev_score, 1.)
        self.assertEqual(backend.stats.calls, 18)
        self.assertEqual(optimizer.metadata['stop_reason'], 'compile_budget_exhausted_validated_candidate')

    def test_budget_exhaustion_keeps_base_and_never_ranks_partial_candidate(self):
        backend = Backend()
        optimizer = ControlledMIPRO(backend, 10)
        best = optimizer.optimize(BASE_INSTRUCTION, rows())
        self.assertEqual(best.instruction, BASE_INSTRUCTION)
        self.assertEqual(len(optimizer.history), 1)
        self.assertLessEqual(backend.stats.calls, 10)
        self.assertEqual(optimizer.metadata['stop_reason'], 'compile_budget_exhausted_kept_base')
        self.assertIs(optimizer.backend, backend)

    def test_final_validation_retry_exhaustion_keeps_base(self):
        class FailsOnValidation(ControlledMIPRO):
            def _compile(inner, *args):
                result = super()._compile(*args)
                inner.backend._backend.failures = [TimeoutError()] * 2
                return result
        backend = Backend()
        optimizer = FailsOnValidation(backend, 17)
        best = optimizer.optimize(BASE_INSTRUCTION, rows())
        self.assertEqual(best.instruction, BASE_INSTRUCTION)
        self.assertEqual(len(optimizer.history), 1)
        self.assertEqual(backend.stats.calls, 17)
        self.assertEqual(backend.stats.failed_attempts, 2)

    def test_ties_keep_base(self):
        class TieBackend(Backend):
            def _generate(self, prompt):
                super()._generate(prompt)
                return 'Negative'
        class TieOptimizer(ControlledMIPRO):
            def _compile(self, dspy, student, *args):
                student.prog.signature = student.prog.signature.with_instructions('Different')
                return student
        best = TieOptimizer(TieBackend(), 20).optimize(BASE_INSTRUCTION, rows())
        self.assertEqual(best.instruction, BASE_INSTRUCTION)

    def test_compiled_demos_are_rejected(self):
        class WithDemos(ControlledMIPRO):
            def _compile(self, dspy, student, *args):
                student.prog.demos = [dspy.Example(harness_text='leaked demo', harness_label='Positive')]
                return student
        with self.assertRaisesRegex(ValueError, 'instruction-only'):
            WithDemos(Backend(), 20).optimize(BASE_INSTRUCTION, rows())

    def test_adapter_rejects_unsupplied_text_and_demos(self):
        state = SimpleNamespace(lock=threading.RLock(), instructions={}, budget_exhausted=False)
        backend = Backend()
        adapter = _make_task_adapter(dspy, BudgetedBackend(backend, 3), SENTIMENT, {'known'}, state)
        sig = dspy.Signature('harness_text -> harness_label', instructions=BASE_INSTRUCTION)
        for demos, text in [([], 'held-out target'), ([{'harness_text': 'known'}], 'known')]:
            with self.assertRaises(ValueError):
                adapter(None, {}, sig, demos, {'harness_text': text})
        self.assertEqual(backend.stats.calls, 0)

    def test_bridge_copies_share_retry_budget_and_reserve_n(self):
        backend = Backend()
        backend.failures = [TimeoutError()]
        state = SimpleNamespace(lock=threading.RLock(), budget_exhausted=False)
        lm = _make_bridge(dspy, BudgetedBackend(backend, 1), temperature=.7, max_tokens=100, state=state)
        for child in [lm.copy(temperature=.8), copy.deepcopy(lm)]:
            with self.assertRaises(CallBudgetExceeded):
                child(prompt='proposal')
        self.assertEqual(backend.stats.calls, 1)
        self.assertEqual(backend.stats.failed_attempts, 1)
        self.assertIsNone(backend._call_limit)
        backend2 = Backend()
        lm2 = _make_bridge(dspy, BudgetedBackend(backend2, 1), temperature=.7, max_tokens=100, state=state)
        with self.assertRaises(CallBudgetExceeded):
            lm2(prompt='proposal', n=2)
        self.assertEqual(backend2.stats.calls, 0)

    def test_seed_and_instruction_only_options_reach_real_optimizers(self):
        lm = dspy.utils.DummyLM(['unused'])
        for cls in [DSPyGEPA, DSPyMIPROv2]:
            opt = cls(Backend(), 30, seed=37)
            with dspy.context(lm=lm):
                real = opt._teleprompter(dspy, lambda a,b,c=None,d=None,e=None: 0., lm, 20)
            self.assertEqual(real.seed, 37)
            if cls is DSPyMIPROv2:
                self.assertEqual((real.max_bootstrapped_demos, real.max_labeled_demos), (0, 0))
        self.assertEqual(DSPyCOPRO.name, 'DSPy-COPRO')

    def test_backend_passes_cap_to_provider_and_restores_config(self):
        fake = SimpleNamespace(OpenAI=Mock())
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Positive'))])
        fake.OpenAI.return_value.chat.completions.create.return_value = response
        with patch.dict(sys.modules, {'openai': fake}):
            backend = OpenAICompatibleBackend('test', 'http://unused', temperature=0., max_tokens=16)
        backend.generate_configured('proposal', temperature=.7, max_tokens=700)
        backend.generate('inference')
        calls = backend.client.chat.completions.create.call_args_list
        self.assertEqual([(c.kwargs['temperature'], c.kwargs['max_tokens']) for c in calls], [(.7, 700), (0., 16)])
        self.assertEqual(backend.stats.calls, 2)
        self.assertEqual([t['generation_settings']['max_tokens'] for t in backend.stats.per_call], [700, 16])

    def test_context_restored_and_nonbudget_errors_remain_visible(self):
        class Broken(ControlledMIPRO):
            def _compile(self, *args):
                raise ValueError('Broken compiler')
        original = (dspy.settings.lm, dspy.settings.adapter)
        for cls in [ControlledMIPRO, Broken]:
            backend = Backend()
            optimizer = cls(backend, 30)
            if cls is Broken:
                with self.assertRaisesRegex(ValueError, 'Broken compiler'):
                    optimizer.optimize(BASE_INSTRUCTION, rows())
            else:
                optimizer.optimize(BASE_INSTRUCTION, rows())
            self.assertEqual((dspy.settings.lm, dspy.settings.adapter), original)
            self.assertIs(optimizer.backend, backend)
            self.assertIsNone(backend._call_limit)

    def test_raw_invalid_labels_get_the_same_zero_score_in_both_paths(self):
        class InvalidBackend(Backend):
            def _generate(self, prompt):
                super()._generate(prompt)
                return 'Positive or Negative'
        class ChecksMetric(ControlledMIPRO):
            def _compile(self, dspy, student, metric, lm, trainset, valset, remaining):
                for ex in trainset + valset:
                    self.assert_score = metric(ex, student(**ex.inputs()))
                    assert self.assert_score == 0.
                return student
        backend = InvalidBackend()
        best = ChecksMetric(backend, 30).optimize(BASE_INSTRUCTION, rows())
        self.assertEqual(best.dev_score, 0.)
        self.assertEqual(evaluate(backend, best.instruction, rows()), 0.)

    def test_harness_logs_search_and_evaluation_separately(self):
        source, test, target = rows(), rows('test'), rows('target')
        dataset = Dataset('source', DomainSplit(source, test), {'target': target})
        result = run_experiment('DSPy-test', ControlledMIPRO, Backend, dataset, budget=30, seed=37)
        self.assertEqual((result.search_api_calls, result.evaluation_api_calls), (17, 16))
        self.assertEqual(result.api_calls, 33)
        self.assertEqual(result.optimizer_metadata['seed'], 37)
        self.assertFalse(set(result.optimizer_metadata['source_train_ids']) & {e.id for e in target + test})
        json.dumps(result.to_dict())

    def test_binary_nli_uses_explicit_schema(self):
        backend = Backend()
        optimizer = ControlledMIPRO(backend, 30, task=BINARY_NLI)
        best = optimizer.optimize('Classify entailment.', rows(task=BINARY_NLI))
        self.assertEqual(best.dev_score, 1.)
        self.assertIn(BINARY_NLI.answer_format, backend.requests[-1][0])

    def test_real_copro_compiles_offline(self):
        backend = Backend()
        optimizer = DSPyCOPRO(backend, 200, seed=37)
        best = optimizer.optimize(BASE_INSTRUCTION, rows())
        self.assertEqual(optimizer.metadata['stop_reason'], 'completed')
        self.assertEqual(best.dev_score, 1.)
        self.assertGreater(len(optimizer.metadata['evaluated_instructions']), 1)
        self.assertLessEqual(backend.stats.calls, 200)

    def test_real_mipro_compiles_offline(self):
        backend = Backend()
        optimizer = DSPyMIPROv2(backend, 500, seed=37)
        best = optimizer.optimize(BASE_INSTRUCTION, rows())
        self.assertEqual(optimizer.metadata['stop_reason'], 'completed')
        self.assertEqual(best.dev_score, 1.)
        self.assertLessEqual(backend.stats.calls, 500)

    def test_real_gepa_compiles_offline(self):
        class SmallGEPA(DSPyGEPA):
            def _teleprompter(self, dspy, metric, lm, remaining):
                return super()._teleprompter(dspy, metric, lm, min(remaining, 24))
        backend = Backend()
        optimizer = SmallGEPA(backend, 200, seed=37)
        best = optimizer.optimize(BASE_INSTRUCTION, rows())
        self.assertEqual(optimizer.metadata['stop_reason'], 'completed')
        self.assertEqual(best.dev_score, 1.)
        self.assertLessEqual(backend.stats.calls, 200)


if __name__ == '__main__':
    unittest.main()
