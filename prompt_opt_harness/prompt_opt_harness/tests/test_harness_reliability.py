"""Offline regressions for exact scoring, attempt accounting and OPRO budgets."""
import builtins
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompt_opt_harness.budget import BudgetedBackend, CallBudgetExceeded
from prompt_opt_harness.tasks import BINARY_NLI
from prompt_opt_harness.datasets import Dataset, DomainSplit, Example
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import LLMBackend, OpenAIBackend, OpenAICompatibleBackend, AnthropicBackend
from prompt_opt_harness.optimizers.base import evaluate, evaluate_per_example, parse_label
from prompt_opt_harness.optimizers import (OPRO, create_dro_optimizer,
    create_regularized_optimizer, create_sapo_optimizer, create_entropy_regularized_optimizer)


def examples(n=4):
    return [Example(str(i), f'Source text {i}', 'Positive', 'source') for i in range(n)]


class Backend(LLMBackend):
    def __init__(self, answers=None):
        super().__init__()
        self.answers = iter(answers) if answers is not None else None
        self.requests = []

    def _generate(self, prompt):
        self.requests.append(prompt)
        answer = next(self.answers) if self.answers is not None else (
            'Positive' if '\n\nText:' in prompt else 'Revised instruction')
        if isinstance(answer, Exception):
            raise answer
        return answer


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {'tiktoken': None}).start()
        patch.dict(os.environ, {'LLM_RETRIES': '2'}).start()
        patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')).start()
        patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')).start()

    def test_only_exact_inputs_share_a_prediction_without_embedding_imports(self):
        rows = [Example('p', 'This is good', 'Positive', 'source'),
                Example('n', 'This is not good', 'Negative', 'source'),
                Example('p2', 'This is good', 'Positive', 'source')]
        real_import = builtins.__import__
        embedding_imports = []
        def import_guard(name, *args, **kwargs):
            if name.startswith('sentence_transformers'):
                embedding_imports.append(name)
                self.fail('Evaluation attempted to load an embedding model')
            return real_import(name, *args, **kwargs)
        for detailed in (False, True):
            backend = Backend(['Positive', 'Negative'])
            with patch('builtins.__import__', side_effect=import_guard):
                if detailed:
                    self.assertTrue(all(ok for _, _, ok in evaluate_per_example(backend, 'Sentiment', rows)))
                else:
                    self.assertEqual(evaluate(backend, 'Sentiment', rows), 1.)
            self.assertEqual(len(backend.requests), 2)
        self.assertEqual(embedding_imports, [])

    def test_cache_is_scoped_to_a_batch_and_preserves_exact_text(self):
        backend = Backend()
        rows = examples(1) + [Example('space', 'Source text 0 ', 'Positive', 'source')]
        evaluate(backend, 'Sentiment', rows)
        evaluate(backend, 'Different instruction', rows)
        evaluate(backend, 'Sentiment', rows)
        self.assertEqual(len(backend.requests), 6)

    def test_complete_labels_only_including_nli(self):
        for raw, expected in [(' Positive ', 'positive'), ('NEGATIVE!', 'negative'),
                              ('"Positive".', 'positive'), ("'Contradiction'", 'contradiction'),
                              ('Entailment', 'entailment')]:
            task = BINARY_NLI if expected in ('entailment', 'contradiction') else None
            self.assertEqual(parse_label(raw, task) if task else parse_label(raw), expected)
        for raw in ['Positive or Negative', 'positively awful', 'Pos', 'neg',
                    'Positive because it is good', 'Entails', 'Not Positive', '',
                    'Positive\nNegative', 'Positive..', '"Positive', 'Neutral']:
            with self.subTest(raw=raw):
                self.assertEqual(parse_label(raw), 'INVALID')
                self.assertEqual(evaluate(Backend([raw]), 'Sentiment', examples(1)), 0.)
        row = Example('nli', 'Premise and hypothesis', 'Contradiction', 'nli')
        self.assertEqual(evaluate(Backend(['Contradiction']), 'Entailment task', [row], BINARY_NLI), 1.)

    def test_aggregate_and_detailed_scores_agree_with_invalids(self):
        rows = examples(3)
        answers = ['Positive', 'Positive or Negative', 'Negative']
        detail = evaluate_per_example(Backend(answers), 'Sentiment', rows)
        self.assertEqual([ok for _, _, ok in detail], [True, False, False])
        self.assertEqual(evaluate(Backend(answers), 'Sentiment', rows), 1/3)
        self.assertEqual(evaluate(Backend(), 'Sentiment', []), 0.)

    def test_slow_success_is_never_retried(self):
        backend = Backend(['Positive'])
        with patch('prompt_opt_harness.llm_backends.time.perf_counter', side_effect=[0, 31]), \
             patch('prompt_opt_harness.llm_backends.time.sleep') as sleep:
            self.assertEqual(backend.generate('test'), 'Positive')
        sleep.assert_not_called()
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(backend.stats.calls, 1)
        self.assertEqual(backend.stats.wall_time, 31)
        self.assertEqual(backend.stats.failed_attempts, 0)

    def test_retry_records_every_attempt_and_backoff(self):
        backend = Backend([TimeoutError(), 'Positive'])
        with patch('prompt_opt_harness.llm_backends.time.perf_counter', side_effect=[0, 2, 2, 2.5, 2.5, 5.5]), \
             patch('prompt_opt_harness.llm_backends.time.sleep'):
            self.assertEqual(backend.generate('test'), 'Positive')
        self.assertEqual(backend.stats.calls, 2)
        self.assertEqual(backend.stats.failed_attempts, 1)
        self.assertEqual(backend.stats.wall_time, 5)
        self.assertEqual(backend.stats.backoff_time, .5)
        self.assertEqual(backend.stats.unknown_usage_attempts, 1)
        self.assertEqual([r['status'] for r in backend.stats.per_call], ['error', 'success'])
        self.assertIsNone(backend.stats.per_call[0]['input_tokens'])
        self.assertEqual(backend.stats.per_call[1]['token_usage'], 'estimated')

    def test_exhausted_retries_are_still_counted(self):
        backend = Backend([TimeoutError(), TimeoutError(), TimeoutError()])
        with patch('prompt_opt_harness.llm_backends.time.sleep'), self.assertRaises(TimeoutError):
            backend.generate('test')
        self.assertEqual((len(backend.requests), backend.stats.calls, backend.stats.failed_attempts), (3, 3, 3))

    def test_permanent_http_error_is_not_retried(self):
        error = RuntimeError('Invalid request')
        error.status_code = 400
        backend = Backend([error])
        with self.assertRaises(RuntimeError):
            backend.generate('test')
        self.assertEqual(backend.stats.calls, 1)
        self.assertEqual(backend.stats.failed_attempts, 1)

    def test_sdk_internal_retries_are_disabled(self):
        openai, anthropic = SimpleNamespace(OpenAI=Mock()), SimpleNamespace(Anthropic=Mock())
        with patch.dict(sys.modules, {'openai': openai, 'anthropic': anthropic}), \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'test-only', 'ANTHROPIC_API_KEY': 'test-only'}):
            OpenAIBackend()
            OpenAICompatibleBackend('test', 'http://localhost')
            AnthropicBackend()
        self.assertTrue(all(call.kwargs['max_retries'] == 0 for call in openai.OpenAI.call_args_list))
        self.assertEqual(anthropic.Anthropic.call_args.kwargs['max_retries'], 0)

    def test_retry_cannot_exceed_budget_and_scope_is_restored(self):
        backend = Backend([TimeoutError(), 'Positive'])
        limited = BudgetedBackend(backend, 1)
        with self.assertRaises(CallBudgetExceeded), patch('prompt_opt_harness.llm_backends.time.sleep') as sleep:
            limited.generate('test')
        sleep.assert_not_called()
        self.assertEqual(len(backend.requests), 1)
        self.assertIsNone(backend._call_limit)
        self.assertEqual(backend.generate('final evaluation outside search'), 'Positive')

    def test_initial_batch_must_fit_before_any_request(self):
        for budget in (0, 1, 3):
            backend = Backend()
            opt = OPRO(backend, budget)
            with self.assertRaises(CallBudgetExceeded):
                opt.optimize('Base instruction', examples(4))
            self.assertEqual(backend.requests, [])
            self.assertEqual(opt.history, [])
            self.assertIs(opt.backend, backend)

    def test_invalid_budget_and_empty_dev_make_no_requests(self):
        for budget in (-1, 1.5, True):
            backend = Backend()
            with self.assertRaises(ValueError):
                OPRO(backend, budget).optimize('Base', examples(1))
            self.assertEqual(backend.requests, [])
        with self.assertRaises(ValueError):
            OPRO(Backend(), 1).optimize('Base', [])

    def test_small_remainders_do_not_generate_unscorable_proposals(self):
        for budget in range(4, 9):
            backend = Backend()
            opt = OPRO(backend, budget)
            self.assertEqual(opt.optimize('Base instruction', examples(4)).instruction, 'Base instruction')
            self.assertEqual(len(backend.requests), 4)
            self.assertEqual(len(opt.history), 1)

    def test_batch_boundary_and_duplicate_proposals_stay_within_cap(self):
        for budget in range(9, 26):
            backend = Backend()
            opt = OPRO(backend, budget)
            opt.optimize('Base instruction', examples(4))
            self.assertLessEqual(backend.stats.calls, budget)
            self.assertTrue(all(c.dev_score == 1. for c in opt.history))
            # Every generated proposal has exactly four classification requests.
            proposal_count = sum('\n\nText:' not in p for p in backend.requests)
            self.assertEqual(len(backend.requests), 4 + 5 * proposal_count)

    def test_duplicate_examples_use_exact_request_reservation(self):
        backend = Backend()
        opt = OPRO(backend, 1)
        self.assertEqual(opt.optimize('Base instruction', examples(1) * 4).dev_score, 1.)
        self.assertEqual(backend.stats.calls, 1)

    def test_retries_do_not_admit_partial_candidate_scores(self):
        backend = Backend(['Positive', 'Positive', 'Revised instruction', TimeoutError(), 'Positive'])
        opt = OPRO(backend, 5)
        with patch('prompt_opt_harness.llm_backends.time.sleep'):
            selected = opt.optimize('Base instruction', examples(2))
        self.assertEqual(backend.stats.calls, 5)
        self.assertEqual(selected.instruction, 'Base instruction')
        self.assertEqual(len(opt.history), 1)
        self.assertIs(opt.backend, backend)
        self.assertIsNone(backend._call_limit)

    def test_incomplete_initial_score_fails_explicitly(self):
        backend = Backend([TimeoutError(), 'Positive'])
        opt = OPRO(backend, 2)
        with patch('prompt_opt_harness.llm_backends.time.sleep'), self.assertRaises(CallBudgetExceeded):
            opt.optimize('Base instruction', examples(2))
        self.assertEqual(opt.history, [])
        self.assertEqual(backend.stats.calls, 2)

    def test_each_search_has_its_own_budget(self):
        backend = Backend()
        backend.generate('Previous call')
        opt = OPRO(backend, 2)
        for expected in (3, 5):
            opt.optimize('Base instruction', examples(2))
            self.assertEqual(backend.stats.calls, expected)
            self.assertEqual(len(opt.history), 1)

    def test_wrapped_opro_reserves_extra_sampling_calls(self):
        for cls, cost in [(create_sapo_optimizer(OPRO), 8),
                          (create_entropy_regularized_optimizer(OPRO), 10),
                          (create_dro_optimizer(OPRO), 2),
                          (create_regularized_optimizer(OPRO), 2)]:
            backend = Backend()
            with self.subTest(cls=cls.name), self.assertRaises(CallBudgetExceeded):
                cls(backend, cost-1).optimize('Base instruction', examples(2))
            self.assertEqual(backend.requests, [])
            cls(backend, cost).optimize('Base instruction', examples(2))
            self.assertEqual(backend.stats.calls, cost)

    def test_entropy_wrapper_also_rejects_malformed_labels(self):
        backend = Backend(['Positive or Negative'] * 5)
        opt = create_entropy_regularized_optimizer(OPRO)(backend, 5)
        self.assertEqual(opt.optimize('Base instruction', examples(1)).dev_score, 0.)

    def test_run_result_separates_search_and_final_attempts(self):
        data = Dataset('source', DomainSplit(examples(2),
            [Example('test', 'Source heldout', 'Positive', 'source')]),
            {'target': [Example('target', 'Target heldout', 'Positive', 'target')]})
        for separate in (False, True):
            backend = Backend()
            final = Backend([TimeoutError(), 'Positive', 'Positive']) if separate else backend
            backend.generate('Call before this run')
            with patch('prompt_opt_harness.llm_backends.time.sleep'):
                result = run_experiment('OPRO', OPRO, lambda: backend, data, budget=2,
                    eval_backend_factory=(lambda: final) if separate else None)
            record = result.to_dict()
            self.assertEqual(record['search_api_calls'], 2)
            self.assertEqual(record['evaluation_api_calls'], 3 if separate else 2)
            self.assertEqual(record['api_calls'], 5 if separate else 4)
            self.assertEqual(record['failed_attempts'], 1 if separate else 0)
            self.assertEqual(record['unknown_usage_attempts'], record['failed_attempts'])
            self.assertEqual(len(record['per_call_traces']), record['api_calls'])
            self.assertEqual(record['scoring_protocol'], 'exact-label-v2-explicit-task')
            self.assertEqual(result.source_test_acc, 1.)


if __name__ == '__main__':
    unittest.main()
