"""Protocol mismatches must fail before the model client is constructed."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core import fingerprint, save_json
from protocol import build_protocol, lock_protocol, validate_execution_context, validate_splits_manifest
from data import prepare
import run


def args():
    return SimpleNamespace(model='gemma4:31b-cloud', endpoint='http://localhost:11434/v1',
        seeds=[0], regimes=['single'], methods=['opro'], budget=512, workers=4,
        train_n=2, val_n=2, test_n=2)


def splits():
    rows = {}
    for domain, prefix in (('sst2', 'sst'), ('amazon', 'amazon'), ('tweets', 'tweets')):
        for split in (('test',) if domain == 'tweets' else ('train', 'val', 'test')):
            key = f'{prefix}_{split}'
            rows[key] = [dict(id=f'{key}:{i}', text=f'{key} text {i}', label=label,
                             domain=domain, split=split) for i, label in enumerate(('Positive', 'Negative'))]
    return dict(config=dict(train_n=2, val_n=2, test_n=2, split_seed=2712026),
                splits=rows, split_hash=fingerprint(rows))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.split = splits()
        save_json(self.root/'splits.json', self.split)
        self.config = build_protocol(args(), self.split)
        self.addCleanup(patch.stopall)
        patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')).start()

    def test_matching_resume_does_not_rewrite_protocol(self):
        pid = lock_protocol(self.root, self.config, 'search')
        before = (self.root/'protocol.json').read_bytes()
        save_json(self.root/'single_opro_0/search.json', dict(protocol_hash=pid))
        self.assertEqual(lock_protocol(self.root, self.config, 'search'), pid)
        self.assertEqual(before, (self.root/'protocol.json').read_bytes())
        self.assertEqual(validate_execution_context(self.root), self.config)

    def test_any_locked_setting_change_is_rejected(self):
        lock_protocol(self.root, self.config, 'search')
        before = (self.root/'protocol.json').read_bytes()
        replacements = dict(model='different', endpoint='http://other', evaluation_cap=1024,
            proposal_cap=4, workers=1, seeds=[1], regimes=['multi'], methods=['gepa'],
            decoding=dict(temperature=0, seed=12345, max_tokens=1),
            proposal_decoding=dict(temperature=.5), split_hash='changed',
            split_config={}, source_file_hashes={}, task=dict(name='nli'), python='other')
        for key, value in replacements.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'configuration changed'):
                lock_protocol(self.root, dict(self.config, **{key:value}), 'evaluate')
        self.assertEqual(before, (self.root/'protocol.json').read_bytes())

    def test_changed_data_rejected_even_if_its_local_hash_is_updated(self):
        lock_protocol(self.root, self.config, 'search')
        self.split['splits']['sst_train'][0]['text'] = 'changed text'
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            validate_splits_manifest(self.split)
        self.split['split_hash'] = fingerprint(self.split['splits'])
        save_json(self.root/'splits.json', self.split)
        with self.assertRaisesRegex(ValueError, 'split_hash'):
            validate_execution_context(self.root)

    def test_prepare_validates_saved_data_before_returning_it(self):
        self.split['splits']['sst_train'][0]['label'] = 'Negative'
        save_json(self.root/'splits.json', self.split)
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            prepare(self.root, 2, 2, 2)

    def test_overlapping_ids_or_texts_are_rejected_even_with_matching_hash(self):
        for field in ('id', 'text'):
            split = copy.deepcopy(self.split)
            split['splits']['sst_test'][0][field] = split['splits']['sst_train'][0][field]
            split['split_hash'] = fingerprint(split['splits'])
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                validate_splits_manifest(split)

    def test_foreign_or_legacy_artifact_cannot_be_skipped_on_resume(self):
        lock_protocol(self.root, self.config, 'search')
        for relative in ('single_opro_0/search.json', 'final_predictions/prompt.json'):
            path = self.root/relative
            for content in ({}, dict(protocol_hash='other run')):
                save_json(path, content)
                with self.assertRaisesRegex(ValueError, 'Artifact does not match'):
                    lock_protocol(self.root, self.config, 'search')
            path.unlink()

    def test_legacy_manifest_is_not_silently_upgraded(self):
        legacy = {k:v for k,v in self.config.items() if k != 'protocol_version'}
        save_json(self.root/'protocol.json', legacy)
        before = (self.root/'protocol.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'Legacy experiment'):
            lock_protocol(self.root, self.config, 'search')
        self.assertEqual(before, (self.root/'protocol.json').read_bytes())

    def test_missing_manifest_with_results_or_evaluation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing protocol'):
            lock_protocol(self.root, self.config, 'evaluate')
        (self.root/'requests.sqlite').write_bytes(b'cache placeholder')
        with self.assertRaisesRegex(ValueError, 'Missing protocol'):
            lock_protocol(self.root, self.config, 'search')
        self.assertFalse((self.root/'protocol.json').exists())

    def test_cli_mismatch_is_rejected_before_client_creation(self):
        lock_protocol(self.root, self.config, 'search')
        for stage in ('prepare', 'search', 'evaluate', 'latency'):
            command = ['run.py', '--output', str(self.root), '--stage', stage,
                       '--train-n', '2', '--val-n', '2', '--test-n', '2',
                       '--methods', 'opro', '--regimes', 'single', '--seeds', '0', '--model', 'changed']
            with patch.object(sys, 'argv', command), patch.object(run, 'Client') as client:
                with self.assertRaisesRegex(ValueError, 'model'):
                    run.main()
                client.assert_not_called()

    def test_cli_final_artifact_mismatch_is_rejected_before_client(self):
        lock_protocol(self.root, self.config, 'search')
        save_json(self.root/'final_predictions/prompt.json', dict(protocol_hash='foreign'))
        command = ['run.py', '--output', str(self.root), '--stage', 'evaluate',
                   '--train-n', '2', '--val-n', '2', '--test-n', '2',
                   '--methods', 'opro', '--regimes', 'single', '--seeds', '0']
        with patch.object(sys, 'argv', command), patch.object(run, 'Client') as client:
            with self.assertRaisesRegex(ValueError, 'Artifact does not match'):
                run.main()
            client.assert_not_called()

    def test_search_final_and_resume_keep_matching_provenance_without_requests(self):
        instances = []
        class FakeClient:
            def __init__(self, cache_path, model, workers, **kwargs):
                self.workers = workers
                self.stats = dict(actual_calls=0, cache_hits=0, failed_attempts=0,
                                  input_tokens=0, output_tokens=0, request_seconds=0.)
                instances.append(self)
            def complete(self, prompt, **kwargs):
                self.stats['actual_calls'] += 1
                return dict(text='Positive', usage=dict(prompt_tokens=10, completion_tokens=1))
            def snapshot(self):
                return dict(self.stats)
        command = ['run.py', '--output', str(self.root), '--train-n', '2', '--val-n', '2',
                   '--test-n', '2', '--methods', 'opro', '--regimes', 'single', '--seeds', '0', '--budget', '2']
        for stage in ('search', 'evaluate', 'search', 'evaluate'):
            with patch.object(sys, 'argv', command+['--stage', stage]), patch.object(run, 'Client', FakeClient):
                run.main()
        self.assertEqual([c.stats['actual_calls'] for c in instances], [2, 12, 0, 0])
        protocol = json.loads((self.root/'protocol.json').read_text(encoding='utf-8'))
        paths = [self.root/'single_opro_0/search.json'] + list((self.root/'final_predictions').glob('*.json'))
        self.assertEqual(len(paths), 3)
        for path in paths:
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['protocol_hash'], fingerprint(protocol))


if __name__ == '__main__':
    unittest.main()
