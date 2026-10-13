"""Offline regression checks for the fresh pilot; never contacts the model."""
import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from adaptive_search import VARIANTS, balanced_order, confirmation_gate, run_search
from core import BASE, BudgetExceeded, Client, Evaluator, Example, fingerprint, save_json
from fresh_data import SIZES, collect_exclusions, validate_fresh
from improvement_pilot import (confirm, evaluate, lock, metrics, paired_interval, report,
                               searches, validate_predictions)


def split(name, n, domain='sst2'):
    return [Example(f'{name}:{i}', f'{name} {i} '+('good' if i%2 else 'bad'),
                    'Positive' if i%2 else 'Negative', domain, name) for i in range(n)]


def manifest():
    raw = {name:[asdict(e) for e in split(name,n,domain)] for name,n,domain in
           [('train',64,'sst2'),('validation',64,'sst2'),('confirmation',80,'sst2'),
            ('source_test',64,'sst2'),('amazon_test',160,'amazon'),('tweets_test',160,'tweets')]}
    return dict(sizes=SIZES, splits=raw, split_hash=fingerprint(raw), excluded_texts=[],
                exclusion_hash=fingerprint([]))


class FakeClient:
    workers = 4
    def __init__(self):
        self.prompts = []; self.proposals = 0
    def snapshot(self): return dict(actual_calls=len(self.prompts))
    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if kwargs.get('temperature') == .7:
            self.proposals += 1
            text = f'```\nRule {self.proposals}: classify overall sentiment as Positive or Negative.\n```'
        else:
            body = prompt.split('\n\nText: ')[1].split('\nAnswer')[0]
            text = 'Positive' if body.endswith('good') else 'Negative'
            if prompt.startswith(BASE) and ' 1 ' in body:
                text = 'Negative'
        return dict(text=text, usage=dict(prompt_tokens=12, completion_tokens=2), finish_reason='stop')


class ImprovementTests(unittest.TestCase):
    def test_racing_budget_and_common_subsets(self):
        train, val = split('train',64), split('validation',64)
        ev = Evaluator(FakeClient(), train+val, budget=512)
        result = run_search(ev,train,val,0,'feedback_racing')
        self.assertEqual(len(result['candidates']),13)
        self.assertEqual(len(result['finalists']),3)
        self.assertEqual(ev.eval_count,512)
        self.assertEqual(ev.proposal_count,12)
        stage16 = [a['example_ids'] for a in ev.audit if len(a['example_ids']) == 16]
        self.assertTrue(all(ids == stage16[0] for ids in stage16))
        self.assertEqual(result['stages'][0]['ids'],result['stages'][1]['ids'][:16])

    def test_feedback_is_only_difference_in_full_variants(self):
        train, val = split('train',64), split('validation',64)
        results=[]; clients=[]
        for mode in VARIANTS[:2]:
            client=FakeClient(); ev=Evaluator(client,train+val,budget=512)
            results.append(run_search(ev,train,val,0,mode)); clients.append(client)
            self.assertEqual(ev.eval_count,512)
            self.assertEqual(ev.proposal_count,6)
        self.assertEqual(results[0]['candidates'],results[1]['candidates'])
        plain=[p for p in clients[0].prompts if p.startswith('Improve an instruction')]
        feedback=[p for p in clients[1].prompts if p.startswith('Improve an instruction')]
        for a,b in zip(plain,feedback):
            self.assertEqual(a,b.split('\nSource TRAINING examples')[0])
            self.assertNotIn('validation 1 ',b)
            self.assertIn('"expected"',b)

    def test_target_or_confirmation_cannot_be_passed_as_training(self):
        train, val=split('train',64),split('validation',64)
        for name in ('confirmation','source_test','tweets_test'):
            bad=[replace(e,split=name) for e in train]
            client=FakeClient();ev=Evaluator(client,bad+val,budget=512)
            with self.assertRaises(ValueError):run_search(ev,bad,val,0,'feedback_full')
            self.assertEqual(client.prompts,[])

    def test_confirmation_not_in_search_allowlist(self):
        train,val=split('train',64),split('validation',64)
        ev=Evaluator(FakeClient(),train+val,budget=512)
        with self.assertRaises(ValueError):ev.evaluate(BASE,split('confirmation',80))
        self.assertEqual(ev.eval_count,0)

    def test_nested_order_independent_of_input_order(self):
        train=split('train',64)
        self.assertEqual(balanced_order(train,7),balanced_order(list(reversed(train)),7))
        for n in (16,32,64):
            self.assertEqual(sum(e.label=='Positive' for e in balanced_order(train,7)[:n]),n//2)

    def test_small_confirmation_gain_retains_base(self):
        base=[0,0]+[1]*78
        result=confirmation_gate(base,[1]*80,3)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['p_value'],.25)

    def test_strong_confirmation_gain_accepted(self):
        result=confirmation_gate([0]*8+[1]*72,[1]*80,3)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['corrected'],8)
        self.assertEqual(result['regressed'],0)

    def test_guard_ties_and_regressions(self):
        self.assertFalse(confirmation_gate([1]*80,[1]*80,3)['accepted'])
        self.assertFalse(confirmation_gate([1]*80,[0]*80,3)['accepted'])

    def test_protocol_mismatch_and_foreign_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);config=dict(model='gemma',cap=4000)
            pid=lock(out,config)
            self.assertEqual(lock(out,config),pid)
            with self.assertRaises(ValueError):lock(out,dict(config,cap=8000))
            save_json(out/'searches/a.json',dict(protocol_hash='wrong'))
            with self.assertRaises(ValueError):lock(out,config)

    def test_exclusions_recover_manifests_and_old_prompts(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'old.json'
            save_json(path,[dict(text=' A  review '),dict(prompt='Instruction\n\nText: ANOTHER review\nAnswer with exactly one word: Positive or Negative.')])
            texts,_=collect_exclusions([path])
            self.assertEqual(texts,{'a review','another review'})

    def test_fresh_splits_reject_previous_exposure_and_overlap(self):
        data=manifest();validate_fresh(data)
        data['excluded_texts']=['train 0 bad'];data['exclusion_hash']=fingerprint(data['excluded_texts'])
        with self.assertRaises(ValueError):validate_fresh(data)
        data=manifest();data['splits']['confirmation'][0]['text']=data['splits']['train'][0]['text']
        data['split_hash']=fingerprint(data['splits'])
        with self.assertRaises(ValueError):validate_fresh(data)

    def test_request_limit_survives_restart_and_counts_retries(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'cache.sqlite'
            client=Client(path,max_actual_calls=2,persist_attempts=True)
            with patch('core.urllib.request.urlopen',side_effect=urllib.error.URLError('offline')) as request, patch('core.time.sleep'):
                with self.assertRaises(BudgetExceeded):client.complete('test')
                self.assertEqual(request.call_count,2)
            self.assertEqual(client.attempted_requests(),2);client.db.close()
            resumed=Client(path,max_actual_calls=2,persist_attempts=True)
            with patch('core.urllib.request.urlopen') as request:
                with self.assertRaises(BudgetExceeded):resumed.complete('test')
                request.assert_not_called()
            resumed.db.close()

    def test_cache_hits_work_after_physical_cap(self):
        with tempfile.TemporaryDirectory() as d:
            client=Client(Path(d)/'cache.sqlite',max_actual_calls=1,persist_attempts=True)
            body=json.dumps(dict(choices=[dict(message=dict(content='Positive'),finish_reason='stop')],usage={})).encode()
            with patch('core.urllib.request.urlopen',return_value=io.BytesIO(body)):
                client.complete('cached')
            with patch('core.urllib.request.urlopen') as request:
                self.assertEqual(client.complete('cached')['text'],'Positive')
                request.assert_not_called()
            self.assertEqual(client.attempted_requests(),1);client.db.close()

    def test_pipeline_freezes_before_final_and_resumes_without_calls(self):
        data=manifest()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            out=Path(d);config=dict(split_hash=data['split_hash'],exclusion_hash=data['exclusion_hash'])
            save_json(out/'splits.json',data);pid=lock(out,config);client=FakeClient()
            with self.assertRaises(FileNotFoundError):evaluate(client,out,data,pid)
            self.assertEqual(client.prompts,[])
            searches(client,out,data,pid)
            self.assertFalse(any('confirmation 0 ' in p or 'amazon_test 0 ' in p for p in client.prompts))
            confirm(client,out,data,pid)
            self.assertFalse(any('amazon_test 0 ' in p for p in client.prompts))
            evaluate(client,out,data,pid);report(out)
            before=len(client.prompts)
            searches(client,out,data,pid);confirm(client,out,data,pid);evaluate(client,out,data,pid)
            self.assertEqual(len(client.prompts),before)
            result=json.loads((out/'summary.json').read_text())
            self.assertEqual(len(result['methods']),8)
            self.assertTrue(all(not d['accepted'] for d in result['confirmation'].values()))

    def test_prediction_corruption_is_rejected(self):
        e=split('source_test',1)[0]
        saved=dict(protocol_hash='p',prompt_id=fingerprint(BASE),instruction=BASE,
                   predictions=[dict(**asdict(e),raw='Negative',prediction='Negative',correct=0)])
        with self.assertRaises(ValueError):validate_predictions(saved,BASE,[e],'p')


if __name__=='__main__':unittest.main()
