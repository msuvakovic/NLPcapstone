"""Offline round-sweep regressions. No model or network required."""
import contextlib
import io
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from core import BASE, Client, Evaluator, Example, fingerprint, save_json
from verify_round_sweep import SerialReplayClient
from round_data import SIZES, SPLIT_SEED, spans, validate
from round_sweep import (CAPS, artifact, choose_cap, confirm, evaluate, final_freeze,
                         interval, lock, predictions, report, search, searches)


def examples(name, n=32):
    domain = 'amazon' if name == 'amazon_test' else 'tweets' if name == 'tweets_test' else 'sst2'
    return [Example(f'{name}:{i}', ' '.join(f'{name}_{i}_{j}' for j in range(12)),
                    'Positive' if i%2 else 'Negative', domain, name) for i in range(n)]


def manifest():
    data = {k:[asdict(e) for e in examples(k,n)] for k,n in SIZES.items()}
    return dict(seed=SPLIT_SEED, sizes=SIZES, splits=data, split_hash=fingerprint(data),
                excluded_texts=[], exclusion_hash=fingerprint([]))


class FakeClient:
    workers = 4
    def __init__(self, special=False):
        self.calls = []; self.special = special
    def snapshot(self):
        return dict(actual_calls=len(self.calls))
    def complete(self, prompt, **kwargs):
        self.calls.append((prompt,kwargs))
        finish = 'stop'
        if kwargs.get('temperature') == .7:
            round_number = kwargs['seed']%1000
            text = f'Instruction {kwargs["seed"]} for overall sentiment.'
            if self.special:
                if round_number == 1: text = BASE
                if round_number == 2: text = ''
                if round_number == 3: text = 'word '*81
                if round_number == 4: finish = 'length'
        else:
            value = prompt.split('\n\nText: ')[1].split()[0]
            index = int(value.split('_')[-2])
            text = 'Positive' if index%2 else 'Negative'
        return dict(text=text,usage=dict(prompt_tokens=10,completion_tokens=2),finish_reason=finish)


class RoundTests(unittest.TestCase):
    def setUp(self):
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
    def tearDown(self):
        self.stdout.__exit__(None,None,None)

    def test_rejections_consume_rounds_and_proposal_budget(self):
        client = FakeClient(special=True)
        train,val = examples('train'),examples('validation')
        ev = Evaluator(client,train+val,budget=576,proposal_limit=16)
        result = search(ev,train,val)
        self.assertEqual([r['rejection'] for r in result['rounds'][:4]],
                         ['duplicate','empty','over_80_words','truncated'])
        self.assertEqual(ev.proposal_count,16)
        self.assertEqual(ev.eval_count,64+12*32)
        self.assertEqual(set(result['checkpoints']),set(map(str,CAPS)))
        self.assertTrue(all(x['selected'] == BASE for x in result['checkpoints'].values()))
        self.assertEqual(result['checkpoints']['0']['resources']['logical_input_tokens'],0)
        self.assertEqual(result['checkpoints']['4']['resources']['proposal_calls'],4)

    def test_nested_prefix_matches_stored_round(self):
        train,val=examples('train'),examples('validation')
        result=search(Evaluator(FakeClient(),train+val,budget=576,seed=2),train,val)
        self.assertEqual(len(result['history']),17)
        for cap in CAPS[1:]:
            self.assertEqual(result['checkpoints'][str(cap)]['selected'],result['rounds'][cap-1]['selected'])
            self.assertEqual(result['checkpoints'][str(cap)]['resources']['proposal_calls'],cap)
        self.assertEqual(result['rounds'][0]['candidate'],'Instruction 2001 for overall sentiment.')

    def test_search_rejects_target_or_confirmation_before_dispatch(self):
        for name in ('confirmation','source_test','amazon_test','tweets_test'):
            train,val=examples(name),examples('validation')
            client=FakeClient()
            with self.assertRaises(ValueError):
                search(Evaluator(client,train+val,budget=576),train,val)
            self.assertEqual(client.calls,[])

    def test_evaluator_rejects_unseen_and_mutated_examples(self):
        train,val=examples('train'),examples('validation')
        client=FakeClient(); ev=Evaluator(client,train+val,budget=576)
        for rows in (examples('confirmation'),[replace(train[0],label='Positive')]):
            with self.assertRaises(ValueError):ev.evaluate(BASE,rows)
        self.assertEqual(client.calls,[])

    def test_fresh_data_checks_prior_exposure_spans_and_balance(self):
        data=manifest();validate(data)
        row=data['splits']['train'][0]
        data['excluded_texts']=[row['text'].upper()]
        # Exclusion entries are normalized in real manifests.
        data['excluded_texts']=[data['excluded_texts'][0].lower()]
        data['exclusion_hash']=fingerprint(data['excluded_texts'])
        with self.assertRaises(ValueError):validate(data)
        data=manifest()
        data['splits']['confirmation'][0]['text']='prefix '+data['splits']['train'][0]['text']+' suffix'
        data['split_hash']=fingerprint(data['splits'])
        with self.assertRaises(ValueError):validate(data)
        data=manifest();data['splits']['train'][0]['label']='Positive'
        data['split_hash']=fingerprint(data['splits'])
        with self.assertRaises(ValueError):validate(data)
        self.assertEqual(len(spans('one two three four five six seven eight nine')),2)

    def test_selects_earliest_within_predeclared_margin(self):
        vectors={str(cap):[[1]*60+[0]*4 for _ in range(3)] for cap in CAPS}
        self.assertEqual(choose_cap(vectors)['selected_cap'],0)
        vectors['2']=[[1]*62+[0]*2 for _ in range(3)]
        vectors['16']=[[1]*62+[0]*2,[1]*62+[0]*2,[1]*63+[0]]
        self.assertEqual(choose_cap(vectors)['selected_cap'],2)
        del vectors['8']
        with self.assertRaises(ValueError):choose_cap(vectors)

    def test_confirmation_interval_pairs_seeds_by_example(self):
        self.assertEqual(interval([0]*64,['Positive','Negative']*32),[0.,0.])
        lo,hi=interval([1]*32+[0]*32,['Positive','Negative']*32)
        self.assertLess(lo,.5);self.assertGreater(hi,.5)

    def test_selection_gains_can_overfit_and_confirmation_retains_base(self):
        class OverfittingClient(FakeClient):
            def complete(self, prompt, **kwargs):
                result=super().complete(prompt,**kwargs)
                if kwargs.get('temperature') == .7:
                    return result
                value=prompt.split('\n\nText: ')[1].split()[0]
                index=int(value.split('_')[-2])
                if prompt.startswith(BASE):
                    errors=4
                elif value.startswith('validation_'):
                    number=int(prompt.split()[1])%1000
                    errors=2 if number < 4 else 0
                else:
                    errors=8
                if index < errors:
                    result['text']='Negative' if result['text']=='Positive' else 'Positive'
                return result
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);data=manifest();pid=lock(out,dict(experiment='overfit'))
            client=OverfittingClient()
            frozen=searches(client,out,data,pid)
            for seed in range(3):
                ck=frozen['checkpoints'][str(seed)]
                self.assertEqual(ck['0']['selection_accuracy'],28/32)
                self.assertEqual(ck['1']['selection_accuracy'],30/32)
                self.assertEqual(ck['4']['selection_accuracy'],1.)
                self.assertEqual(ck['16']['selected'],ck['4']['selected'])
            decision=confirm(client,out,data,pid)
            self.assertEqual(decision['selected_cap'],0)
            self.assertEqual(decision['mean_accuracy']['0'],60/64)
            self.assertEqual(decision['mean_accuracy']['16'],56/64)
            final=evaluate(client,out,data,pid)
            self.assertEqual(len(final),7)
            result=report(out,data,pid)
            self.assertLess(result['final']['seed_0_rounds_16']['ood_accuracy'],
                            result['final']['base']['ood_accuracy'])

    def test_freeze_order_and_full_resume_without_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);data=manifest();pid=lock(out,dict(experiment='fake'))
            client=FakeClient()
            with self.assertRaises(FileNotFoundError):confirm(client,out,data,pid)
            with self.assertRaises(FileNotFoundError):evaluate(client,out,data,pid)
            self.assertEqual(client.calls,[])
            searches(client,out,data,pid)
            self.assertFalse(any('confirmation_' in p or 'amazon_test_' in p for p,_ in client.calls))
            selection=confirm(client,out,data,pid)
            self.assertEqual(selection['selected_cap'],0)
            self.assertFalse(any('amazon_test_' in p for p,_ in client.calls))
            evaluate(client,out,data,pid)
            result=report(out,data,pid)
            self.assertEqual(result['selected_cap'],0)
            self.assertEqual(result['final']['base']['ood_accuracy'],1.)
            before=len(client.calls)
            searches(client,out,data,pid);confirm(client,out,data,pid);evaluate(client,out,data,pid)
            self.assertEqual(len(client.calls),before)
            path=out/'searches/0.json'
            value=artifact(path,pid);value['seed']=123
            save_json(path,dict(protocol_hash=pid,data_hash=fingerprint(value),data=value))
            with self.assertRaises(ValueError):final_freeze(out,pid)

    def test_protocol_and_artifact_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);pid=lock(out,dict(model='A'))
            with self.assertRaises(ValueError):lock(out,dict(model='B'))
            artifact(out/'searches/0.json',pid,dict(a=1))
            save_json(out/'searches/0.json',dict(protocol_hash=pid,data_hash='wrong',data=dict(a=2)))
            with self.assertRaises(ValueError):lock(out,dict(model='A'))

    def test_prediction_tampering_rejected_even_if_rehashed(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);data=examples('confirmation',64);pid='p'
            predictions(FakeClient(),out,'confirmation_predictions',BASE,data,pid)
            path=out/'confirmation_predictions'/(fingerprint(BASE)+'.json')
            value=artifact(path,pid)
            value['predictions'][0]['correct']=0
            save_json(path,dict(protocol_hash=pid,data_hash=fingerprint(value),data=value))
            with self.assertRaises(ValueError):predictions(None,out,'confirmation_predictions',BASE,data,pid)

    def test_serial_replay_matches_exact_cache_and_never_dispatches(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache.sqlite'
            live=Client(path,model='fake',persist_attempts=True)
            body=b'{"choices":[{"message":{"content":"Positive"},"finish_reason":"stop"}],"usage":{}}'
            with patch('core.urllib.request.urlopen',return_value=io.BytesIO(body)):
                saved=live.complete('cached prompt')
            live.db.close()
            replay=SerialReplayClient(path,dict(model='fake',endpoint='http://localhost:11434/v1'))
            try:
                self.assertEqual(replay.workers,1)
                with patch('urllib.request.urlopen',side_effect=AssertionError('No network')) as network:
                    with ThreadPoolExecutor(max_workers=replay.workers) as pool:
                        results=list(pool.map(replay.complete,['cached prompt']*100))
                    self.assertTrue(all(r==saved for r in results))
                    with self.assertRaises(ValueError):replay.complete('cache miss')
                    network.assert_not_called()
                self.assertEqual(replay.db.execute('SELECT COUNT(*) FROM request_attempts').fetchone()[0],1)
            finally:
                replay.db.close()


if __name__ == '__main__':
    unittest.main()
