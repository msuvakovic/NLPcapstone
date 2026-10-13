"""Read-only evidence audit. Recompute scores from cached model responses, never infer."""
import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from adaptive_search import VARIANTS, confirmation_gate
from core import BASE, fingerprint, parse_label, save_json, task_prompt
from fresh_data import ROOT, examples, validate_fresh
from improvement_pilot import read, validate_bindings, validate_predictions


def audit(output):
    output = Path(output)
    protocol = read(output/'protocol.json'); pid = fingerprint(protocol)
    manifest = read(output/'splits.json'); validate_fresh(manifest)
    assert manifest['split_hash'] == protocol['split_hash']
    assert manifest['exclusion_hash'] == protocol['exclusion_hash']
    for name, saved_hash in protocol['code'].items():
        assert fingerprint((ROOT/'tools/prompt_bench'/name).read_text(encoding='utf-8')) == saved_hash, name
    validate_bindings(output, pid)
    all_examples = {e.id:e for name in manifest['splits'] for e in examples(manifest,name)}
    allowed = {e.id for name in ('train','validation') for e in examples(manifest,name)}
    validation = examples(manifest,'validation')
    cache_path = (output/'requests.sqlite').resolve()
    cache = sqlite3.connect(cache_path.as_uri()+'?mode=ro', uri=True)
    try:
        attempts = cache.execute('SELECT COUNT(*) FROM request_attempts').fetchone()[0]
        assert attempts <= protocol['max_persistent_attempts']
        returned_models = Counter(json.loads(row[0]).get('model')
                                  for row in cache.execute('SELECT value FROM responses'))
        checked = 0
        def cached_prediction(instruction, example):
            nonlocal checked
            payload = dict(model=protocol['model'], messages=[dict(role='user',content=task_prompt(instruction,example.text))],
                           **protocol['decoding'], stream=False)
            key = fingerprint(dict(endpoint=protocol['endpoint'],payload=payload))
            row = cache.execute('SELECT value FROM responses WHERE key=?',(key,)).fetchone()
            assert row is not None, 'Missing cached prediction: ' + example.id
            checked += 1
            return json.loads(row[0])
        for variant in VARIANTS:
            result = read(output/'searches'/(variant+'.json'))
            resources = result['resources']
            assert resources['unique_example_evaluations'] <= 512
            assert resources['proposal_calls'] <= 16
            assert sum(batch['new_evaluations'] for batch in result['evaluation_audit']) == resources['unique_example_evaluations']
            assert all(set(batch['example_ids']) <= allowed for batch in result['evaluation_audit'])
            assert len(result['vectors']) == len(result['finalists'])
            for instruction, vector in zip(result['finalists'],result['vectors']):
                computed = [int(parse_label(cached_prediction(instruction,e)['text']) == e.label) for e in validation]
                assert computed == vector, 'Validation cache mismatch: ' + variant
            for stage in result['stages']:
                assert len(stage['ids']) == stage['size']
                for entry in stage['scores']:
                    values = [int(parse_label(cached_prediction(entry['instruction'],all_examples[eid])['text'])
                                  == all_examples[eid].label) for eid in stage['ids']]
                    assert sum(values)/len(values) == entry['score'], 'Stage score mismatch'
            winner = max(range(len(result['finalists'])), key=lambda i:(sum(result['vectors'][i]),
                result['finalists'][i] == BASE, -len(result['finalists'][i]), -i))
            assert result['selected'] == result['finalists'][winner]
        confirmation_data = examples(manifest,'confirmation')
        confirmation = read(output/'confirmation.json')
        base = read(output/'confirmation_predictions'/(fingerprint(BASE)+'.json'))
        for name, decision in confirmation['decisions'].items():
            challenger = read(output/'confirmation_predictions'/(fingerprint(decision['raw_prompt'])+'.json'))
            computed = confirmation_gate([r['correct'] for r in base['predictions']],
                [r['correct'] for r in challenger['predictions']], comparisons=3)
            assert all(decision[k] == v for k,v in computed.items()), 'Confirmation decision mismatch: ' + name
            assert decision['selected'] == (decision['raw_prompt'] if computed['accepted'] else BASE)
        total_predictions = 0
        for folder, data in [('confirmation_predictions',confirmation_data),
            ('final_predictions',[e for k in ('source_test','amazon_test','tweets_test') for e in examples(manifest,k)])]:
            for path in (output/folder).glob('*.json'):
                record = read(path)
                validate_predictions(record,record['instruction'],data,pid)
                for example,row in zip(data,record['predictions']):
                    cached = cached_prediction(record['instruction'],example)
                    assert row['raw'] == cached['text'] and row['usage'] == cached['usage'], 'Raw-response mismatch'
                    total_predictions += 1
        frozen = read(output/'final_freeze.json')
        unique_final = set(frozen['selections'].values())
        assert {p.stem for p in (output/'final_predictions').glob('*.json')} == {fingerprint(p) for p in unique_final}
        return dict(status='PASS',protocol_hash=pid,searches=len(VARIANTS),
            fresh_examples=len(all_examples),excluded_texts=len(manifest['excluded_texts']),
            unique_final_prompts=len(unique_final),stored_predictions=total_predictions,
            cache_checks=checked,physical_attempts=attempts,physical_cap=4000,
            returned_models=dict(returned_models),new_model_requests=0)
    finally:
        cache.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='output/improvement_pilot_2026-10-04')
    args=parser.parse_args()
    result=audit(args.output)
    save_json(Path(args.output)/'artifact_audit.json',result)
    print(json.dumps(result,indent=2))
