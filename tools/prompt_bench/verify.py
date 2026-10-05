"""Audit completed experiment artifacts before interpreting performance."""
import argparse
import json
from pathlib import Path

from core import fingerprint
from run import assert_complete_searches
from protocol import validate_artifact_bindings


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='output/prompt_benchmark_2026-09-29')
    parser.add_argument('--search-only',action='store_true')
    args=parser.parse_args(); root=Path(args.output)
    raw=json.loads((root/'splits.json').read_text(encoding='utf-8'))
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    if protocol.get('protocol_version') == 2:
        if protocol['split_hash'] != fingerprint(raw['splits']):
            raise ValueError('Data no longer matches the protocol')
        validate_artifact_bindings(root, protocol)
    splits=raw['splits']; seen=set(); total=0
    for name,rows in splits.items():
        labels=[r['label'] for r in rows]
        assert labels.count('Positive')==labels.count('Negative'), name
        for r in rows:
            text=' '.join(r['text'].lower().split())
            assert text not in seen, ('Cross-split duplicate',name,r['id'])
            seen.add(text);total+=1
    assert fingerprint(splits)==raw['split_hash']
    tests={r['id']:r for name,rows in splits.items() if name.endswith('_test') for r in rows}
    search_files=list(root.glob('*/search.json'))
    assert_complete_searches(root,[json.loads(p.read_text(encoding='utf-8')) for p in search_files])
    for p in search_files:
        run=json.loads(p.read_text(encoding='utf-8'))
        allowed=set(run['training_ids']+run['validation_ids'])
        assert not (allowed & set(tests)), p
        if run['regime']=='single':
            assert all(i.startswith('sst2:') for i in allowed),p
        assert not any(i.startswith('tweets:') for i in allowed),p
        for batch in run['evaluation_audit']:
            assert set(batch['example_ids']) <= allowed, p
        res=run['resources']
        assert res['unique_example_evaluations']<=res['evaluation_cap'],p
        assert res['proposal_calls']<=res['proposal_cap'],p
        assert sum(x['new_evaluations'] for x in run['evaluation_audit'])==res['unique_example_evaluations'],p
        assert all(len(v)==len(run['validation_ids']) for v in run['vectors']),p
        assert all(score in (0,1) for v in run['vectors'] for score in v),p
        assert all(s['instruction'] in run['candidates'] for s in run['selections'].values()),p
        if run['method']=='gepa':
            official=run['official_result']
            for candidate,scores in zip(official['candidates'],official['val_subscores']):
                actual=dict(zip((fingerprint(i) for i in run['validation_ids']),
                                run['vectors'][run['candidates'].index(candidate['instruction'])]))
                assert actual==scores,('GEPA cache mismatch',p)
    if args.search_only:
        print(f'PASS: {total} distinct examples; {len(search_files)} completed capped searches; source isolation and GEPA validation scores verified.')
        return
    frozen=json.loads((root/'frozen_prompts.json').read_text(encoding='utf-8'))
    prediction_files=list((root/'final_predictions').glob('*.json'))
    assert set(frozen)=={p.stem for p in prediction_files},'Incomplete final evaluation'
    for p in prediction_files:
        obj=json.loads(p.read_text(encoding='utf-8'))
        assert fingerprint(obj['instruction'])==p.stem
        assert len(obj['predictions'])==len(tests)
        assert {r['id'] for r in obj['predictions']}==set(tests)
        for row in obj['predictions']:
            assert row['correct']==int(row['prediction']==row['label'])
            assert all(row[k]==tests[row['id']][k] for k in ['text','label','domain','split'])
    print(f'PASS: {total} distinct examples; {len(search_files)} capped source-only search runs; '
          f'{len(prediction_files)} frozen prompts with {len(tests)} final predictions each.')


if __name__=='__main__': main()
