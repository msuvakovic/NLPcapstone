"""Validate repaired OPRO against saved real responses; never send API requests."""
import json
import argparse
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from core import BASE, Client, Evaluator, Example, fingerprint, save_json
from methods import ROOT, run_opro
from prompt_opt_harness.datasets import Dataset, DomainSplit
from prompt_opt_harness.harness import validate_splits


def main():
    benchmark = ROOT/'output/prompt_benchmark_2026-09-29'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'output/leakage_fix_2026-10-04')
    output = parser.parse_args().output
    raw = json.loads((benchmark/'splits.json').read_text(encoding='utf-8'))
    assert fingerprint(raw['splits']) == raw['split_hash']
    splits = {k:[Example(**row) for row in rows] for k,rows in raw['splits'].items()}
    train, val = splits['sst_train'], splits['sst_val']
    tests = splits['sst_test']+splits['amazon_test']+splits['tweets_test']
    validate_splits(Dataset('sst2', DomainSplit(train+val,splits['sst_test']),
                           {'amazon':splits['amazon_test'],'tweets':splits['tweets_test']}))
    # Zero physical-call cap plus a network tripwire: a cache miss must fail.
    client = Client(benchmark/'requests.sqlite', max_actual_calls=0)
    runs = []
    try:
        with patch('urllib.request.urlopen', side_effect=AssertionError('Network disabled for replay')):
            for seed in (0,1,2):
                ev = Evaluator(client, train+val, budget=512, seed=seed)
                result = run_opro(ev,train,val,seed,output/f'seed_{seed}')
                original = json.loads((benchmark/f'single_opro_{seed}/search.json').read_text(encoding='utf-8'))
                for key in ('candidates','vectors','original_selection'):
                    assert result[key] == original[key], f'Seed {seed}: replay differs in {key}'
                assert ev.eval_count == original['resources']['unique_example_evaluations']
                assert ev.proposal_count == original['resources']['proposal_calls']
                assert all(set(batch['example_ids']) <= {e.id for e in val} for batch in ev.audit)
                # Freeze the chosen prompt before giving the final evaluator any tests.
                selected = result['original_selection']
                final = Evaluator(client,tests,budget=len(tests))
                rows = [dict(**asdict(e),**r) for e,r in zip(tests,final.evaluate(selected,tests))]
                original_rows = json.loads((benchmark/'final_predictions'/f'{fingerprint(selected)}.json').read_text(encoding='utf-8'))['predictions']
                assert rows == original_rows, 'Final prediction replay differs'
                accuracy = {d:sum(r['correct'] for r in rows if r['domain']==d)/300
                            for d in ('sst2','amazon','tweets')}
                result.update(seed=seed,resources=ev.summary(),evaluation_audit=ev.audit,
                              final_accuracy=accuracy,provenance='Replay of September 29 real responses; no fresh inference')
                save_json(output/f'opro_seed_{seed}.json',result)
                runs.append(dict(seed=seed,selected_prompt=selected,accuracy=accuracy,
                                 source_validation_accuracy=max(map(lambda v:sum(v)/len(v),result['vectors'])),
                                 search_evaluations=ev.eval_count,proposal_calls=ev.proposal_count))
        stats = client.snapshot()
        assert stats['actual_calls'] == 0
        summary = dict(result='PASS',model=client.model,kind='Post-fix cache-only real-response replay',
                       source_artifacts=str(benchmark),runs=runs,transport=stats,
                       source_hashes={name:fingerprint((ROOT/name).read_text(encoding='utf-8')) for name in (
                           'prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/harness.py',
                           'prompt_opt_harness/prompt_opt_harness/prompt_opt_harness/optimizers/opro.py')})
        save_json(output/'replay_summary.json',summary)
        print(json.dumps(summary,indent=2))
    finally:
        client.db.close()


if __name__ == '__main__': main()
