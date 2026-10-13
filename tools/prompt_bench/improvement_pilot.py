"""Frozen, small Gemma pilot: source feedback, racing, independent confirmation."""
import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from adaptive_search import VARIANTS, confirmation_gate, run_search
from core import BASE, HUMAN, Client, Evaluator, difference, fingerprint, parse_label, save_json
from fresh_data import ROOT, SIZES, examples, prepare_fresh, validate_fresh


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen artifact changed: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)


def configuration(args, manifest):
    files = ('core.py', 'data.py', 'protocol.py', 'fresh_data.py', 'adaptive_search.py', 'improvement_pilot.py')
    return dict(version=1, experiment='source-feedback-racing-confirmation', model=args.model,
        endpoint=args.endpoint.rstrip('/'), workers=4, seeds=[0], variants=list(VARIANTS),
        evaluation_cap=512, proposal_cap=16, max_persistent_attempts=4000,
        sizes=SIZES, split_hash=manifest['split_hash'], exclusion_hash=manifest['exclusion_hash'],
        split_seed=manifest['seed'], decoding=dict(temperature=0.0, seed=12345, max_tokens=16),
        proposal_decoding=dict(temperature=0.7, max_tokens=768, words=80),
        confirmation=dict(alpha=0.05, minimum_gain=0.01, comparisons=3),
        python=platform.python_version(), numpy=np.__version__,
        code={name:fingerprint((ROOT/'tools/prompt_bench'/name).read_text(encoding='utf-8')) for name in files})


def validate_bindings(output, protocol_hash):
    for directory in ('searches', 'confirmation_predictions', 'final_predictions'):
        for path in (output/directory).glob('*.json'):
            if read(path).get('protocol_hash') != protocol_hash:
                raise ValueError('Foreign artifact: ' + str(path))
    for name in ('search_freeze.json', 'confirmation.json', 'final_freeze.json'):
        if (output/name).exists() and read(output/name).get('protocol_hash') != protocol_hash:
            raise ValueError('Foreign frozen artifact: ' + name)
    if (output/'search_freeze.json').exists():
        freeze = read(output/'search_freeze.json')
        if set(freeze['selections']) != set(VARIANTS):
            raise ValueError('Incomplete search freeze')
        for variant in VARIANTS:
            search = read(output/'searches'/(variant+'.json'))
            if (fingerprint(search) != freeze['search_hashes'][variant] or
                search['selected'] != freeze['selections'][variant] or
                search['selected'] not in search['finalists']):
                raise ValueError('Search changed after freezing')
    if (output/'confirmation.json').exists():
        confirmation = read(output/'confirmation.json')
        if confirmation['search_freeze_hash'] != fingerprint(read(output/'search_freeze.json')):
            raise ValueError('Confirmation belongs to different search freeze')
    if (output/'final_freeze.json').exists():
        final = read(output/'final_freeze.json')
        confirmation = read(output/'confirmation.json')
        expected = {'base':BASE, 'human':HUMAN}
        for name, decision in confirmation['decisions'].items():
            expected[name] = decision['raw_prompt']
            expected[name+'+guard'] = decision['selected']
        if final['selections'] != expected or final['confirmation_hash'] != fingerprint(confirmation):
            raise ValueError('Final selections changed after confirmation')


def lock(output, config):
    path = output/'protocol.json'
    if not path.exists() and any(p.name != 'splits.json' for p in output.iterdir()):
        raise ValueError('Unbound artifacts in output directory')
    immutable(path, config)
    pid = fingerprint(config)
    validate_bindings(output, pid)
    return pid


def validate_predictions(saved, instruction, data, pid):
    if (saved['instruction'] != instruction or saved['protocol_hash'] != pid or
        saved['prompt_id'] != fingerprint(instruction) or
        [r['id'] for r in saved['predictions']] != [e.id for e in data]):
        raise ValueError('Saved prediction identity mismatch')
    for row, example in zip(saved['predictions'], data):
        if any(row[key] != value for key, value in asdict(example).items()):
            raise ValueError('Saved prediction data mismatch')
        if row['prediction'] != parse_label(row['raw']) or row['correct'] != int(row['prediction'] == example.label):
            raise ValueError('Saved prediction scoring mismatch')


def evaluate_saved(client, output, folder, instruction, data, pid):
    prompt_id = fingerprint(instruction)
    path = output/folder/(prompt_id+'.json')
    if path.exists():
        saved = read(path)
        validate_predictions(saved, instruction, data, pid)
        return saved
    print('EVALUATE', folder, prompt_id[:12], len(data), flush=True)
    start = time.perf_counter(); before = client.snapshot()
    ev = Evaluator(client, data, budget=len(data))
    predictions = []
    for i in range(0, len(data), 32):
        batch = data[i:i+32]
        predictions.extend(dict(**asdict(e), **r) for e, r in zip(batch, ev.evaluate(instruction, batch)))
    result = dict(protocol_hash=pid, prompt_id=prompt_id, instruction=instruction,
        predictions=predictions, physical_resources=difference(before, client.snapshot()),
        logical_resources=ev.summary(), wall_seconds=time.perf_counter()-start)
    save_json(path, result)
    return result


def searches(client, output, manifest, pid):
    train, validation = examples(manifest, 'train'), examples(manifest, 'validation')
    records = {}
    for variant in VARIANTS:
        path = output/'searches'/(variant+'.json')
        if path.exists():
            records[variant] = read(path)
            continue
        print('SEARCH', variant, flush=True)
        before = client.snapshot(); start = time.perf_counter()
        ev = Evaluator(client, train+validation, budget=512, proposal_limit=16, seed=0)
        result = run_search(ev, train, validation, 0, variant)
        result.update(protocol_hash=pid, seed=0, wall_seconds=time.perf_counter()-start,
                      physical_resources=difference(before, client.snapshot()))
        save_json(path, result); records[variant] = result
        print('SEARCH DONE', variant, 'candidates', len(result['candidates']),
              'validation', result['validation_accuracy'], flush=True)
    freeze = dict(protocol_hash=pid, selections={name:r['selected'] for name,r in records.items()},
                  search_hashes={name:fingerprint(r) for name,r in records.items()})
    immutable(output/'search_freeze.json', freeze)
    return freeze


def confirm(client, output, manifest, pid):
    freeze = read(output/'search_freeze.json')
    if set(freeze['selections']) != set(VARIANTS):
        raise ValueError('Searches incomplete')
    for variant in VARIANTS:
        result = read(output/'searches'/(variant+'.json'))
        if fingerprint(result) != freeze['search_hashes'][variant] or result['selected'] != freeze['selections'][variant]:
            raise ValueError('Search changed after freeze')
    data = examples(manifest, 'confirmation')
    base_rows = evaluate_saved(client, output, 'confirmation_predictions', BASE, data, pid)['predictions']
    base = [r['correct'] for r in base_rows]
    decisions = {}
    for variant, prompt in freeze['selections'].items():
        rows = evaluate_saved(client, output, 'confirmation_predictions', prompt, data, pid)['predictions']
        decision = confirmation_gate(base, [r['correct'] for r in rows], comparisons=len(VARIANTS))
        decision.update(raw_prompt=prompt, selected=prompt if decision['accepted'] else BASE)
        decisions[variant] = decision
    result = dict(protocol_hash=pid, search_freeze_hash=fingerprint(freeze), decisions=decisions)
    immutable(output/'confirmation.json', result)
    selections = {'base':BASE, 'human':HUMAN}
    for variant, prompt in freeze['selections'].items():
        selections[variant] = prompt
        selections[variant+'+guard'] = decisions[variant]['selected']
    immutable(output/'final_freeze.json', dict(protocol_hash=pid, selections=selections,
                                             confirmation_hash=fingerprint(result)))


def evaluate(client, output, manifest, pid):
    freeze = read(output/'final_freeze.json')
    if freeze['confirmation_hash'] != fingerprint(read(output/'confirmation.json')):
        raise ValueError('Confirmation changed after final freeze')
    data = [e for name in ('source_test', 'amazon_test', 'tweets_test') for e in examples(manifest, name)]
    for prompt in dict.fromkeys(freeze['selections'].values()):
        evaluate_saved(client, output, 'final_predictions', prompt, data, pid)


def metrics(rows):
    result = {}
    for domain in ('sst2', 'amazon', 'tweets'):
        group = [r for r in rows if r['domain'] == domain]
        f1s = []
        for label in ('Positive', 'Negative'):
            tp = sum(r['prediction'] == label and r['label'] == label for r in group)
            fp = sum(r['prediction'] == label and r['label'] != label for r in group)
            fn = sum(r['prediction'] != label and r['label'] == label for r in group)
            f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        result[domain] = dict(accuracy=sum(r['correct'] for r in group)/len(group),
                              macro_f1=sum(f1s)/2, n=len(group))
    result['ood_accuracy'] = (result['amazon']['accuracy']+result['tweets']['accuracy'])/2
    result['worst_ood_accuracy'] = min(result[d]['accuracy'] for d in ('amazon','tweets'))
    result['invalid'] = sum(r['prediction'] == 'INVALID' for r in rows)
    result['mean_input_tokens'] = float(np.mean([r['usage'].get('prompt_tokens',0) for r in rows]))
    return result


def paired_interval(rows, base):
    if [r['id'] for r in rows] != [r['id'] for r in base]:
        raise ValueError('Unpaired final predictions')
    rng = np.random.default_rng(1042026)
    boot = np.zeros(4000)
    for domain in ('amazon', 'tweets'):
        for label in ('Positive', 'Negative'):
            ids = [i for i,r in enumerate(rows) if r['domain'] == domain and r['label'] == label]
            diffs = np.array([rows[i]['correct']-base[i]['correct'] for i in ids])
            boot += diffs[rng.integers(0, len(ids), size=(4000, len(ids)))].mean(axis=1)/4
    return np.quantile(boot, [.025,.975]).tolist()


def report(output):
    protocol = read(output/'protocol.json'); pid = fingerprint(protocol)
    manifest = read(output/'splits.json'); validate_fresh(manifest); validate_bindings(output, pid)
    if manifest['split_hash'] != protocol['split_hash'] or manifest['exclusion_hash'] != protocol['exclusion_hash']:
        raise ValueError('Report data differs from the protocol')
    freeze = read(output/'final_freeze.json')
    expected = [e for k in ('source_test','amazon_test','tweets_test') for e in examples(manifest,k)]
    all_rows = {}
    for prompt in dict.fromkeys(freeze['selections'].values()):
        record = read(output/'final_predictions'/(fingerprint(prompt)+'.json'))
        validate_predictions(record, prompt, expected, pid)
        all_rows[prompt] = record['predictions']
    base = all_rows[BASE]; summary = []
    for name, prompt in freeze['selections'].items():
        rows = all_rows[prompt]
        row = dict(method=name, prompt_id=fingerprint(prompt), **metrics(rows))
        row['ood_delta'] = row['ood_accuracy'] - metrics(base)['ood_accuracy']
        row['ood_delta_ci95'] = paired_interval(rows, base)
        row['prompt_words'] = len(prompt.split())
        summary.append(row)
    results = {v:read(output/'searches'/(v+'.json')) for v in VARIANTS}
    decisions = read(output/'confirmation.json')['decisions']
    lines = ['# Fresh Gemma4 improvement pilot', '',
        'One optimizer seed; three custom history-based search variants. These are component ablations, '
        'not a new reference OPRO or GEPA reproduction. All arms use 512 candidate/example evaluations '
        'and at most 16 proposals. All pay for the same 64-example baseline training diagnostic.', '',
        'Training/selection/confirmation/source-test sizes: 64/64/80/64 SST-2 examples; '
        '160 fresh Amazon and 160 fresh tweet examples. Exact normalized texts recovered from previous '
        f'manifests and logs ({len(manifest["excluded_texts"])} unique texts) were excluded. '
        'This cannot exclude unlogged human exposure, semantic paraphrases, or pretraining contamination.', '',
        'History-full and feedback-full differ only in the source-training examples included in proposals. '
        'Feedback-racing screens 12 new candidates on 16 training examples, expands four to 32, then '
        'compares two survivors plus the base on the same 64 selection examples. '
        'All proposal variants include concise/simplification edits and an 80-word limit; brevity is not a separate ablation.', '',
        'All raw winners were frozen before source confirmation. The optional guard compares each winner '
        'with the base using a one-sided exact paired test, Bonferroni alpha 0.05/3 and at least 1 pp gain. '
        'A rejection means insufficient confirmation, not proof of equal quality. All deployed and raw '
        'prompts were frozen before final evaluation; the same saved predictions serve identical prompts.', '',
        '| Method | Source | Amazon | Tweets | Mean OOD | Worst OOD | OOD delta, pp (95% CI) | Input tokens/example |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    for row in summary:
        lo, hi = row['ood_delta_ci95']
        lines.append(f"| {row['method']} | {row['sst2']['accuracy']:.2%} | {row['amazon']['accuracy']:.2%} | "
            f"{row['tweets']['accuracy']:.2%} | {row['ood_accuracy']:.2%} | {row['worst_ood_accuracy']:.2%} | "
            f"{100*row['ood_delta']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] | {row['mean_input_tokens']:.1f} |")
    lines += ['', '| Search | Candidates screened | Full selection scores | Evaluations | Proposals | Search tokens | Seconds |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, r in results.items():
        cost = r['resources']
        lines.append(f"| {name} | {len(r['candidates'])} | {len(r['finalists'])} | "
            f"{cost['unique_example_evaluations']} | {cost['proposal_calls']} | "
            f"{cost['logical_input_tokens']+cost['logical_output_tokens']} | {r['wall_seconds']:.1f} |")
    lines += ['', '## Confirmation and interpretation', '']
    for name, d in decisions.items():
        lines.append(f"- {name}: {d['corrected']} fixes, {d['regressed']} regressions; "
                     f"p={d['p_value']:.4f}; {'accepted' if d['accepted'] else 'base retained'}.")
    lines += ['', 'Macro-F1 and invalid counts are in summary.json. Paired final-test intervals resample '
        'within domain/label groups; they describe sample uncertainty conditional on this one seed. '
        'They are exploratory and not adjusted for all final comparisons. Repeated seeds, independent '
        'data partitions and another model are needed for broad claims. A smaller source/OOD gap alone '
        'is not a performance gain.', '',
        'Tokens are reported, not equated: matching evaluation/proposal caps is not matching token cost. '
        'Timing includes cache reuse, provider load and execution order, so it does not establish a '
        'controlled latency speedup. Output caps stay at 16 in every arm.']
    save_json(output/'summary.json', dict(protocol_hash=pid, methods=summary,
        confirmation=decisions, searches={v:dict(resources=r['resources'],
        physical_resources=r['physical_resources'], wall_seconds=r['wall_seconds']) for v,r in results.items()}))
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/improvement_pilot_2026-10-04')
    parser.add_argument('--stage', choices=['prepare','search','confirm','evaluate','report','all'], default='prepare')
    parser.add_argument('--model', default='gemma4:31b-cloud')
    parser.add_argument('--endpoint', default='http://localhost:11434/v1')
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'report':
        report(output); return
    if (output/'protocol.json').exists() and not (output/'splits.json').exists():
        raise ValueError('Missing frozen splits; cannot regenerate')
    if args.stage in ('confirm','evaluate') and not (output/'protocol.json').exists():
        raise ValueError('Missing protocol')
    manifest = prepare_fresh(output)
    pid = lock(output, configuration(args, manifest))
    if args.stage == 'prepare':
        print('FROZEN', pid, 'split sizes', {k:len(v) for k,v in manifest['splits'].items()}); return
    client = Client(output/'requests.sqlite', model=args.model, endpoint=args.endpoint, workers=4,
                    max_actual_calls=4000, persist_attempts=True)
    try:
        if args.stage in ('search','all'): searches(client, output, manifest, pid)
        if args.stage in ('confirm','all'): confirm(client, output, manifest, pid)
        if args.stage in ('evaluate','all'): evaluate(client, output, manifest, pid)
        if args.stage == 'all': report(output)
    finally:
        save_json(output/'execution.json', dict(protocol_hash=pid, attempted_requests=client.attempted_requests(),
            cap=4000, last_session=client.snapshot()))
        client.db.close()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
