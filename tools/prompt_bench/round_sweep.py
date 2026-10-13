"""Bounded three-seed Gemma round-count pilot; final tests never select rounds.

A round is one proposal attempt, including duplicates/invalid proposals. Prefix
checkpoints share a single trajectory per seed. This is the custom source-
feedback search from adaptive_search, not a DSPy/GEPA round-count comparison.
"""
import argparse
import json
import platform
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from adaptive_search import feedback_examples, proposal_prompt
from core import BASE, Client, Evaluator, fingerprint, parse_label, save_json, strip_fence
from fresh_data import ROOT, examples
from improvement_pilot import immutable, metrics, paired_interval, read
from round_data import SIZES, prepare, validate

CAPS = (0, 1, 2, 4, 8, 16)
SEEDS = (0, 1, 2)
MAX_REQUESTS = 4000
MARGIN = .01
DEFAULT_OUTPUT = ROOT/'output/round_sweep_2026-10-13'


def configuration(args, manifest):
    files = ('core.py', 'data.py', 'protocol.py', 'fresh_data.py', 'adaptive_search.py',
             'improvement_pilot.py', 'round_data.py', 'round_sweep.py')
    return dict(version=1, experiment='source-feedback-round-prefix-sweep',
        model=args.model, endpoint=args.endpoint.rstrip('/'), workers=4,
        caps=list(CAPS), seeds=list(SEEDS), max_persistent_attempts=MAX_REQUESTS,
        sizes=SIZES, split_hash=manifest['split_hash'], exclusion_hash=manifest['exclusion_hash'],
        classification=dict(max_tokens=16, temperature=0, seed=12345),
        proposal=dict(max_tokens=768, temperature=.7, words=80, seed='1000*seed+round'),
        selection='smallest cap within 1 percentage point of best mean source confirmation accuracy',
        selection_margin=MARGIN, final_arms='base, selected cap per seed, 16 rounds per seed',
        default_policy='retain base; this small pilot is exploratory, not a noninferiority proof',
        source_pool=manifest['source_sampling'], overlap_rule=manifest['overlap_rule'],
        python=platform.python_version(), numpy=np.__version__,
        code={f:fingerprint((ROOT/'tools/prompt_bench'/f).read_text(encoding='utf-8')) for f in files})


def artifact(path, pid, data=None):
    if data is not None:
        value = dict(protocol_hash=pid, data_hash=fingerprint(data), data=data)
        immutable(path, value)
    value = read(path)
    if value.get('protocol_hash') != pid or value.get('data_hash') != fingerprint(value.get('data')):
        raise ValueError('Changed or foreign artifact: ' + str(path))
    return value['data']


def lock(output, config):
    output.mkdir(parents=True, exist_ok=True)
    if not (output/'protocol.json').exists() and (output/'requests.sqlite').exists():
        raise ValueError('Existing response cache has no protocol')
    immutable(output/'protocol.json', config)
    pid = fingerprint(config)
    for folder in ('searches', 'confirmation_predictions', 'final_predictions'):
        for path in (output/folder).glob('*.json'):
            artifact(path, pid)
    for name in ('search_freeze.json', 'round_selection.json', 'final_freeze.json'):
        if (output/name).exists():
            artifact(output/name, pid)
    return pid


def search(ev, train, validation):
    if len(train) != 32 or len(validation) != 32 or ev.proposal_limit != 16:
        raise ValueError('Expected fixed 32/32 data and 16 proposal attempts')
    if any(e.domain != 'sst2' or e.split != 'train' for e in train) or any(
            e.domain != 'sst2' or e.split != 'validation' for e in validation):
        raise ValueError('Only source training and selection data allowed')
    if {e.id for e in train} & {e.id for e in validation}:
        raise ValueError('Overlapping search splits')
    base_train = ev.evaluate(BASE, train)
    feedback = feedback_examples(train, base_train, ev.seed)
    rows = ev.evaluate(BASE, validation)
    history = [dict(instruction=BASE, score=sum(r['correct'] for r in rows)/len(rows))]
    scores = {BASE: rows}
    rounds = []
    checkpoints = {'0':dict(selected=BASE, selection_accuracy=history[0]['score'],
        resources=dict(unique_example_evaluations=0, proposal_calls=0,
                       logical_input_tokens=0, logical_output_tokens=0))}
    for number in range(1, 17):
        # Truncation is a rejected attempt, not a shorter silently accepted prompt.
        try:
            raw = ev.propose(proposal_prompt(history, feedback, number-1))
            candidate = strip_fence(raw)
            reason = ('empty' if not candidate else 'duplicate' if candidate in scores else
                      'over_80_words' if len(candidate.split()) > 80 else None)
        except RuntimeError as exc:
            if str(exc) != 'Truncated proposal; do not silently evaluate an incomplete instruction':
                raise
            raw, candidate, reason = None, None, 'truncated'
        if reason is None:
            rows = ev.evaluate(candidate, validation)
            scores[candidate] = rows
            history.append(dict(instruction=candidate, score=sum(r['correct'] for r in rows)/len(rows)))
        winner = max(history, key=lambda r: (r['score'], r['instruction'] == BASE, -len(r['instruction'])))
        rounds.append(dict(round=number, candidate=candidate, raw=raw, rejection=reason,
                           selected=winner['instruction'], selection_accuracy=winner['score'],
                           resources=ev.summary()))
        if number in CAPS:
            checkpoints[str(number)] = dict(selected=winner['instruction'],
                selection_accuracy=winner['score'], resources=ev.summary())
        print(f'SEED {ev.seed} ROUND {number}/16 selection={winner["score"]:.4f} '
              f'rejection={reason}', flush=True)
    return dict(seed=ev.seed, rounds=rounds, checkpoints=checkpoints, history=history,
                selection_rows=scores, validation_ids=[e.id for e in validation],
                feedback=feedback, audit=ev.audit)


def searches(client, output, manifest, pid):
    records = {}
    train, validation = examples(manifest, 'train'), examples(manifest, 'validation')
    for seed in SEEDS:
        path = output/'searches'/f'{seed}.json'
        if path.exists():
            result = artifact(path, pid)
        else:
            ev = Evaluator(client, train+validation, budget=576, proposal_limit=16, seed=seed)
            result = search(ev, train, validation)
            artifact(path, pid, result)
        if result['seed'] != seed or len(result['rounds']) != 16 or set(result['checkpoints']) != set(map(str, CAPS)):
            raise ValueError('Incomplete or mismatched search')
        records[str(seed)] = result
    freeze = dict(search_hashes={k:fingerprint(r) for k,r in records.items()},
        checkpoints={k:r['checkpoints'] for k,r in records.items()})
    artifact(output/'search_freeze.json', pid, freeze)
    return freeze


def load_search_freeze(output, pid):
    freeze = artifact(output/'search_freeze.json', pid)
    if set(freeze['search_hashes']) != set(map(str, SEEDS)):
        raise ValueError('Not all seeds were frozen')
    for seed in SEEDS:
        result = artifact(output/'searches'/f'{seed}.json', pid)
        if (fingerprint(result) != freeze['search_hashes'][str(seed)] or
                result['checkpoints'] != freeze['checkpoints'][str(seed)]):
            raise ValueError('Search changed after freezing')
    return freeze


def predictions(client, output, folder, prompt, data, pid):
    path = output/folder/(fingerprint(prompt)+'.json')
    if not path.exists():
        ev = Evaluator(client, data, budget=len(data))
        rows = []
        for i in range(0, len(data), 32):
            batch = data[i:i+32]
            rows.extend(dict(**asdict(e), **r) for e,r in zip(batch, ev.evaluate(prompt, batch)))
        artifact(path, pid, dict(instruction=prompt, predictions=rows))
    result = artifact(path, pid)
    if result['instruction'] != prompt or len(result['predictions']) != len(data):
        raise ValueError('Wrong prediction prompt/length')
    for row, e in zip(result['predictions'], data):
        if (any(row.get(k) != v for k,v in asdict(e).items()) or
                row['prediction'] != parse_label(row['raw']) or
                row['correct'] != int(row['prediction'] == e.label)):
            raise ValueError('Prediction identity/scoring changed')
    return result['predictions']


def choose_cap(vectors):
    if set(vectors) != set(map(str, CAPS)):
        raise ValueError('Missing round checkpoint')
    values = {k:np.asarray(v, dtype=float) for k,v in vectors.items()}
    if any(v.shape != (3, 64) or not np.isin(v, [0,1]).all() for v in values.values()):
        raise ValueError('Require three seeds paired on all 64 confirmation examples')
    means = {k:float(v.mean()) for k,v in values.items()}
    best = max(means.values())
    selected = min(cap for cap in CAPS if means[str(cap)] >= best-MARGIN-1e-12)
    return dict(selected_cap=selected, best_mean_accuracy=best, mean_accuracy=means,
                margin=MARGIN, vectors=vectors)


def confirm(client, output, manifest, pid):
    freeze = load_search_freeze(output, pid)  # fail before any confirmation dispatch
    data = examples(manifest, 'confirmation')
    vectors = {}
    for cap in CAPS:
        vectors[str(cap)] = []
        for seed in SEEDS:
            prompt = freeze['checkpoints'][str(seed)][str(cap)]['selected']
            rows = predictions(client, output, 'confirmation_predictions', prompt, data, pid)
            vectors[str(cap)].append([r['correct'] for r in rows])
    selection = dict(choose_cap(vectors), search_freeze_hash=fingerprint(freeze))
    artifact(output/'round_selection.json', pid, selection)
    arms = {'base':BASE}
    for seed in SEEDS:
        for cap in sorted({selection['selected_cap'], 16}):
            arms[f'seed_{seed}_rounds_{cap}'] = freeze['checkpoints'][str(seed)][str(cap)]['selected']
    final = dict(arms=arms, selection_hash=fingerprint(selection))
    artifact(output/'final_freeze.json', pid, final)
    return selection


def final_freeze(output, pid):
    search_freeze = load_search_freeze(output, pid)
    selection = artifact(output/'round_selection.json', pid)
    final = artifact(output/'final_freeze.json', pid)
    if (selection['search_freeze_hash'] != fingerprint(search_freeze) or
            final['selection_hash'] != fingerprint(selection)):
        raise ValueError('Changed frozen selection')
    expected = {'base':BASE}
    for seed in SEEDS:
        for cap in sorted({selection['selected_cap'],16}):
            expected[f'seed_{seed}_rounds_{cap}'] = search_freeze['checkpoints'][str(seed)][str(cap)]['selected']
    if final['arms'] != expected:
        raise ValueError('Changed final evaluation arms')
    return final


def evaluate(client, output, manifest, pid):
    final = final_freeze(output, pid)  # all selections frozen before first final request
    data = [e for name in ('source_test','amazon_test','tweets_test') for e in examples(manifest,name)]
    return {name:predictions(client, output, 'final_predictions', prompt, data, pid)
            for name,prompt in final['arms'].items()}


def interval(deltas, labels):
    """Resample examples, keeping all seeds paired (not 192 independent rows)."""
    deltas = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(10132026)
    boot = np.zeros(5000)
    for label in ('Positive','Negative'):
        group = deltas[np.asarray(labels) == label]
        boot += group[rng.integers(len(group), size=(5000,len(group)))].mean(axis=1)/2
    return np.quantile(boot,[.025,.975]).tolist()


def report(output, manifest, pid):
    selection = artifact(output/'round_selection.json', pid)
    expected = choose_cap(selection['vectors'])
    if any(selection[k] != v for k,v in expected.items()):
        raise ValueError('Changed round decision')
    # Passing None cannot issue calls: a missing prediction causes failure.
    rows = evaluate(None, output, manifest, pid)
    freeze = load_search_freeze(output, pid)
    labels = [e.label for e in examples(manifest,'confirmation')]
    baseline = np.asarray(selection['vectors']['0']).mean(axis=0)
    curve = []
    for cap in CAPS:
        ck = [freeze['checkpoints'][str(seed)][str(cap)] for seed in SEEDS]
        vector = np.asarray(selection['vectors'][str(cap)]).mean(axis=0)
        curve.append(dict(rounds=cap, selection_accuracy=float(np.mean([x['selection_accuracy'] for x in ck])),
            confirmation_accuracy=float(vector.mean()),
            confirmation_seed_accuracy=np.asarray(selection['vectors'][str(cap)]).mean(axis=1).tolist(),
            delta_vs_base=float((vector-baseline).mean()), delta_ci95=interval(vector-baseline, labels),
            mean_search_evaluations=float(np.mean([x['resources']['unique_example_evaluations'] for x in ck])),
            mean_search_tokens=float(np.mean([x['resources']['logical_input_tokens']+
                x['resources']['logical_output_tokens'] for x in ck]))))
    final = {}
    base_metrics = metrics(rows['base'])
    for name, pred in rows.items():
        score = metrics(pred)
        score['ood_delta_ci95'] = paired_interval(pred, rows['base'])
        score['ood_delta'] = score['ood_accuracy']-base_metrics['ood_accuracy']
        final[name] = score
    summary = dict(protocol_hash=pid, selected_cap=selection['selected_cap'], curve=curve, final=final,
                   default=BASE, conclusion_scope='exploratory one-partition custom-search pilot')
    save_json(output/'summary.json', summary)
    lines = ['# Gemma4 optimization-round pilot', '',
        f'**Smallest cap within 1 percentage point of the best mean source confirmation score: '
        f'{selection["selected_cap"]} rounds.** This is an exploratory choice, not proof of equivalence '
        'or a universal optimum. The base prompt remains the default.', '',
        'Three search seeds; caps 0, 1, 2, 4, 8, 16 are nested prefixes of the same trajectory '
        'within each seed. One round means one rewrite attempt, even when duplicate, empty, too long '
        'or truncated. This custom source-feedback search is not a DSPy/GEPA round benchmark.', '',
        'Every new candidate is scored on the same 32 source selection examples. Proposals receive '
        'baseline feedback from 32 training examples. Base wins selection ties, then shorter '
        'instructions. All 18 checkpoint choices freeze before 64 separate source examples select '
        'the round cap. Final tests (32 source, 64 Amazon, 64 tweets) only audit frozen base, chosen '
        'cap and 16-round prompts. They never choose the cap or rewrite a prompt.', '',
        'SST-2 official validation was largely exhausted by prior runs. This experiment samples '
        'the official training pool with at least 12 words per example; it may contain phrases. '
        'Source percentages are not comparable to earlier validation-based pilots. All splits '
        'are balanced and exclude normalized texts and shared contiguous eight-word spans recovered '
        'from earlier local logs/manifests and from other current splits. This cannot guarantee '
        'sentence-family separation, semantic independence or absence from model pretraining.', '',
        'Classification is fixed at 16 output tokens, temperature 0, seed 12345; proposals use '
        '768 tokens, temperature 0.7, seed 1000*search_seed+round. Search-only cost is charged '
        'logically for every seed, regardless of exact-cache sharing. Round 0 needs no search; '
        'baseline measurements are evaluation overhead. Worst-case planned requests: 3,792; '
        'persistent hard cap: 4,000 including retries and interrupted attempts.', '',
        '| Rounds | Search selection accuracy | Confirmation accuracy (mean of seeds) | Delta vs base, pp (95% CI) | Mean search evaluations | Mean search tokens |',
        '| --- | ---: | ---: | --- | ---: | ---: |']
    for row in curve:
        lo,hi = row['delta_ci95']
        lines.append(f'| {row["rounds"]} | {row["selection_accuracy"]:.2%} | '
            f'{row["confirmation_accuracy"]:.2%} | {100*row["delta_vs_base"]:+.2f} '
            f'[{100*lo:+.2f}, {100*hi:+.2f}] | {row["mean_search_evaluations"]:.1f} | '
            f'{row["mean_search_tokens"]:.0f} |')
    lines += ['', 'Confirmation is development data used to choose the cap, not an unbiased final '
        'performance estimate. Intervals resample examples within labels with all seeds kept paired. '
        'They are descriptive, unadjusted for multiple comparisons, and may be degenerate when '
        'predictions match. Three seeds share one partition; small gains or rare failures remain '
        'unresolved. A 1 pp selection tolerance does not establish statistical noninferiority.', '',
        '| Frozen final arm | Source | Amazon | Tweets | Mean OOD | OOD delta, pp (95% CI) | Invalid |',
        '| --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for name, row in final.items():
        lo,hi = row['ood_delta_ci95']
        lines.append(f'| {name} | {row["sst2"]["accuracy"]:.2%} | {row["amazon"]["accuracy"]:.2%} | '
            f'{row["tweets"]["accuracy"]:.2%} | {row["ood_accuracy"]:.2%} | '
            f'{100*row["ood_delta"]:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] | {row["invalid"]} |')
    lines += ['', 'Identical selected prompts reuse predictions; repeated rows are not independent '
        'replications. Macro F1, per-seed scores, raw predictions, rejected attempts and costs '
        'are saved in JSON. Timing/cache/provider load do not establish latency or billed savings. '
        'The cloud alias cannot guarantee immutable model weights.', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


class ReplayClient:
    """Read-only exact responses; a cache miss fails without a network fallback."""
    workers = 4
    def __init__(self, path, config):
        self.db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True, check_same_thread=False)
        self.model, self.endpoint = config['model'], config['endpoint']
        self.calls = 0
    def snapshot(self):
        return dict(actual_calls=0)
    def complete(self, prompt, *, temperature=0., seed=12345, max_tokens=16):
        payload = dict(model=self.model, messages=[dict(role='user', content=prompt)],
                       temperature=temperature, seed=seed, max_tokens=max_tokens, stream=False)
        key = fingerprint(dict(endpoint=self.endpoint, payload=payload))
        row = self.db.execute('SELECT value FROM responses WHERE key=?', (key,)).fetchone()
        if row is None:
            raise ValueError('Missing exact cached response: ' + key)
        return json.loads(row[0])


def audit(output, manifest, pid, config):
    client = ReplayClient(output/'requests.sqlite', config)
    try:
        for seed in SEEDS:
            train, val = examples(manifest,'train'), examples(manifest,'validation')
            ev = Evaluator(client, train+val, budget=576, proposal_limit=16, seed=seed)
            rebuilt = search(ev, train, val)
            if rebuilt != artifact(output/'searches'/f'{seed}.json',pid):
                raise ValueError('Search cache replay mismatch')
        # Recompute every held-out prediction from cache, not from reported scores.
        for folder, names in [('confirmation_predictions',['confirmation']),
                             ('final_predictions',['source_test','amazon_test','tweets_test'])]:
            data = [e for name in names for e in examples(manifest,name)]
            for path in (output/folder).glob('*.json'):
                saved = artifact(path,pid)
                ev = Evaluator(client,data,budget=len(data))
                rebuilt = [dict(**asdict(e), **r) for e,r in zip(data,ev.evaluate(saved['instruction'],data))]
                if rebuilt != saved['predictions']:
                    raise ValueError('Prediction cache replay mismatch')
        confirm(client,output,manifest,pid)
        report(output,manifest,pid)
        count = client.db.execute('SELECT COUNT(*) FROM request_attempts').fetchone()[0]
        if count > MAX_REQUESTS:
            raise ValueError('Request cap exceeded')
        cached = [json.loads(r[0]) for r in client.db.execute('SELECT value FROM responses')]
        result = dict(protocol_hash=pid, passed=True, model_calls=0, attempted_requests=count,
            cached_responses=len(cached), attempts_without_cached_success=count-len(cached),
            returned_models=sorted({r['model'] for r in cached}),
            input_tokens=sum(r['usage'].get('prompt_tokens',0) for r in cached),
            output_tokens=sum(r['usage'].get('completion_tokens',0) for r in cached),
            checks='all three searches, every confirmation/final prediction, frozen selection, cap, report')
        save_json(output/'verification.json',result)
        return result
    finally:
        client.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--model',default='gemma4:31b-cloud')
    parser.add_argument('--endpoint',default='http://localhost:11434/v1')
    parser.add_argument('--stage',choices=['prepare','search','confirm','evaluate','report','audit','all'],default='prepare')
    args = parser.parse_args()
    manifest = prepare(args.output)
    config = configuration(args,manifest)
    pid = lock(args.output,config)
    if args.stage == 'prepare':
        print('Prepared',pid,'sizes',SIZES,flush=True)
        return
    if args.stage == 'audit':
        print(json.dumps(audit(args.output,manifest,pid,config),indent=2))
        return
    if args.stage == 'report':
        report(args.output,manifest,pid)
        return
    client = Client(args.output/'requests.sqlite',model=args.model,endpoint=args.endpoint,
                    workers=4,max_actual_calls=MAX_REQUESTS,persist_attempts=True)
    start = time.perf_counter()
    try:
        if args.stage in ('search','all'): searches(client,args.output,manifest,pid)
        if args.stage in ('confirm','all'): confirm(client,args.output,manifest,pid)
        if args.stage in ('evaluate','all'): evaluate(client,args.output,manifest,pid)
        if args.stage == 'all': report(args.output,manifest,pid)
    finally:
        save_json(args.output/'execution.json',dict(protocol_hash=pid,stage=args.stage,
            last_session=client.snapshot(),total_attempts=client.attempted_requests(),
            last_session_wall_seconds=time.perf_counter()-start))
        client.db.close()


if __name__ == '__main__':
    main()
