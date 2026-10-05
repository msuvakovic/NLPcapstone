"""Small, source-only output-cap experiment; keep final targets out of tuning."""
import concurrent.futures
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from core import BASE, Client, fingerprint, parse_label, save_json, task_prompt

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'output/token_caps_2026-10-04'
CAPS = [1, 2, 4, 8, 16]


def classify(client, phase, example, cap):
    response = client.complete(task_prompt(BASE, example['text']), max_tokens=cap)
    prediction = parse_label(response['text'])
    return dict(phase=phase, id=example['id'], label=example['label'], cap=cap,
                prediction=prediction, correct=int(prediction == example['label']), **response)


def metrics(rows, reference):
    ref = {r['id']:r for r in reference}
    assert len(rows) == len(ref) and {r['id'] for r in rows} == set(ref)
    f1 = []
    for label in ('Positive', 'Negative'):
        tp = sum(r['label'] == label and r['prediction'] == label for r in rows)
        fp = sum(r['label'] != label and r['prediction'] == label for r in rows)
        fn = sum(r['label'] == label and r['prediction'] != label for r in rows)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return dict(n=len(rows), accuracy=float(np.mean([r['correct'] for r in rows])),
                macro_f1=float(np.mean(f1)), invalid=sum(r['prediction']=='INVALID' for r in rows),
                limit_finishes=sum(r['finish_reason']=='length' for r in rows),
                matches_baseline=sum(r['prediction']==ref[r['id']]['prediction'] for r in rows),
                completion_tokens=float(np.mean([r['usage']['completion_tokens'] for r in rows])),
                prompt_tokens=float(np.mean([r['usage']['prompt_tokens'] for r in rows])),
                median_seconds=float(np.median([r['seconds'] for r in rows])),
                p95_seconds=float(np.quantile([r['seconds'] for r in rows],.95)),
                returned_models=dict(Counter(r['model'] for r in rows)))


def select_caps(summary):
    baseline = summary[16]
    eligible = [cap for cap, m in summary.items() if m['invalid']==0
                and m['accuracy'] >= baseline['accuracy'] and m['macro_f1'] >= baseline['macro_f1']
                and m['matches_baseline']==m['n']]
    if not eligible:
        raise ValueError('No reliable cap, including baseline; inspect outputs before continuing')
    natural = [cap for cap in eligible if summary[cap]['limit_finishes']==0]
    return dict(minimum_valid_cap=min(eligible),
                minimum_natural_stop_cap=min(natural) if natural else None,
                reference_cap=16)


def run_phase(client, phase, examples, caps, saved):
    completed = {(r['phase'],r['id'],r['cap']) for r in saved}
    jobs = [(e,c) for e in examples for c in caps if (phase,e['id'],c) not in completed]
    random.Random(20261004 + (phase=='confirmation')).shuffle(jobs)
    for start in range(0,len(jobs),20):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(classify,client,phase,e,c) for e,c in jobs[start:start+20]]
            for future in concurrent.futures.as_completed(futures):
                saved.append(future.result())
                save_json(OUTPUT/'responses.json',saved)
        print(phase, min(start+20,len(jobs)), '/',len(jobs),'pending jobs completed;',client.snapshot(),flush=True)
    phase_rows = [r for r in saved if r['phase']==phase]
    reference = [r for r in phase_rows if r['cap']==16]
    return {cap:metrics([r for r in phase_rows if r['cap']==cap],reference) for cap in caps}


def main():
    source = ROOT/'output/prompt_benchmark_2026-09-29'
    raw = json.loads((source/'splits.json').read_text(encoding='utf-8'))
    assert fingerprint(raw['splits'])==raw['split_hash']
    splits = raw['splits']
    selection = [r for label in ('Positive','Negative')
                 for r in [x for x in splits['sst_train'] if x['label']==label][:12]]
    confirmation = splits['sst_val']
    final_ids = {r['id'] for name,rows in splits.items() if name.endswith('_test') for r in rows}
    assert len(selection)==24 and len(confirmation)==64
    assert not ({r['id'] for r in selection+confirmation} & final_ids)
    assert not ({r['id'] for r in selection} & {r['id'] for r in confirmation})
    assert all(r['domain']=='sst2' and r['split'] in ('train','val') for r in selection+confirmation)
    protocol = dict(model='gemma4:31b-cloud',instruction=BASE,temperature=0.,seed=12345,
        caps=CAPS,reference_cap=16,workers=4,selection_ids=[r['id'] for r in selection],
        confirmation_ids=[r['id'] for r in confirmation],split_hash=raw['split_hash'],
        selection_rule='Smallest cap with valid labels, identical reference predictions and no accuracy/F1 loss. Also report smallest such cap with no length finishes.',
        confirmation_rule='Freeze both selected caps before confirmation; compare against cap 16.',
        request_limit=400,scope='Base-prompt binary sentiment classification only; no optimizer proposal/reflection changes.',
        timing='Randomized interleaved caps, four workers. Successful-request latency only; provider caching/load uncontrolled. Application cache is new for this experiment; resumes reuse original records.')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    manifest=OUTPUT/'protocol.json'
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding='utf-8'))==protocol, 'Protocol changed'
    else: save_json(manifest,protocol)
    saved=json.loads((OUTPUT/'responses.json').read_text(encoding='utf-8')) if (OUTPUT/'responses.json').exists() else []
    client=Client(OUTPUT/'requests.sqlite',max_actual_calls=400)
    try:
        pilot=run_phase(client,'selection',selection,CAPS,saved)
        chosen=select_caps(pilot)
        frozen=OUTPUT/'selected_caps.json'
        if frozen.exists(): assert json.loads(frozen.read_text(encoding='utf-8'))==chosen
        else: save_json(frozen,chosen)
        confirm_caps=sorted({16,chosen['minimum_valid_cap'],chosen['minimum_natural_stop_cap']}-{None})
        confirmed=run_phase(client,'confirmation',confirmation,confirm_caps,saved)
        # Report the preselected caps' confirmation, rather than choosing a new cap on these labels.
        valid_cap=chosen['minimum_valid_cap']
        matched=confirmed[valid_cap]['invalid']==0 and confirmed[valid_cap]['matches_baseline']==len(confirmation)
        result=dict(protocol=protocol,selection=pilot,chosen=chosen,confirmation=confirmed,
                    minimum_valid_cap_confirmed=matched,transport_this_execution=client.snapshot())
        save_json(OUTPUT/'summary.json',result)
        lines=['# Gemma output-token cap experiment','',
               'Source-only tuning: 24 balanced SST-2 training examples for selection and 64 separate SST-2 validation examples for confirmation. No final source or target tests were used to select a cap. The base instruction and decoding settings are fixed.', '']
        for title, summary in [('Selection (24 examples per cap)',pilot),('Confirmation (64 examples per cap)',confirmed)]:
            lines += ['## '+title,'','| Cap | Accuracy | Macro F1 | Invalid | Length finishes | Agreement with 16 | Mean output tokens | Median latency | p95 latency |',
                      '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
            for cap,m in sorted(summary.items()):
                lines.append(f"| {cap} | {m['accuracy']:.2%} | {m['macro_f1']:.4f} | {m['invalid']} | {m['limit_finishes']} | {m['matches_baseline']}/{m['n']} | {m['completion_tokens']:.2f} | {m['median_seconds']:.3f}s | {m['p95_seconds']:.3f}s |")
            lines += ['']
        lines += ['## Interpretation','',
            f"Preselected smallest complete-label cap: **{valid_cap}**. Confirmation matched all reference labels with no invalid outputs: **{matched}**.",
            f"Preselected smallest tested cap allowing natural stopping on the selection sample: **{chosen['minimum_natural_stop_cap']}**. See its confirmation length-finish count above. Cap 3 was not tested.", '',
            'A `length` finish means the token ceiling was reached; a complete Positive/Negative label can still be valid. Partial labels are invalid and count as errors. The table reports these separately.', '',
            'A cap is an upper bound, not the number of tokens always generated. Reducing 16 to 4 or 8 saves nothing when an answer already stops at two tokens. Input tokens are unchanged. Token reductions are not verified billing savings.', '',
            'These results cover one fixed sentiment prompt and one model endpoint. The small source-only confirmation cannot establish rare-failure rates or OOD accuracy. Provider caching and load were uncontrolled; single-request latency excludes failed attempts and is descriptive. Cached resumes do not create independent measurements.', '',
            'Do not apply a one-word classification cap to prompt proposals, reflections, NLI labels, explanations or reasoning tasks. Existing frozen benchmark settings and results remain unchanged.']
        (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('COMPLETE',json.dumps(result,indent=2),flush=True)
    finally: client.db.close()


if __name__=='__main__': main()
