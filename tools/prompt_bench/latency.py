"""Counterbalanced concurrency measurements, run alone after accuracy evaluation."""
import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from core import BASE,Client,Example,difference,fingerprint,parse_label,save_json,task_prompt
from protocol import validate_execution_context


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='output/prompt_benchmark_2026-09-29')
    parser.add_argument('--workers',nargs='+',type=int,default=[1,2,3,4])
    args=parser.parse_args(); root=Path(args.output)
    if sorted(set(args.workers))!=args.workers or args.workers[0]!=1:
        raise ValueError('Use distinct increasing positive worker counts starting at one')
    protocol=validate_execution_context(root)
    split=json.loads((root/'splits.json').read_text(encoding='utf-8'))
    sample=[Example(**e) for e in split['splits']['sst_train'][:24]]
    config=dict(model=protocol['model'],workers=args.workers,repetitions=2,
                protocol_hash=fingerprint(protocol),
                sample_ids=[e.id for e in sample],prompt_hash=fingerprint(BASE),
                order='ascending then descending',client_cache=False)
    manifest=root/'latency_protocol.json'
    destination=root/'latency.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8'))!=config:
            raise ValueError('Saved timing protocol differs; use a new output directory')
    elif destination.exists():
        raise ValueError('Existing timing results lack the matching protocol')
    else:
        save_json(manifest,config)
    records=json.loads(destination.read_text(encoding='utf-8')) if destination.exists() else []
    completed={(r['repetition'],r['workers']) for r in records}
    client=Client(root/'requests.sqlite',model=protocol['model'],endpoint=protocol['endpoint'],workers=max(args.workers),max_actual_calls=500)
    for repetition in range(2):
        for workers in (args.workers if repetition==0 else args.workers[::-1]):
            if (repetition,workers) in completed:
                continue
            before=client.snapshot(); start=time.perf_counter()
            def classify(e):
                r=client.complete(task_prompt(BASE,e.text),cache=False)
                return dict(id=e.id,prediction=parse_label(r['text']),seconds=r['seconds'])
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                rows=list(pool.map(classify,sample))
            records.append(dict(repetition=repetition,workers=workers,rows=rows,
                                wall_seconds=time.perf_counter()-start,
                                resources=difference(before,client.snapshot())))
            save_json(destination,records)
            print('LATENCY',workers,'workers',round(records[-1]['wall_seconds'],2),'seconds',flush=True)
    warm_path=root/'warm_cache_latency.json'
    if not warm_path.exists():
        before=client.snapshot(); start=time.perf_counter()
        def cached_classify(e):
            r=client.complete(task_prompt(BASE,e.text),cache=True)
            return dict(id=e.id,prediction=parse_label(r['text']))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.workers)) as pool:
            rows=list(pool.map(cached_classify,sample))
        save_json(warm_path,dict(workers=max(args.workers),rows=rows,
                                wall_seconds=time.perf_counter()-start,
                                resources=difference(before,client.snapshot())))
    print('TIMING COMPLETE',flush=True)


if __name__=='__main__': main()
