"""Reproducible sentiment pilot. Run from the repository root with Python 3.13."""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from core import BASE,HUMAN,Client,Evaluator,difference,fingerprint,save_json
from data import prepare,balanced_half
from methods import METHODS,choose_candidates,reported_selections
from protocol import build_protocol, lock_protocol


def assert_complete_searches(output, searches):
    protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    expected={(regime,method,seed) for regime in protocol['regimes']
              for method in protocol['methods'] for seed in protocol['seeds']
              if regime=='single' or method=='gepa'}
    actual={(r['regime'],r['method'],r['seed']) for r in searches}
    if actual!=expected or len(searches)!=len(actual):
        raise ValueError(f'Search protocol incomplete: missing={expected-actual}, unexpected={actual-expected}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='output/prompt_benchmark_2026-09-29')
    parser.add_argument('--stage',choices=['prepare','search','evaluate','latency'],default='search')
    parser.add_argument('--methods',nargs='+',choices=list(METHODS),default=['opro','gepa','diagnose'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--regimes',nargs='+',choices=['single','multi'],default=['single','multi'])
    parser.add_argument('--train-n',type=int,default=64)
    parser.add_argument('--val-n',type=int,default=64)
    parser.add_argument('--test-n',type=int,default=300)
    parser.add_argument('--budget',type=int,default=512)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--model',default='gemma4:31b-cloud')
    parser.add_argument('--endpoint',default='http://localhost:11434/v1')
    parser.add_argument('--split-seed',type=int,default=2712026)
    args=parser.parse_args()
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    for key in ('methods','regimes','seeds'):
        if len(getattr(args,key))!=len(set(getattr(args,key))):
            raise ValueError(f'Duplicate {key} in experiment configuration')
    if args.budget<=0 or args.workers<=0:
        raise ValueError('Budget and workers must be positive')
    if (output/'protocol.json').exists() and not (output/'splits.json').exists():
        raise ValueError('Frozen experiment is missing its split manifest; do not regenerate it')
    splits=prepare(output,args.train_n,args.val_n,args.test_n,args.split_seed)
    config=build_protocol(args,json.loads((output/'splits.json').read_text(encoding='utf-8')))
    protocol_hash=lock_protocol(output,config,args.stage)
    if args.stage=='prepare':
        print({k:len(v) for k,v in splits.items()},flush=True)
        return
    client=Client(output/'requests.sqlite',model=args.model,workers=args.workers,endpoint=args.endpoint,
                  max_actual_calls=config['max_actual_calls_per_session'])
    if args.stage=='search':
        for regime in args.regimes:
            if regime=='single':
                train,val=splits['sst_train'],splits['sst_val']
            else:
                train=balanced_half(splits['sst_train'])+balanced_half(splits['amazon_train'])
                val=balanced_half(splits['sst_val'])+balanced_half(splits['amazon_val'])
            for method in args.methods:
                # Multi-source tests use GEPA's identical candidate pool for both selectors.
                if regime=='multi' and method!='gepa': continue
                for seed in args.seeds:
                    directory=output/f'{regime}_{method}_{seed}'
                    directory.mkdir(exist_ok=True)
                    result_path=directory/'search.json'
                    if result_path.exists():
                        print('RESUME: already completed',directory.name,flush=True); continue
                    ev=Evaluator(client,train+val,budget=args.budget,seed=seed)
                    before=client.snapshot(); start=time.perf_counter()
                    print('START SEARCH',directory.name,flush=True)
                    result=METHODS[method](ev,train,val,seed,directory)
                    result['selections']={rule:choose_candidates(result['candidates'],result['vectors'],val,seed,rule)
                                          for rule in ['mean','stable']+(['worst_group'] if regime=='multi' else [])}
                    result.update(regime=regime,method=method,seed=seed,protocol_hash=protocol_hash,
                                  workers=args.workers,
                                  resources=ev.summary(),physical_resources=difference(before,client.snapshot()),
                                  wall_seconds=time.perf_counter()-start,
                                  training_ids=[e.id for e in train],validation_ids=[e.id for e in val],
                                  evaluation_audit=ev.audit)
                    save_json(result_path,result)
                    print('DONE SEARCH',directory.name,'candidates',len(result['candidates']),
                          'val',result['selections']['mean']['validation_accuracy'],
                          'evals',ev.eval_count,'seconds',round(result['wall_seconds'],1),flush=True)
    elif args.stage=='evaluate':
        searches=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(output.glob('*/search.json'))]
        if not searches: raise ValueError('No frozen search results to evaluate')
        assert_complete_searches(output, searches)
        searches.sort(key=lambda r:(r['regime']!='single',{'opro':0,'gepa':1,'diagnose':2}[r['method']],r['seed']))
        prompts={fingerprint(BASE):BASE,fingerprint(HUMAN):HUMAN}
        for run in searches:
            for sel in reported_selections(run).values(): prompts[fingerprint(sel['instruction'])]=sel['instruction']
            if run.get('original_selection'):
                prompts[fingerprint(run['original_selection'])]=run['original_selection']
        # Freeze all selected prompts before any final scores are computed.
        frozen_path=output/'frozen_prompts.json'
        if frozen_path.exists():
            if json.loads(frozen_path.read_text(encoding='utf-8'))!=prompts:
                raise ValueError('Selections changed after final-prompt freeze; use a new experiment directory')
        else:
            save_json(frozen_path,prompts)
            save_json(output/'source_revision_at_freeze.json',{
                p.name:fingerprint(p.read_text(encoding='utf-8'))
                for p in Path(__file__).parent.glob('*.py')})
        tests=[e for k in ('sst_test','amazon_test','tweets_test') for e in splits[k]]
        for number,(pid,prompt) in enumerate(prompts.items(),1):
            destination=output/'final_predictions'/f'{pid}.json'
            if destination.exists(): continue
            start=time.perf_counter(); before=client.snapshot()
            print('START FINAL',number,'/',len(prompts),'prompt',pid[:12],flush=True)
            ev=Evaluator(client,tests,budget=len(tests))
            # Chunking checkpoints progress and makes the long evaluation visible.
            predictions=[]
            for i in range(0,len(tests),100):
                batch=tests[i:i+100]
                predictions.extend(dict(**asdict(e),**r) for e,r in zip(batch,ev.evaluate(prompt,batch)))
                print('FINAL PROGRESS',pid[:12],min(i+100,len(tests)),'/',len(tests),flush=True)
            save_json(destination,dict(instruction=prompt,prompt_id=pid,predictions=predictions,protocol_hash=protocol_hash,
                                       wall_seconds=time.perf_counter()-start,
                                       physical_resources=difference(before,client.snapshot())))
            print('DONE FINAL',pid[:12],flush=True)
    else:
        import concurrent.futures
        from core import task_prompt,parse_label
        # Separate from accuracy tests; no final data used to tune concurrency.
        sample=splits['sst_train'][:24]
        records=[]
        for repetition in range(2):
            for workers in ([1,args.workers] if repetition==0 else [args.workers,1]):
                before=client.snapshot(); start=time.perf_counter()
                def classify(e):
                    r=client.complete(task_prompt(BASE,e.text),cache=False)
                    return dict(id=e.id,prediction=parse_label(r['text']),seconds=r['seconds'])
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    rows=list(pool.map(classify,sample))
                records.append(dict(repetition=repetition,workers=workers,rows=rows,
                                    wall_seconds=time.perf_counter()-start,
                                    resources=difference(before,client.snapshot())))
                save_json(output/'latency.json',records)
                print('LATENCY',workers,'workers',round(records[-1]['wall_seconds'],2),'seconds',flush=True)
        before=client.snapshot(); start=time.perf_counter()
        def cached_classify(e):
            r=client.complete(task_prompt(BASE,e.text),cache=True)
            return dict(id=e.id,prediction=parse_label(r['text']))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows=list(pool.map(cached_classify,sample))
        save_json(output/'warm_cache_latency.json',dict(workers=args.workers,rows=rows,
                    wall_seconds=time.perf_counter()-start,resources=difference(before,client.snapshot())))
        print('WARM CACHE',round(time.perf_counter()-start,4),'seconds',flush=True)
    print('SESSION RESOURCES',client.snapshot(),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
