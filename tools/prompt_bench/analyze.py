"""Summarize frozen-prompt results with paired, domain/label-stratified intervals."""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

from core import BASE,HUMAN,fingerprint,save_json
from methods import reported_selections
from overfit import diagnose_members, report_lines


def paired_ci(matrix, reference, labels, domains, n_boot=5000):
    # Average the paired per-example difference across optimizer seeds first.
    # This interval captures test-sample uncertainty, conditional on these seeds.
    delta=np.asarray(matrix).mean(axis=0)-reference
    rng=np.random.default_rng(20260929)
    bootstrap=np.zeros(n_boot)
    strata=sorted(set(zip(domains,labels)))
    for domain,label in strata:
        idx=np.array([i for i,(d,l) in enumerate(zip(domains,labels)) if d==domain and l==label])
        bootstrap+=delta[rng.choice(idx,size=(n_boot,len(idx)),replace=True)].mean(axis=1)/len(strata)
    return [float(x) for x in np.quantile(bootstrap,[.025,.975])]


def source_target_contrast_ci(matrix, reference, labels, domains, sources, targets, n_boot=5000):
    """Interval for (source improvement - target improvement), with equal domain weights."""
    delta=np.asarray(matrix).mean(axis=0)-reference
    rng=np.random.default_rng(20260929)
    bootstrap=np.zeros(n_boot)
    strata=sorted(set(zip(domains,labels)))
    for domain,label in strata:
        if domain in sources:
            weight=1/sum(d in sources for d,l in strata)
        elif domain in targets:
            weight=-1/sum(d in targets for d,l in strata)
        else:
            continue
        idx=np.array([i for i,(d,l) in enumerate(zip(domains,labels)) if d==domain and l==label])
        bootstrap+=weight*delta[rng.choice(idx,size=(n_boot,len(idx)),replace=True)].mean(axis=1)
    return [float(x) for x in np.quantile(bootstrap,[.025,.975])]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='output/prompt_benchmark_2026-09-29')
    args=parser.parse_args(); root=Path(args.output)
    predictions={p.stem:json.loads(p.read_text(encoding='utf-8')) for p in (root/'final_predictions').glob('*.json')}
    searches=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(root.glob('*/search.json'))]
    split_manifest=json.loads((root/'splits.json').read_text(encoding='utf-8'))
    split_rows={r['id']:r for rows in split_manifest['splits'].values() for r in rows}
    if fingerprint(BASE) not in predictions: raise ValueError('Missing baseline final predictions')
    def prediction_rows(prompt):
        rows=predictions[fingerprint(prompt)]['predictions']
        return {r['id']:r for r in rows}
    baseline=prediction_rows(BASE)
    cases=sorted(baseline)
    baseline_vector=np.array([baseline[i]['correct'] for i in cases])
    domains=[baseline[i]['domain'] for i in cases]
    labels=[baseline[i]['label'] for i in cases]
    groups={}
    for name,prompt in [('Base instruction',BASE),('Human-written',HUMAN)]:
        groups[('single',name)]=[(prompt,None)]
        groups[('multi',name)]=[(prompt,None)]
    for run in searches:
        method={'opro':'Existing OPRO (corrected evaluation)','gepa':'Official GEPA','diagnose':'ESPO-inspired'}[run['method']]
        if run.get('original_selection'):
            groups.setdefault((run['regime'],method),[]).append((run['original_selection'],run))
        for rule,selection in reported_selections(run).items():
            # Native optimizer selection is primary; mean re-selection is a separate control.
            suffix={'mean':' + mean selection','stable':' + stable selection',
                    'worst_group':' + worst-source-group selection'}[rule]
            groups.setdefault((run['regime'],method+suffix),[]).append((selection['instruction'],run))
    summary=[]
    matrices={}
    for (regime,name),members in groups.items():
        if any(fingerprint(p) not in predictions for p,r in members):
            raise ValueError(f'Incomplete final evaluation: {regime}/{name}')
        matrix=np.array([[prediction_rows(p)[i]['correct'] for i in cases] for p,r in members])
        matrices[(regime,name)]=matrix
        per_domain={d:matrix[:,[i for i,x in enumerate(domains) if x==d]].mean(axis=1).tolist()
                    for d in sorted(set(domains))}
        targets=['amazon','tweets'] if regime=='single' else ['tweets']
        idx=[i for i,d in enumerate(domains) if d in targets]
        domain_means={d:float(np.mean(v)) for d,v in per_domain.items()}
        ood=matrix[:,idx].mean(axis=1)
        interval=paired_ci(matrix[:,idx],baseline_vector[idx],[labels[i] for i in idx],[domains[i] for i in idx])
        sources=['sst2'] if regime=='single' else ['sst2','amazon']
        source_idx=[i for i,d in enumerate(domains) if d in sources]
        source_mean=float(np.mean([domain_means[d] for d in sources]))
        resources=[r['resources'] for p,r in members if r]
        invalid=sum(sum(x['prediction']=='INVALID' for x in prediction_rows(p).values()) for p,r in members)
        row=dict(regime=regime,method=name,optimizer_seeds=[r['seed'] for p,r in members if r],
                 unique_selected_prompts=len(set(p for p,r in members)),domain_accuracy=domain_means,
                 per_seed_domain_accuracy=per_domain,source_accuracy=source_mean,
                 source_delta_vs_base=float((matrix[:,source_idx]-baseline_vector[source_idx]).mean()),
                 source_delta_ci95=paired_ci(matrix[:,source_idx],baseline_vector[source_idx],
                                             [labels[i] for i in source_idx],[domains[i] for i in source_idx]),
                 excess_source_gain_ci95=source_target_contrast_ci(matrix,baseline_vector,labels,domains,sources,targets),
                 ood_accuracy=float(ood.mean()),ood_seed_sd=float(ood.std(ddof=1)) if len(ood)>1 else 0.,
                 ood_delta_vs_base=float((matrix[:,idx]-baseline_vector[idx]).mean()),
                 ood_delta_ci95=interval,ood_gap=source_mean-float(ood.mean()),
                 worst_ood_accuracy=float(np.mean([min(per_domain[d][j] for d in targets) for j in range(len(members))])),
                 mean_instruction_chars=float(np.mean([len(p) for p,r in members])),
                 mean_final_input_tokens_per_example=float(np.mean([
                     x.get('usage',{}).get('prompt_tokens',0) for p,r in members for x in prediction_rows(p).values()])),
                 mean_search_evaluations=float(np.mean([r['unique_example_evaluations'] for r in resources])) if resources else 0,
                 mean_proposal_calls=float(np.mean([r['proposal_calls'] for r in resources])) if resources else 0,
                 mean_search_input_tokens=float(np.mean([r['logical_input_tokens'] for r in resources])) if resources else 0,
                 mean_search_output_tokens=float(np.mean([r['logical_output_tokens'] for r in resources])) if resources else 0,
                 invalid_predictions=invalid,
                 prompt_hashes=[fingerprint(p) for p,r in members])
        # Macro F1 on each domain, averaged over optimizer seeds.
        f1s={}
        for d in sorted(set(domains)):
            seed_values=[]
            for p,r in members:
                rows=[x for x in prediction_rows(p).values() if x['domain']==d]
                fs=[]
                for label in ('Positive','Negative'):
                    tp=sum(x['label']==label and x['prediction']==label for x in rows)
                    fp=sum(x['label']!=label and x['prediction']==label for x in rows)
                    fn=sum(x['label']==label and x['prediction']!=label for x in rows)
                    fs.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
                seed_values.append(float(np.mean(fs)))
            f1s[d]=float(np.mean(seed_values))
        row['domain_macro_f1']=f1s
        row['overfitting']=diagnose_members(members,prediction_rows,BASE,baseline,sources,targets,split_rows)
        summary.append(row)
    save_json(root/'summary.json',summary)
    save_json(root/'overfitting.json',[dict(regime=r['regime'],method=r['method'],
                                     source_gain_ci95=r['source_delta_ci95'],
                                     target_gain_ci95=r['ood_delta_ci95'],
                                     excess_source_gain_ci95=r['excess_source_gain_ci95'],**r['overfitting'])
                                     for r in summary if r['overfitting']])
    comparisons=[]
    for regime,left,right in [
        ('single','Official GEPA','Existing OPRO (corrected evaluation)'),
        ('single','ESPO-inspired + stable selection','Existing OPRO (corrected evaluation)'),
        ('single','ESPO-inspired + stable selection','Official GEPA'),
        ('single','Official GEPA + stable selection','Official GEPA'),
        ('single','ESPO-inspired + stable selection','ESPO-inspired + mean selection'),
        ('multi','Official GEPA + worst-source-group selection','Official GEPA'),
    ]:
        if (regime,left) not in matrices or (regime,right) not in matrices: continue
        target_domains=['amazon','tweets'] if regime=='single' else ['tweets']
        idx=[i for i,d in enumerate(domains) if d in target_domains]
        left_matrix=matrices[(regime,left)][:,idx]
        right_matrix=matrices[(regime,right)][:,idx]
        comparisons.append(dict(regime=regime,method=left,reference=right,
            delta=float(left_matrix.mean()-right_matrix.mean()),
            ci95=paired_ci(left_matrix,right_matrix.mean(axis=0),[labels[i] for i in idx],[domains[i] for i in idx])))
    save_json(root/'comparisons.json',comparisons)
    physical={k:sum(r['physical_resources'][k] for r in searches)+
                    sum(p['physical_resources'][k] for p in predictions.values())
              for k in ['actual_calls','failed_attempts','input_tokens','output_tokens','cache_hits']}
    latency=json.loads((root/'latency.json').read_text(encoding='utf-8')) if (root/'latency.json').exists() else []
    cache_inventory=None
    if (root/'requests.sqlite').exists():
        with sqlite3.connect(root/'requests.sqlite') as connection:
            cached=[json.loads(r[0]) for r in connection.execute('SELECT value FROM responses')]
        cache_inventory=dict(distinct_received_responses=len(cached),
                             input_tokens=sum(r['usage'].get('prompt_tokens',0) for r in cached),
                             output_tokens=sum(r['usage'].get('completion_tokens',0) for r in cached))
        save_json(root/'cache_inventory.json',cache_inventory)
    lines=['**Prompt optimization performance pilot**','',
           'Model: Gemma 4 31B through the existing Ollama cloud connection. Official GEPA package: 0.1.4.',
           'Search used 64 training and 64 validation examples, a cap of 512 distinct candidate/example evaluations and 16 proposal calls per run. Each final domain has 300 balanced examples. Optimization and final evaluation used exact request caching; no semantic prediction reuse was allowed.',
           '', 'Single-source optimization uses only SST-2 source examples. Amazon and tweets are held out. Multi-source optimization uses equal-sized SST-2/Amazon source subsets; only tweets are OOD in that regime.', '',
           'Intervals below are paired 95% bootstrap intervals for the difference versus the fixed base instruction, stratified by target domain and label. They describe test-sample uncertainty conditional on the evaluated optimizer seeds. They are exploratory, not adjusted for multiple method comparisons, and do not establish generalization to other models or datasets.', '']
    for regime,description in [('single','Single-source results: SST-2 to Amazon and tweets'),('multi','Separate multi-source results: SST-2 + Amazon to tweets')]:
        lines += [f'**{description}**','',
                  '| Method | SST-2 | Amazon | Tweets | OOD mean | Worst OOD | Seed SD, pp | Change vs base (95% CI), pp |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in summary:
            if r['regime']!=regime: continue
            d=r['domain_accuracy']; lo,hi=r['ood_delta_ci95']
            lines.append(f"| {r['method']} | {100*d['sst2']:.2f}% | {100*d['amazon']:.2f}% | {100*d['tweets']:.2f}% | {100*r['ood_accuracy']:.2f}% | {100*r['worst_ood_accuracy']:.2f}% | {100*r['ood_seed_sd']:.2f} | {100*r['ood_delta_vs_base']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] |")
        lines += ['']
    lines += ['**Direct comparisons of the implementations**','',
              '| Regime | Method | Reference | OOD difference (95% CI), pp |',
              '| --- | --- | --- | ---: |']
    for comparison in comparisons:
        lo,hi=comparison['ci95']
        lines.append(f"| {comparison['regime']} | {comparison['method']} | {comparison['reference']} | {100*comparison['delta']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ['']
    historical_path=root/'historical_log_audit.json'
    historical_audit=json.loads(historical_path.read_text(encoding='utf-8')) if historical_path.exists() else None
    lines += report_lines(summary,historical_audit)
    lines += ['**Search resources and selected prompt sizes**','',
              '| Regime / method | Evaluations/run | Proposals/run | Search input tokens/run | Search output tokens/run | Instruction characters | Final input tokens/example |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in summary:
        if r['regime']=='multi' and not r['optimizer_seeds']: continue
        lines.append(f"| {r['regime']} / {r['method']} | {r['mean_search_evaluations']:.0f} | {r['mean_proposal_calls']:.1f} | {r['mean_search_input_tokens']:.0f} | {r['mean_search_output_tokens']:.0f} | {r['mean_instruction_chars']:.0f} | {r['mean_final_input_tokens_per_example']:.1f} |")
    lines += ['', 'Search token counts are attributed to each run even when transport caching reused a previously obtained response. Selection variants share the same search and candidate pool; their costs are not additive. Equal evaluation caps do not imply equal token consumption. Input counts include any provider-cached prompt tokens. Account billing rates were not verified, so no dollar saving is claimed.',
              '',f'Physical calls recorded in completed search/final result files: {physical}. Startup probes and any resumed interrupted work are not included in that subtotal.', '']
    if cache_inventory:
        lines += [f'Received-response cache inventory, including interrupted and invalidated searches: {cache_inventory}. This inventory excludes uncached timing calls and cannot account for provider work whose response was lost to a timeout.', '']
    if latency:
        lines += ['**Request concurrency timing with application caching disabled**','',
                  'Each block sends the same 24 source requests. Worker counts run in forward/reverse order for two repetitions. Provider-side prompt caching and service load were not controlled, so these are descriptive timings for this connection rather than a universal speedup estimate.', '']
        by_worker={w:[r['wall_seconds'] for r in latency if r['workers']==w] for w in sorted({r['workers'] for r in latency})}
        for w,seconds in by_worker.items():
            failures=sum(r['resources']['failed_attempts'] for r in latency if r['workers']==w)
            runs=', '.join(f'{x:.2f}' for x in seconds)
            lines.append(f'- {w} worker(s): {np.mean(seconds):.2f} seconds on average for 24 requests (runs: {runs}; {failures} failed attempts).')
        for w in sorted(by_worker):
            if w==1: continue
            speedup=np.mean(by_worker[1])/np.mean(by_worker[w])
            lines.append(f'- {w} workers versus one: {speedup:.2f}x observed wall-time speedup. This tests request concurrency, not GEPA parallel proposal search.')
        first_predictions={r['id']:r['prediction'] for r in latency[0]['rows']}
        comparisons_count=sum(len(block['rows']) for block in latency[1:])
        agreements=sum(r['prediction']==first_predictions[r['id']] for block in latency[1:] for r in block['rows'])
        lines.append(f'- Prediction agreement with the first serial block: {agreements}/{comparisons_count} repeated responses. This checks observed label consistency, not guaranteed service determinism.')
        warm_path=root/'warm_cache_latency.json'
        if warm_path.exists():
            warm=json.loads(warm_path.read_text(encoding='utf-8'))
            lines.append(f"- Warm exact-cache replay of 24 requests: {warm['wall_seconds']:.4f} seconds, {warm['resources']['actual_calls']} model calls, {warm['resources']['cache_hits']} cache hits. This benefit applies to identical repeated requests; cached outputs are not independent model samples.")
        lines += ['']
    lines += ['**Limits and provenance**','',
        '- This is a bounded sentiment pilot on one task model, with three optimization seeds and one frozen dataset partition. Five-seed, larger-sample and cross-model replication remain necessary for broad claims.',
        '- The ESPO-inspired method is a custom component experiment, not the full published ESPO algorithm. It uses source error diagnosis, diverse concise proposals and stable selection.',
        '- Worst-source-group selection is a re-ranking ablation on the same multi-source GEPA candidates, not a full distributionally robust Bayesian optimizer.',
        '- The source SST-2 validation dataset was partitioned into custom disjoint source training, selection and final-test subsets. Amazon was also partitioned for the multi-source regime. Original hidden-label benchmark test sets were not used.',
        '- Binary tweet sentiment excludes neutral labels; Amazon text is truncated at 500 characters to match the existing project loader. Exact normalized-text duplicates were excluded across all splits.',
        '- Same-prompt final predictions are reused across seeds; identical prompts do not count as independent inference replications. Model service revision and perfect decoding determinism cannot be guaranteed by a seed alone.',
        '- All selected prompts were frozen before final evaluation. The optimizer only receives an explicit allowlist of source training and validation examples.',
        '- GEPA uses globally unique, filesystem-safe example IDs. Initial positional-ID cache collisions were caught by independent validation scoring, archived and corrected before any final-test evaluation. Final library scores must match the independent per-example scores.',
        '', 'Reproducibility: `protocol.json`, `splits.json`, each run\'s `search.json`, `frozen_prompts.json`, `final_predictions/`, and `summary.json` accompany this report. The raw response cache is `requests.sqlite`.',
        '', 'Research references: [official GEPA](https://github.com/gepa-ai/gepa), [ESPO](https://arxiv.org/abs/2609.04197), [GEPA parallel proposals](https://gepa-ai.github.io/gepa/blog/2026/07/30/parallel-proposals/).']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote',root/'report.md')
    print(json.dumps([dict(method=r['method'],regime=r['regime'],ood=r['ood_accuracy'],delta=r['ood_delta_vs_base'],ci=r['ood_delta_ci95']) for r in summary],indent=2))


if __name__=='__main__': main()
