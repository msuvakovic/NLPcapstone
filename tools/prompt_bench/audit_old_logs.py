"""Check whether recent saved classification traces match the mock simulator."""
import json
from pathlib import Path

import methods
from core import BASE,save_json
from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.llm_backends import MockBackend, _CLASSIFY_TEXT_RE


def main():
    mock=MockBackend(load_demo_dataset().lookup())
    records=[]
    for p in sorted(Path('prompt_opt_harness/prompt_opt_harness/logs').glob('*20260921*.json')):
        obj=json.loads(p.read_text(encoding='utf-8'))
        traces=[t for t in obj.get('per_call_traces',[]) if _CLASSIFY_TEXT_RE.search(t.get('prompt',''))]
        if not traces:
            continue
        matches=sum(mock._generate(t['prompt'])==t['response'] for t in traces)
        records.append(dict(file=p.as_posix(),classification_traces=len(traces),
                            exact_mock_matches=matches,all_match=matches==len(traces),
                            reported_wall_seconds=obj.get('wall_time')))
    result=dict(scope='September 21 saved individual runs with classification traces',
                caveat='Exact replay is strong evidence of mock provenance, not model performance.',
                files=len(records),all_files_match=all(r['all_match'] for r in records),
                classification_traces=sum(r['classification_traces'] for r in records),
                exact_mock_matches=sum(r['exact_mock_matches'] for r in records),records=records)
    real_path=Path('prompt_opt_harness/logs/compare_real_ollama_multiseed_20260915T203141Z.json')
    if real_path.exists():
        real=json.loads(real_path.read_text(encoding='utf-8'))
        result['real_sweep']=dict(file=real_path.as_posix(),runs=len(real),
                                  unchanged_base_prompt=sum(r['best_instruction']==BASE for r in real),
                                  recorded_backends=sorted({r.get('real_backend','unknown') for r in real}),
                                  interpretation='Gaps for unchanged prompts cannot be attributed to newly learned prompt rules; domain difficulty and sampling remain possible explanations.')
    save_json('output/prompt_benchmark_2026-09-29/historical_log_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__': main()
