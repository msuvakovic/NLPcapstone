"""Offline report integration test. Synthetic fixtures never enter benchmark output."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from core import BASE,HUMAN,fingerprint,save_json
from analyze import source_target_contrast_ci


class ReportTests(unittest.TestCase):
    def test_source_target_contrast_normalizes_domain_counts(self):
        interval=source_target_contrast_ci([[1,0,0,0,0,0]],[0]*6,
                    ['Positive','Negative']*3,['source']*2+['a']*2+['b']*2,
                    ['source'],['a','b'],n_boot=100)
        self.assertEqual(interval,[.5,.5])

    def test_report_and_overfit_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            tests=[dict(id=f'{domain}:test:{i}',text=f'{domain} item {i}',
                        label='Positive' if i==0 else 'Negative',domain=domain,split='test')
                   for domain in ('sst2','amazon','tweets') for i in range(2)]
            validation=[dict(id=f'sst2:val:{i}',text=f'validation {i}',label=e['label'],domain='sst2',split='val')
                        for i,e in enumerate(tests[:2])]
            save_json(root/'splits.json',dict(splits=dict(test=tests,val=validation)))
            resources=dict(unique_example_evaluations=4,proposal_calls=1,
                           logical_input_tokens=40,logical_output_tokens=4)
            physical=dict(actual_calls=0,failed_attempts=0,input_tokens=0,output_tokens=0,cache_hits=0)
            improved='A synthetic improved instruction.'
            for prompt in (BASE,HUMAN,improved):
                rows=[dict(**e,prediction=e['label'] if prompt!=BASE or i%2==0 else 'INVALID',
                           correct=int(prompt!=BASE or i%2==0)) for i,e in enumerate(tests)]
                save_json(root/'final_predictions'/f'{fingerprint(prompt)}.json',
                          dict(instruction=prompt,predictions=rows,physical_resources=physical))
            for regime in ('single','multi'):
                for method in ('opro','gepa','diagnose'):
                    if regime=='multi' and method!='gepa': continue
                    for seed in range(3):
                        selection=dict(instruction=improved,index=1,validation_accuracy=1.)
                        run=dict(regime=regime,method=method,seed=seed,candidates=[BASE,improved],
                                 vectors=[[1,0],[1,1]],resources=resources,physical_resources=physical,
                                 selections={r:selection for r in ('mean','stable','worst_group')},
                                 original_selection=improved if method!='diagnose' else None,
                                 training_ids=[],validation_ids=[e['id'] for e in validation])
                        save_json(root/f'{regime}_{method}_{seed}'/'search.json',run)
            script=Path(__file__).with_name('analyze.py')
            result=subprocess.run([sys.executable,'-B',str(script),'--output',str(root)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            rows=json.loads((root/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(len(rows),11)
            for row in rows:
                if row['overfitting']:
                    self.assertAlmostEqual(row['overfitting']['mean']['selection_optimism'],0.)
                    self.assertAlmostEqual(row['ood_delta_vs_base'],.5)
            self.assertIn('Overfitting checks',(root/'report.md').read_text(encoding='utf-8'))
            self.assertTrue((root/'comparisons.json').exists())


if __name__=='__main__': unittest.main()
