"""Offline regression for the repaired OPRO target-feedback boundary.

These artificial scores test selection dependencies, not model performance.
"""
import json
import random
from pathlib import Path
from types import SimpleNamespace

from core import save_json
import methods  # Register the existing package import path.
from prompt_opt_harness.optimizers.opro import OPRO


def reproduce():
    records=[]
    for source_only in (False,True):
        for target_favors in ('Base instruction','Revised instruction'):
            class Backend:
                def __init__(self): self.stats=SimpleNamespace(calls=0)
                def generate(self,prompt):
                    self.stats.calls+=1
                    return 'Revised instruction'
            class ControlledOPRO(OPRO):
                def _evaluation_calls(self, examples):
                    return 1  # This artificial scorer increments once per batch.
                def _dev_score(self,instruction,examples):
                    self.backend.stats.calls+=1
                    return .5 if examples=='source' else float(instruction==target_favors)
            optimizer=ControlledOPRO(Backend(),budget=5,seed=0)
            random.seed(0)
            if not source_only:
                try:
                    optimizer.optimize('Base instruction','source',ood_examples={'target':'target'})
                except ValueError:
                    assert optimizer.backend.stats.calls == 0
                    records.append(dict(source_only=False,target_favors=target_favors,rejected=True))
                else:
                    raise AssertionError('OPRO accepted target feedback')
            else:
                best=optimizer.optimize('Base instruction','source')
                records.append(dict(source_only=True,target_favors=target_favors,
                                    selected=best.instruction,source_score=best.dev_score))
    assert records[0]['rejected'] and records[1]['rejected']
    assert records[2]['selected']==records[3]['selected']=='Base instruction'
    return dict(result='PASS',kind='Offline control-flow regression; artificial scores',
                conclusion='Target feedback is rejected before calls; source-only selection is invariant.',
                records=records)


if __name__=='__main__':
    result=reproduce()
    save_json(Path('output/leakage_fix_2026-10-04/direct_api_regression.json'),result)
    print(json.dumps(result,indent=2))
