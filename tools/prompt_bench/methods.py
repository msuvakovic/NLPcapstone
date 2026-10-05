"""Official GEPA, existing source-only OPRO, and explicitly named adaptations."""
import json
import random
import sys
from pathlib import Path

import numpy as np

from core import BASE, BudgetExceeded, fingerprint, strip_fence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'.benchmark_deps'))
sys.path.insert(0, str(ROOT/'prompt_opt_harness'/'prompt_opt_harness'))


class KeyedLoader:
    """Globally unique IDs prevent train/validation cache-key collisions."""
    def __init__(self, examples):
        # GEPA also places IDs in file names; hashes are portable on Windows.
        self.examples={fingerprint(e.id):e for e in examples}
        if len(self.examples)!=len(examples):
            raise ValueError('Duplicate data IDs')
    def all_ids(self):
        return list(self.examples)
    def fetch(self, ids):
        return [self.examples[i] for i in ids]
    def __len__(self):
        return len(self.examples)


class Utf8Logger:
    """Keep complete UTF-8 logs and concise, console-safe progress messages."""
    def __init__(self, path):
        self.path=path
        path.parent.mkdir(parents=True,exist_ok=True)
    def log(self,message):
        with self.path.open('a',encoding='utf-8') as stream:
            stream.write(str(message)+'\n')
        if not any(part in message for part in ('Individual valset scores',
                'New valset pareto front scores','Updated valset pareto front programs')):
            line=str(message).splitlines()[0][:240]
            print(line.encode('ascii',errors='backslashreplace').decode('ascii'),flush=True)


def vector(ev, instruction, val):
    return [r['correct'] for r in ev.evaluate(instruction, val)]


def choose_candidates(candidates, vectors, val, seed, rule='mean'):
    matrix = np.asarray(vectors, dtype=float)
    mean = matrix.mean(axis=1)
    if rule == 'mean':
        scores = mean
    elif rule == 'stable':
        rng = np.random.default_rng(seed + 991)
        # Resample the SAME column indices for all candidates; fractional ties.
        weights = rng.multinomial(len(val), np.ones(len(val))/len(val), size=1000)
        boot = matrix @ weights.T
        wins = boot == boot.max(axis=0)
        scores = (wins / wins.sum(axis=0)).mean(axis=1)
    elif rule == 'worst_group':
        groups = sorted({e.domain for e in val})
        if len(groups) < 2:
            raise ValueError('Worst-group selection needs multiple actual source domains')
        group_scores = np.array([matrix[:,[i for i,e in enumerate(val) if e.domain==g]].mean(axis=1)
                                 for g in groups])
        scores = .5*group_scores.mean(axis=0) + .5*group_scores.min(axis=0)
    else:
        raise ValueError(rule)
    # Raw accuracy remains a tie-breaker, then shorter instruction, then fixed index.
    index = max(range(len(candidates)), key=lambda i:(round(float(scores[i]),12),
                                                     float(mean[i]),-len(candidates[i]),-i))
    return dict(instruction=candidates[index], index=index, selection_rule=rule,
                selection_score=float(scores[index]), validation_accuracy=float(mean[index]))


def run_opro(ev, train, val, seed, directory):
    from prompt_opt_harness.optimizers.opro import OPRO
    class Stats:
        @property
        def calls(self):
            return ev.eval_count + ev.proposal_count
    class Backend:
        stats = Stats()
        def generate(self, prompt):
            return ev.propose(prompt)
    class ExactOPRO(OPRO):
        def _evaluation_calls(self, dev_examples):
            # This protocol caps distinct evaluations and proposals separately.
            # Evaluator reserves whole batches, including cache hits, itself.
            return 0
        def _dev_score(self, instruction, examples):
            return float(np.mean(vector(ev,instruction,examples)))
    optimizer = ExactOPRO(Backend(), budget=ev.budget, seed=seed)
    random.seed(seed)
    try:
        best = optimizer.optimize(BASE,val,ood_examples=None)
        reason = 'completed'
    except BudgetExceeded:
        best = max(optimizer.history,key=lambda c:c.dev_score)
        reason = 'next_batch_exceeds_cap'
    candidates = list(dict.fromkeys(c.instruction for c in optimizer.history))
    return dict(candidates=candidates, vectors=[vector(ev,c,val) for c in candidates],
                original_selection=best.instruction, stop_reason=reason,
                implementation='Repository OPRO, source-only inputs, exact evaluator, hard cap')


def run_gepa(ev, train, val, seed, directory):
    import gepa
    from gepa.core.adapter import EvaluationBatch
    # Replay interrupted searches from the exact response cache with fresh logical
    # accounting; loading only GEPA's checkpoint would lose our evaluator budget.
    run_directory=directory/'gepa'
    attempt=0
    while run_directory.exists():
        attempt+=1
        run_directory=directory/f'gepa_replay_{attempt}'
    class Adapter:
        # Required explicitly by gepa 0.1.4's reflective proposer.
        propose_new_texts = None
        def evaluate(self, batch, candidate, capture_traces=False):
            before = ev.eval_count
            rows = ev.evaluate(candidate['instruction'],batch)
            return EvaluationBatch(outputs=[r['prediction'] for r in rows],
                    scores=[float(r['correct']) for r in rows],
                    trajectories=[dict(text=e.text,expected=e.label,predicted=r['prediction'])
                                  for e,r in zip(batch,rows)] if capture_traces else None,
                    num_metric_calls=ev.eval_count-before)
        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            return {key:[{'Inputs':t['text'],'Generated Outputs':t['predicted'],
                          'Feedback':f"Expected {t['expected']}; correctness={score}."}
                         for t,score in zip(eval_batch.trajectories,eval_batch.scores)]
                    for key in components_to_update}
    def reflect(prompt):
        return ev.propose(prompt, max_tokens=2048)
    if {e.id for e in train} & {e.id for e in val}:
        raise ValueError('Training and validation must be disjoint')
    result = gepa.optimize(seed_candidate={'instruction':BASE}, trainset=KeyedLoader(train),valset=KeyedLoader(val),
            adapter=Adapter(),reflection_lm=reflect,reflection_minibatch_size=8,
            candidate_selection_strategy='pareto',use_merge=True,max_merge_invocations=3,
            max_metric_calls=ev.budget,cache_evaluation=True,skip_perfect_score=False,
            stop_callbacks=lambda state: ev.remaining < len(val)+16 or ev.proposal_count>=ev.proposal_limit or state.i>=64,
            run_dir=str(run_directory),seed=seed,raise_on_exception=True,
            logger=Utf8Logger(run_directory/'run_log.txt'),
            display_progress_bar=False)
    candidates = list(dict.fromkeys(c['instruction'] for c in result.candidates))
    vectors=[vector(ev,c,val) for c in candidates]
    # Independently verify every library validation score against actual examples.
    for candidate, scores in zip(result.candidates,result.val_subscores):
        actual=dict(zip((fingerprint(e.id) for e in val),vectors[candidates.index(candidate['instruction'])]))
        if actual!=scores:
            raise RuntimeError('GEPA validation cache differs from independent evaluation')
    return dict(candidates=candidates,vectors=vectors,
                original_selection=result.best_candidate['instruction'],
                official_result=result.to_dict(),stop_reason='library_stop',
                implementation='Official gepa 0.1.4, Pareto selection, merge enabled, default reflection template')


def run_diagnose(ev, train, val, seed, directory):
    candidates = [BASE]
    vectors = [vector(ev,BASE,val)]
    rows = ev.evaluate(BASE,train)
    errors = [dict(text=e.text,expected=e.label,predicted=r['prediction'])
              for e,r in zip(train,rows) if not r['correct']]
    if not errors:
        return dict(candidates=candidates,vectors=vectors,stop_reason='no_training_errors',
                    implementation='ESPO-inspired adaptation')
    diagnosis = ev.propose('Analyze these binary sentiment classification errors. Group them into recurring '
        'semantic failure patterns. Explain each pattern and propose general rules. Do not quote or memorize '
        'individual examples. Keep the diagnosis under 250 words.\n'+json.dumps(errors,ensure_ascii=False))
    strategies = [
        'Revise the instruction to address the diagnosed root causes.',
        'Write a concise instruction consolidating the most useful rules, with no redundant caveats.',
        'Remove or soften assumptions that cause false positives or false negatives.',
        'Write domain-independent decision rules that also work outside movie reviews.',
        'Write an alternative instruction that carefully handles conflicting positive and negative evidence.',
        'Simplify the decision procedure while retaining the general lessons from the diagnosis.',
    ]
    random.Random(seed).shuffle(strategies)
    for strategy in strategies:
        if ev.remaining < len(val) or ev.proposal_count >= ev.proposal_limit:
            break
        prompt = ('Write a replacement instruction for binary sentiment classification. '+strategy+
            '\nUse at most 160 words. Output ONLY the instruction in a fenced code block. '
            'The classifier must return Positive or Negative. Do not embed training examples.\n'
            f'Current instruction:\n{BASE}\nDiagnosis:\n{diagnosis}')
        instruction = strip_fence(ev.propose(prompt))
        if instruction and instruction not in candidates:
            candidates.append(instruction)
            vectors.append(vector(ev,instruction,val))
    return dict(candidates=candidates,vectors=vectors,diagnosis=diagnosis,
                stop_reason='proposal_set_or_budget_exhausted',
                implementation='ESPO-inspired: source error diagnosis, varied concise edits, cached bootstrap selection; not full ESPO reproduction')


METHODS = {'opro':run_opro,'gepa':run_gepa,'diagnose':run_diagnose}


def reported_selections(run):
    """Prespecified component comparisons, excluding unrelated selector variants."""
    if run['method']=='opro':
        rules=[]  # Preserve the repository optimizer's native selection.
    elif run['regime']=='multi':
        rules=['worst_group']
    elif run['method']=='gepa':
        rules=['stable']
    else:
        rules=['mean','stable']
    return {rule:run['selections'][rule] for rule in rules}
