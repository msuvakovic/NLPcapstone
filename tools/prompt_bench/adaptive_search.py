"""Budget-matched history/feedback/racing ablations, not a reference OPRO reproduction."""
import json
import math
import random

from core import BASE, fingerprint, strip_fence

VARIANTS = ('history_full', 'feedback_full', 'feedback_racing')


def balanced_order(examples, seed):
    """Identical nested, label-balanced prefixes for all candidates in a run."""
    rng = random.Random(seed)
    groups = [[e for e in sorted(examples, key=lambda e: e.id) if e.label == label]
              for label in ('Positive', 'Negative')]
    if len(groups[0]) != len(groups[1]) or not groups[0]:
        raise ValueError('Balanced nonempty binary source data required')
    for group in groups:
        rng.shuffle(group)
    return [e for pair in zip(*groups) for e in pair]


def feedback_examples(examples, rows, seed):
    rng = random.Random(seed)
    indexed = list(zip(examples, rows))
    rng.shuffle(indexed)
    failures = [(e, r) for e, r in indexed if not r['correct']][:4]
    successes = [(e, r) for e, r in indexed if r['correct']][:4]
    return [dict(text=e.text, expected=e.label, predicted=r['prediction'])
            for e, r in failures + successes]


def proposal_prompt(history, examples, attempt):
    strategies = (
        'Repair a recurring semantic mistake using a general decision rule.',
        'Simplify the strongest instruction; delete redundant or overly specific rules.',
        'Consider an alternative way to resolve mixed or implicit sentiment.',
        'Write a concise alternative without domain-specific vocabulary.',
    )
    top = sorted(history, key=lambda row: (-row['score'], len(row['instruction'])))[:5]
    return ('Improve an instruction for binary sentiment classification. The classifier returns '
            'exactly Positive or Negative. Produce a standalone instruction of at most 80 words. '
            'Do not embed examples or quote their text. Return ONLY the instruction in a fenced code block.\n'
            + strategies[attempt % len(strategies)] + '\nPrevious instructions and accuracy on the SAME '
            'source subset:\n' + json.dumps(top, ensure_ascii=False)
            + ('\nSource TRAINING examples (including failures and successes):\n'
               + json.dumps(examples, ensure_ascii=False) if examples else ''))


def run_search(ev, train, validation, seed, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown search variant')
    if ev.budget != 512 or len(train) != 64 or len(validation) != 64:
        raise ValueError('This pilot schedule is frozen at 64/64 examples and 512 evaluations')
    if {e.id for e in train} & {e.id for e in validation}:
        raise ValueError('Search and selection sets overlap')
    if any(e.split != 'train' or e.domain != 'sst2' for e in train) or any(
            e.split != 'validation' or e.domain != 'sst2' for e in validation):
        raise ValueError('This source-only pilot accepts SST training/selection examples only')
    ordered = balanced_order(train, seed)
    # Every arm pays for the same diagnostics. Only feedback arms expose them to
    # the proposer, isolating feedback from an unequal evaluation budget.
    base_train = ev.evaluate(BASE, ordered)
    feedback = feedback_examples(ordered, base_train, seed) if variant != 'history_full' else []
    racing = variant == 'feedback_racing'
    initial_set = ordered[:16] if racing else validation
    initial_rows = ev.evaluate(BASE, initial_set)
    history = [dict(instruction=BASE, score=sum(r['correct'] for r in initial_rows)/len(initial_rows))]
    candidates, rejected = [BASE], []
    max_new = 12 if racing else 6
    for attempt in range(ev.proposal_limit):
        if len(candidates) - 1 >= max_new:
            break
        candidate = strip_fence(ev.propose(proposal_prompt(history, feedback, attempt)))
        if not candidate or candidate in candidates or len(candidate.split()) > 80:
            rejected.append(dict(attempt=attempt, prompt_hash=fingerprint(candidate),
                                 reason='empty_duplicate_or_over_80_words'))
            continue
        rows = ev.evaluate(candidate, initial_set)
        candidates.append(candidate)
        history.append(dict(instruction=candidate, score=sum(r['correct'] for r in rows)/len(rows)))
    stages = [dict(size=len(initial_set), ids=[e.id for e in initial_set], scores=history)]
    if racing:
        # 64 diagnostics + 12*16 screening + 4*16 extra + 3*64 validation = 512.
        survivors = sorted(history[1:], key=lambda r: (-r['score'], len(r['instruction'])))[:4]
        expanded = []
        for entry in survivors:
            rows = ev.evaluate(entry['instruction'], ordered[:32])
            expanded.append(dict(instruction=entry['instruction'], score=sum(r['correct'] for r in rows)/32))
        stages.append(dict(size=32, ids=[e.id for e in ordered[:32]], scores=expanded))
        finalists = [BASE] + [r['instruction'] for r in sorted(
            expanded, key=lambda r: (-r['score'], len(r['instruction'])))[:2]]
    else:
        finalists = candidates
    vectors = [[r['correct'] for r in ev.evaluate(c, validation)] for c in finalists]
    # Favor the base on exact ties, then brevity. Confirmation is a separate phase.
    selected = max(range(len(finalists)), key=lambda i: (sum(vectors[i]), finalists[i] == BASE,
                                                        -len(finalists[i]), -i))
    return dict(variant=variant, candidates=candidates, finalists=finalists, vectors=vectors,
                selected=finalists[selected], validation_accuracy=sum(vectors[selected])/len(validation),
                stages=stages, rejected=rejected, feedback_example_count=len(feedback),
                resources=ev.summary(), evaluation_audit=ev.audit)


def confirmation_gate(base, challenger, comparisons, alpha=0.05, minimum_gain=0.01):
    """One-sided exact paired test, with Bonferroni across prespecified finalists."""
    if len(base) != len(challenger) or not base or comparisons < 1:
        raise ValueError('Paired confirmation vectors and a positive comparison count required')
    if any(x not in (0, 1) for x in list(base) + list(challenger)):
        raise ValueError('Confirmation vectors must be binary')
    wins = sum(a == 0 and b == 1 for a, b in zip(base, challenger))
    losses = sum(a == 1 and b == 0 for a, b in zip(base, challenger))
    n = wins + losses
    p = sum(math.comb(n, k) for k in range(wins, n+1)) / 2**n if n else 1.0
    gain = (wins-losses)/len(base)
    threshold = alpha/comparisons
    return dict(accepted=gain >= minimum_gain and p <= threshold, gain=gain,
                corrected= wins, regressed=losses, p_value=p, alpha=threshold,
                minimum_gain=minimum_gain, n=len(base))
