"""Descriptive overfitting diagnostics; never feeds final scores into selection."""
import re

import numpy as np


def gap_metrics(validation, baseline_validation, source, baseline_source,
                target, baseline_target):
    validation_gain = validation - baseline_validation
    source_gain = source - baseline_source
    target_gain = target - baseline_target
    return dict(validation_accuracy=validation, source_accuracy=source,
                target_accuracy=target, validation_gain=validation_gain,
                source_gain=source_gain, target_gain=target_gain,
                raw_validation_source_gap=validation-source,
                selection_optimism=validation_gain-source_gain,
                excess_source_gain=source_gain-target_gain)


def copied_spans(instruction, examples, width=8):
    """Flag long exact token matches for review, not proof of memorization."""
    def grams(text):
        words=re.findall(r"\w+",text.lower())
        return {tuple(words[i:i+width]) for i in range(len(words)-width+1)}
    prompt_grams=grams(instruction)
    return [dict(example_id=e['id'],matched_spans=[' '.join(g) for g in sorted(matches)])
            for e in examples if (matches := prompt_grams & grams(e['text']))]


def diagnose_members(members, prediction_rows, base, baseline, sources, targets, split_rows):
    results=[]
    for prompt,run in members:
        if run is None:
            continue
        selected=run['candidates'].index(prompt)
        base_index=run['candidates'].index(base)
        rows=prediction_rows(prompt)
        def accuracy(data,domains):
            return float(np.mean([v['correct'] for v in data.values() if v['domain'] in domains]))
        metrics=gap_metrics(float(np.mean(run['vectors'][selected])),
                            float(np.mean(run['vectors'][base_index])),
                            accuracy(rows,sources),accuracy(baseline,sources),
                            accuracy(rows,targets),accuracy(baseline,targets))
        allowed=set(run['training_ids']+run['validation_ids'])
        results.append(dict(seed=run['seed'],candidate_count=len(run['candidates']),
                            selected_candidate=selected,**metrics,
                            source_text_matches=copied_spans(prompt,[split_rows[i] for i in sorted(allowed)])))
    if not results:
        return None
    return dict(per_seed=results,mean={key:float(np.mean([r[key] for r in results]))
                                     for key in metrics},
                any_source_text_matches=any(r['source_text_matches'] for r in results))


def report_lines(summary, historical_audit=None):
    lines=['**Overfitting checks**','',
           'The original harness gives final OOD examples to the optimizer and later reports scores on those same examples. OPRO uses their labels to rank candidates, so those results are not an untouched transfer test. The offline leakage regression demonstrates that changing only target feedback can change its selected prompt. This is confirmed evaluation leakage; it does not by itself establish how much the previous reported accuracy was inflated.', '',
           'The new experiment freezes every prompt before inspecting final results. Validation gain and final-source gain below are both measured relative to the same base instruction. Selection optimism is validation gain minus final-source gain; subtracting the baseline helps account for an easier validation split. Excess source gain is final-source gain minus OOD gain. Positive values are warning signs, not conclusive evidence from a small sample.', '',
           '| Regime / method | Validation gain, pp | Final-source gain (95% CI), pp | OOD gain, pp | Selection optimism, pp | Excess source gain (95% CI), pp |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in summary:
        if not row.get('overfitting'):
            continue
        d=row['overfitting']['mean']
        lo,hi=row['source_delta_ci95']
        contrast_lo,contrast_hi=row['excess_source_gain_ci95']
        lines.append(f"| {row['regime']} / {row['method']} | {100*d['validation_gain']:+.2f} | {100*d['source_gain']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] | {100*d['target_gain']:+.2f} | {100*d['selection_optimism']:+.2f} | {100*d['excess_source_gain']:+.2f} [{100*contrast_lo:+.2f}, {100*contrast_hi:+.2f}] |")
    lines += ['', 'Final-source gain uses 300 held-out SST-2 examples in the single-source regime, or 600 held-out SST-2/Amazon examples in the multi-source regime. Validation has only 64 examples: one changed answer moves its score by 1.56 percentage points. Optimizer seeds reuse that same split, so three seeds do not create three independent validation datasets.', '',
              'Raw source-to-OOD accuracy gaps can reflect domain difficulty, label conventions and truncation; they are not sufficient evidence of overfitting. The comparison intervals above quantify final-test uncertainty conditional on the selected prompts, not uncertainty over new optimizer seeds or new dataset partitions. Adaptive validation selection also makes ordinary validation confidence intervals misleading.', '']
    matched=[row['method'] for row in summary if (row.get('overfitting') or {}).get('any_source_text_matches')]
    lines.append('Selected instructions have exact eight-token matches to source examples; inspect `overfitting.json` before interpreting them as memorization.' if matched else 'No selected instruction contains an exact eight-token span copied from the allowed source examples. This does not exclude semantic memorization or contamination from model pretraining.')
    lines += ['', 'For future confirmatory runs, keep final target labels inaccessible during search, use an inner source validation set for candidate selection and an outer source holdout for audit, repeat with fresh source splits, and lock the method before evaluating another target domain. Prompt brevity and stable selection are tested regularizers here, not assumed cures.', '']
    if historical_audit:
        lines += ['**Why some previous logs appear to overfit**','',
                  f"All {historical_audit['exact_mock_matches']:,} of {historical_audit['classification_traces']:,} classification responses across {historical_audit['files']} September 21 saved runs exactly replay under the repository's mock backend. That backend explicitly adds a source-only bonus for movie-domain fragments. These logs demonstrate simulated behavior, not empirical overfitting by a real language model. The separate September 15 real-model sweep must not be conflated with them. See `historical_log_audit.json` and `MockBackend` in the original `llm_backends.py`.", '']
        if historical_audit.get('real_sweep'):
            old=historical_audit['real_sweep']
            lines += [f"In the September 15 real Ollama sweep, {old['unchanged_base_prompt']} of {old['runs']} runs returned the unchanged base instruction. Source/OOD gaps for those unchanged prompts cannot be attributed to new rules learned by prompt optimization. The small samples and limited prompt search do not establish optimization-induced overfitting.", '']
    return lines
