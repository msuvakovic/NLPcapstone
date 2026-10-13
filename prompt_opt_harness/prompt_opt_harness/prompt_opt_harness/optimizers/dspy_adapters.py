"""Optional DSPy instruction optimizers with the harness's exact inference path.

Use DSPyGEPA, DSPyMIPROv2 or DSPyCOPRO explicitly. No TextGrad alias: COPRO
is a different algorithm. DSPy is imported only when optimization is requested.
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
import random
import threading
from types import SimpleNamespace

from ..budget import BudgetedBackend, CallBudgetExceeded
from ..llm_backends import LLMBackend
from ..prompts import build_classification_prompt
from ..tasks import SENTIMENT
from .base import Candidate, Optimizer, parse_label


DSPY_VERSION = '3.2.1'


def _load_dspy():
    try:
        import dspy
    except ImportError as exc:
        raise ImportError('Install the optional requirements-dspy.txt in a separate environment') from exc
    if importlib.metadata.version('dspy') != DSPY_VERSION:
        raise RuntimeError(f'This adapter is tested against dspy=={DSPY_VERSION}; use requirements-dspy.txt')
    return dspy


def split_source(examples, seed, task):
    """Stable, stratified source-only split; reject duplicate IDs/text and bad labels."""
    groups = {label.casefold(): [] for label in task.labels}
    ids, texts = set(), set()
    for ex in examples:
        text = ' '.join(ex.text.casefold().split())
        if ex.id in ids or text in texts:
            raise ValueError('Duplicate source ID/text in DSPy input')
        ids.add(ex.id)
        texts.add(text)
        label = ex.label.strip().casefold()
        if label not in groups:
            raise ValueError(f'Label outside task {task.name}: {ex.label}')
        groups[label].append(ex)
    train, val = [], []
    rng = random.Random(seed)
    for label in sorted(groups):
        rows = sorted(groups[label], key=lambda ex: ex.id)
        if len(rows) < 2:
            raise ValueError('DSPy needs at least two source examples per task label for disjoint train/validation')
        rng.shuffle(rows)
        split = max(1, len(rows) // 2)
        train.extend(rows[:split])
        val.extend(rows[split:])
    return train, val


def _make_bridge(dspy, backend, *, temperature, max_tokens, state):
    """All LM copies share one serialized budget/counter; no provider client or cache."""
    class HarnessLM(dspy.BaseLM):
        def __init__(self):
            super().__init__(model=getattr(backend, 'model', 'harness'),
                             temperature=temperature, max_tokens=max_tokens, cache=False)

        def copy(self, **kwargs):
            result = copy.copy(self)
            result.kwargs = dict(self.kwargs, **kwargs)
            result.history = []
            return result

        def __deepcopy__(self, memo):
            result = self.copy()
            memo[id(self)] = result
            return result

        def forward(self, prompt=None, messages=None, **kwargs):
            options = dict(self.kwargs, **kwargs)
            if messages is not None:
                prompt = 'Complete the following conversation. Return only the assistant reply.\n' + json.dumps(
                    messages, ensure_ascii=False)
            if not isinstance(prompt, str):
                raise TypeError('DSPy bridge requires a text prompt')
            n = options.get('n', 1)
            if not isinstance(n, int) or isinstance(n, bool) or n < 1:
                raise ValueError('DSPy generation count must be a positive integer')
            choices = []
            with state.lock:
                try:
                    backend.require_calls(n)
                    for _ in range(n):
                        response = backend.generate_configured(
                            prompt, temperature=options.get('temperature'),
                            max_tokens=options.get('max_tokens'))
                        choices.append(SimpleNamespace(message=SimpleNamespace(content=response)))
                except CallBudgetExceeded:
                    state.budget_exhausted = True
                    raise
            return SimpleNamespace(choices=choices, model=self.model, usage={})

    return HarnessLM()


def _make_task_adapter(dspy, backend, task, allowed_texts, state):
    class ExactTaskAdapter(dspy.ChatAdapter):
        def __init__(self):
            super().__init__(use_json_adapter_fallback=False)

        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            if (set(signature.input_fields) != {'harness_text'} or
                    set(signature.output_fields) != {'harness_label'}):
                return super().__call__(lm, lm_kwargs, signature, demos, inputs)
            if demos:
                raise ValueError('Instruction-only DSPy adapter rejects few-shot demonstrations')
            text = inputs['harness_text']
            if text not in allowed_texts:
                raise ValueError('DSPy task evaluation must use the supplied source examples only')
            # Exact final-evaluation formatting/settings; DSPy field prefixes
            # and proposal decoding settings never reach task inference.
            instruction = signature.instructions
            prompt = build_classification_prompt(instruction, text, task)
            with state.lock:
                state.instructions.setdefault(instruction, None)
                try:
                    raw = backend.generate(prompt)
                except CallBudgetExceeded:
                    state.budget_exhausted = True
                    raise
                if text in getattr(state, 'validation_labels', {}):
                    label = parse_label(raw, task)
                    state.validation_scores.setdefault(instruction, {})[text] = float(
                        label != 'INVALID' and label == state.validation_labels[text])
            return [{'harness_label': raw}]

    return ExactTaskAdapter()


class DSPyOptimizerAdapter(Optimizer):
    """Compile source-only, then independently score the exported instruction.

    budget caps all search attempts: retries, proposals, baseline scoring and
    exported-instruction validation. Final source/target tests are separate.
    """
    name = 'dspy-optimizer'
    method = None

    def __init__(self, backend, budget, seed=0, task=SENTIMENT, *,
                 proposal_temperature=0.7, proposal_max_tokens=2048):
        super().__init__(backend, budget, seed, task)
        if not isinstance(backend, LLMBackend):
            raise TypeError('DSPy requires an LLMBackend with attempt accounting and retry limits')
        BudgetedBackend(backend, budget)
        if not isinstance(proposal_max_tokens, int) or isinstance(proposal_max_tokens, bool) or proposal_max_tokens < 1:
            raise ValueError('proposal_max_tokens must be a positive integer')
        self.proposal_temperature = proposal_temperature
        self.proposal_max_tokens = proposal_max_tokens
        self.metadata = {}

    def _teleprompter(self, dspy, metric, lm, remaining):
        if self.method == 'GEPA':
            return dspy.GEPA(metric=metric, max_metric_calls=max(1, remaining),
                             reflection_lm=lm, seed=self.seed, num_threads=1,
                             track_stats=True)
        if self.method == 'MIPROv2':
            return dspy.MIPROv2(metric=metric, auto='light', seed=self.seed,
                                max_bootstrapped_demos=0, max_labeled_demos=0,
                                num_threads=1, max_errors=1,
                                init_temperature=self.proposal_temperature)
        if self.method == 'COPRO':
            return dspy.COPRO(metric=metric, prompt_model=lm, breadth=2, depth=2,
                              init_temperature=self.proposal_temperature)
        raise ValueError('Choose DSPyGEPA, DSPyMIPROv2 or DSPyCOPRO')

    def _compile(self, dspy, student, metric, lm, trainset, valset, remaining):
        optimizer = self._teleprompter(dspy, metric, lm, remaining)
        if self.method == 'COPRO':
            # COPRO has one optimization/evaluation set. Keep external source
            # validation independent of its search.
            return optimizer.compile(student, trainset=trainset,
                                     eval_kwargs={'num_threads': 1,
                                                  'display_progress': False})
        return optimizer.compile(student, trainset=trainset, valset=valset)

    def optimize(self, task_desc, dev_examples, ood_examples=None):
        if ood_examples is not None:
            raise ValueError('DSPy accepts source data only; evaluate held-out targets after selection')
        train, val = split_source(list(dev_examples), self.seed, self.task)
        dspy = _load_dspy()
        self.history = []
        self.metadata = dict(
            implementation=f'DSPy {DSPY_VERSION} {self.method}, instruction-only',
            source_train_ids=[ex.id for ex in train], source_validation_ids=[ex.id for ex in val],
            seed=self.seed, budget=self.budget, inference_format='harness-exact-classification',
            optimizer_seed=self.seed if self.method in ('GEPA', 'MIPROv2') else None,
            validation_role='source selection; not independent confirmation',
            output_prefix_optimization=False,
            inference_temperature=getattr(self.backend, 'temperature', None),
            inference_max_tokens=getattr(self.backend, 'max_tokens', None),
            proposal_temperature=self.proposal_temperature, proposal_max_tokens=self.proposal_max_tokens,
            proposal_settings_policy='defaults; DSPy may override temperature per proposal',
            candidate_pool_scope='base and fully validated exported instruction',
            token_usage='backend estimates; not provider billing',
        )
        original = self.backend
        self.backend = BudgetedBackend(original, self.budget)
        try:
            self.backend.require_calls(len(val))
            baseline = Candidate(task_desc, self._dev_score(task_desc, val))
            self.history.append(baseline)
            allowance = self.backend.remaining - len(val)
            if allowance < 1:
                self.metadata['stop_reason'] = 'insufficient_compile_budget'
                return baseline
            compile_backend = BudgetedBackend(original, allowance)
            state = SimpleNamespace(lock=threading.RLock(), budget_exhausted=False, instructions={},
                                    validation_labels={ex.text: ex.label.strip().casefold() for ex in val},
                                    validation_scores={})
            lm = _make_bridge(dspy, compile_backend, temperature=self.proposal_temperature,
                              max_tokens=self.proposal_max_tokens, state=state)
            adapter = _make_task_adapter(dspy, compile_backend, self.task,
                                         {ex.text for ex in train + val}, state)
            signature = dspy.Signature('harness_text -> harness_label', instructions=task_desc)

            class TaskModule(dspy.Module):
                def __init__(self):
                    super().__init__()
                    self.prog = dspy.Predict(signature)

                def forward(self, harness_text):
                    return self.prog(harness_text=harness_text)

            def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
                label = parse_label(pred.harness_label, self.task)
                score = float(label != 'INVALID' and label == gold.harness_label.casefold())
                if self.method == 'GEPA':
                    return dspy.Prediction(score=score, feedback=(
                        f'Expected {gold.harness_label}; received {pred.harness_label!r}. '
                        + self.task.answer_format))
                return score

            def examples(rows):
                return [dspy.Example(harness_text=ex.text, harness_label=ex.label.strip()).with_inputs(
                    'harness_text') for ex in rows]

            try:
                with dspy.context(lm=lm, adapter=adapter, max_errors=1):
                    compiled = self._compile(dspy, TaskModule(), metric, lm,
                                             examples(train), examples(val), allowance)
                if state.budget_exhausted:
                    raise CallBudgetExceeded('DSPy compilation exhausted its allowance')
                predictors = compiled.predictors()
                if len(predictors) != 1 or predictors[0].demos:
                    raise ValueError('Compiled program violates the instruction-only contract')
                instruction = predictors[0].signature.instructions
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError('DSPy returned an empty instruction')
                self.metadata['stop_reason'] = 'completed'
            except Exception:
                if not state.budget_exhausted:
                    raise
                # DSPy can stop mid-candidate. Recover only an instruction that
                # completed this exact validation set, then use the reserved
                # requests to independently check its exported behavior.
                complete = [baseline] + [Candidate(text, sum(scores.values()) / len(val))
                    for text, scores in state.validation_scores.items() if len(scores) == len(val)]
                instruction = max(complete, key=lambda candidate: candidate.dev_score).instruction
                self.metadata['stop_reason'] = ('compile_budget_exhausted_kept_base' if instruction == task_desc
                                               else 'compile_budget_exhausted_validated_candidate')
            finally:
                self.metadata['evaluated_instructions'] = list(state.instructions)
            if instruction != task_desc:
                self.backend.require_calls(len(val))
                score = self._dev_score(instruction, val)
                self.history.append(Candidate(instruction, score))
            return max(self.history, key=lambda candidate: candidate.dev_score)
        except CallBudgetExceeded:
            if not self.history:
                raise
            self.metadata['stop_reason'] = 'validation_budget_exhausted_kept_base'
            return self.history[0]
        finally:
            self.backend = original


class DSPyGEPA(DSPyOptimizerAdapter):
    name = 'DSPy-GEPA'
    method = 'GEPA'


class DSPyMIPROv2(DSPyOptimizerAdapter):
    name = 'DSPy-MIPROv2'
    method = 'MIPROv2'


class DSPyCOPRO(DSPyOptimizerAdapter):
    name = 'DSPy-COPRO'
    method = 'COPRO'
