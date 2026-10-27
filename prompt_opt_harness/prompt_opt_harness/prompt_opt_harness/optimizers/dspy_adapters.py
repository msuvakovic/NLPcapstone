"""Real DSPy GEPA and MIPROv2, instruction-only, run through this harness's own model calls.

Based on Albert's tested adapter (`albert` branch) and the `apinero` branch's budget capping, rewritten for this
harness. Needs `pip3 install "dspy==3.2.1"` (checked at start; nothing else imports dspy).

How it stays comparable with the other optimizers:
- every model call, including DSPy's own proposal and reflection calls, goes through the harness backend, so
  tokens, call counts and the hard budget are the same as for OPRO and the rest;
- the task call uses the harness's exact prompt (`task.build_prompt`), not DSPy's own formatting, so the
  instruction DSPy finds is scored the way the final test scores it;
- DSPy searches on half of the tuning questions and validates on the other half; the starting prompt and the
  exported instruction are then both scored on ALL tuning questions, like every other optimizer, so the
  dev numbers are comparable;
- if the budget runs out inside DSPy, the best instruction DSPy fully validated is kept (or the starting
  prompt), never a half-scored one.
"""
from __future__ import annotations

import contextlib
import copy
import io
import importlib.metadata
import json
import os
import random
import threading
from types import SimpleNamespace

from .base import Candidate, Optimizer

DSPY_SERIES = "3.2"


class CallBudgetExceeded(RuntimeError):
    """Raised before a model call would exceed the allowance given to DSPy."""


class CappedBackend:
    """Counts calls from creation and refuses the call after `cap`; everything else passes through."""

    def __init__(self, backend, cap: int):
        self.backend, self.cap, self.start, self.rejected = backend, max(0, cap), backend.stats.calls, 0

    stats = property(lambda self: self.backend.stats)
    last_meta = property(lambda self: self.backend.last_meta)

    @property
    def used(self) -> int:
        return self.backend.stats.calls - self.start

    def generate(self, prompt, temperature=None, seed=None):
        if self.used >= self.cap:
            self.rejected += 1
            raise CallBudgetExceeded(f"DSPy allowance of {self.cap} calls used up")
        return self.backend.generate(prompt, temperature=temperature, seed=seed)


def _load_dspy():
    os.environ.setdefault("TQDM_DISABLE", "1")
    try:
        import dspy
    except ImportError as exc:
        raise ImportError('DSPy is not installed. Run: pip3 install "dspy==3.2.1"') from exc
    version = importlib.metadata.version("dspy")
    if not version.startswith(DSPY_SERIES):
        raise RuntimeError(f"This adapter was written for dspy {DSPY_SERIES}.x (found {version}). Run: pip3 install \"dspy==3.2.1\"")
    try:
        dspy.disable_logging()  # DSPy's own INFO logs and progress bars would bury the run's output
    except Exception:
        pass
    return dspy


def _make_lm(dspy, capped, state):
    """DSPy's own proposal/reflection calls, routed through the capped harness backend."""

    class HarnessLM(dspy.BaseLM):
        def __init__(self):
            super().__init__(model="harness", temperature=0.7, max_tokens=2048, cache=False)

        def copy(self, **kwargs):
            clone = copy.copy(self)
            clone.kwargs = dict(self.kwargs, **kwargs)
            clone.history = []
            return clone

        def __deepcopy__(self, memo):
            clone = self.copy()
            memo[id(self)] = clone
            return clone

        def forward(self, prompt=None, messages=None, **kwargs):
            options = dict(self.kwargs, **kwargs)
            if messages is not None:
                prompt = "Complete the following conversation. Return only the assistant reply.\n" + json.dumps(messages, ensure_ascii=False)
            n = options.get("n", 1)
            choices = []
            with state.lock:
                try:
                    for i in range(n):
                        text = capped.generate(prompt, temperature=options.get("temperature"), seed=state.next_seed())
                        choices.append(SimpleNamespace(message=SimpleNamespace(content=text)))
                except CallBudgetExceeded:
                    state.exhausted = True
                    raise
            return SimpleNamespace(choices=choices, model="harness", usage={})

    return HarnessLM()


def _make_adapter(dspy, capped, task, state):
    class ExactTaskAdapter(dspy.ChatAdapter):
        def __init__(self):
            super().__init__(use_json_adapter_fallback=False)

        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            if set(signature.input_fields) != {"harness_text"} or set(signature.output_fields) != {"harness_label"}:
                return super().__call__(lm, lm_kwargs, signature, demos, inputs)  # DSPy's own proposal prompts
            if demos:
                raise ValueError("instruction-only: few-shot demonstrations are not allowed")
            text, instruction = inputs["harness_text"], signature.instructions
            with state.lock:
                state.seen.setdefault(instruction, None)
                try:
                    raw = capped.generate(task.build_prompt(instruction, text))
                except CallBudgetExceeded:
                    state.exhausted = True
                    raise
                if text in state.val_labels:
                    state.val_scores.setdefault(instruction, {})[text] = float(task.is_correct(raw, state.val_labels[text]))
            return [{"harness_label": raw}]

    return ExactTaskAdapter()


class _State:
    def __init__(self, seed, val_labels):
        self.lock, self.exhausted, self.seen = threading.RLock(), False, {}
        self.val_labels, self.val_scores, self._seed, self._n = val_labels, {}, seed, 0

    def next_seed(self):
        self._n += 1
        return self._seed * 100000 + self._n


class DSPyOptimizerAdapter(Optimizer):
    name = "dspy-optimizer"
    method = None  # "GEPA" or "MIPROv2"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = {}

    def _teleprompter(self, dspy, metric, lm, allowance):
        if self.method == "GEPA":
            # GEPA counts metric evaluations, not requests; leave room for its reflection calls
            return dspy.GEPA(metric=metric, max_metric_calls=max(1, allowance // 2), reflection_lm=lm,
                             seed=self.seed, num_threads=1, track_stats=True)
        if self.method == "MIPROv2":
            return dspy.MIPROv2(metric=metric, auto="light", seed=self.seed, max_bootstrapped_demos=0,
                                max_labeled_demos=0, num_threads=1, max_errors=1, init_temperature=0.7)
        raise ValueError("method must be GEPA or MIPROv2")

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        dspy = _load_dspy()
        dev = list(dev_examples)
        if len(dev) < 4:
            raise ValueError("DSPy needs at least 4 tuning questions to split into train and validation")
        base = self._score(task_desc, dev)  # starting prompt on ALL tuning questions, like every optimizer
        allowance = self.budget - self.backend.stats.calls - self._score_cost(dev)  # keep one pass for the exported prompt
        self.metadata = {"implementation": f"DSPy {importlib.metadata.version('dspy')} {self.method}, instruction-only",
                         "allowance_calls": allowance}
        if allowance < 1:
            self.metadata["stop_reason"] = "budget_too_small_kept_starting_prompt"
            return base

        rows = sorted(dev, key=lambda ex: ex.id)
        random.Random(self.seed).shuffle(rows)
        half = len(rows) // 2
        train, val = rows[:half], rows[half:]
        capped = CappedBackend(self.backend, allowance)
        state = _State(self.seed, {ex.text: ex.label for ex in val})
        lm = _make_lm(dspy, capped, state)
        adapter = _make_adapter(dspy, capped, self.task, state)
        signature = dspy.Signature("harness_text -> harness_label", instructions=task_desc)

        class TaskModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.prog = dspy.Predict(signature)

            def forward(self, harness_text):
                return self.prog(harness_text=harness_text)

        task = self.task

        def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            score = float(task.is_correct(pred.harness_label, gold.harness_label))
            if self.method == "GEPA":
                return dspy.Prediction(score=score, feedback=(
                    f"Correct answer: {gold.harness_label}. The model's answer: {task.extract(pred.harness_label)}. "
                    f"End of its response: {pred.harness_label[-300:]!r}"))
            return score

        def examples(items):
            return [dspy.Example(harness_text=ex.text, harness_label=ex.label).with_inputs("harness_text") for ex in items]

        instruction = task_desc
        try:
            try:
                with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()), \
                        dspy.context(lm=lm, adapter=adapter, max_errors=1):  # DSPy prints progress bars
                    compiled = self._teleprompter(dspy, metric, lm, allowance).compile(
                        TaskModule(), trainset=examples(train), valset=examples(val))
                if state.exhausted:
                    raise CallBudgetExceeded("DSPy used up its allowance")
                predictors = compiled.predictors()
                if len(predictors) != 1 or predictors[0].demos:
                    raise ValueError("compiled program is not instruction-only")
                instruction = predictors[0].signature.instructions
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError("DSPy returned an empty instruction")
                self.metadata["stop_reason"] = "completed"
            except Exception:
                if not state.exhausted:
                    raise
                # out of calls inside DSPy: keep the best instruction it fully validated, else the starting prompt
                complete = [(sum(s.values()) / len(val), text) for text, s in state.val_scores.items() if len(s) == len(val)]
                instruction = max(complete)[1] if complete else task_desc
                self.metadata["stop_reason"] = "allowance_used_kept_best_validated"
        finally:
            self.metadata["instructions_tried"] = len(state.seen)
            self.metadata["calls_used_by_dspy"] = capped.used
            self.metadata["calls_refused"] = capped.rejected

        if instruction.strip() and instruction != task_desc:
            self._score(instruction, dev)
        return self._best()


class DSPyGEPA(DSPyOptimizerAdapter):
    name = "DSPy-GEPA"
    method = "GEPA"


class DSPyMIPROv2(DSPyOptimizerAdapter):
    name = "DSPy-MIPROv2"
    method = "MIPROv2"
