from __future__ import annotations

from typing import Optional

from .base import Candidate, Optimizer
from ..tasks import Task

try:
    import dspy
except ImportError:
    dspy = None

class DSPyOptimizerAdapter(Optimizer):
    name = "dspy-optimizer"
    teleprompter_cls = None  # e.g. dspy.GEPA or dspy.teleprompt.MIPROv2
    teleprompter_kwargs: dict = {}

    def __init__(self, backend, budget: int, seed: int = 0, task: Optional[Task] = None):
        if dspy is None:
            raise ImportError("pip install dspy>=3.2.1")
        if task is None:
            from ..tasks import SENTIMENT_TASK
            task = SENTIMENT_TASK
        super().__init__(backend, budget, seed, task)
        self._setup_dspy_lm()

    def _setup_dspy_lm(self):
        from ..llm_backends import BudgetExceeded

        class CustomDSPyLM(dspy.LM):
            def __init__(self, backend, model_name="dummy/dummy"):
                # Never invoke LiteLLM: both calls and replies go through the
                # harness backend so accounting and hard limits are reliable.
                super().__init__(model_name, cache=False)
                self.custom_backend = backend

            def __call__(self, prompt=None, messages=None, **kwargs):
                if prompt is None and messages is not None:
                    prompt = "\n".join(m.get("content", "") for m in messages)
                if prompt is None:
                    prompt = ""
                try:
                    response = self.custom_backend.generate(prompt)
                except BudgetExceeded:
                    # DSPy evaluators often catch per-example exceptions and
                    # continue, which would hide a budget stop as a bad score.
                    # Return an invalid/empty completion, record the stop, and
                    # let the adapter report it explicitly in run metadata.
                    return [""]
                return [response]

        # Always route DSPy's task, proposal, and reflection calls through the
        # shared budgeted backend. Direct provider LMs bypass call accounting.
        lm = CustomDSPyLM(self.backend)
        dspy.settings.configure(lm=lm)
        self.dspy_lm = lm

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        if len(dev_examples) > self.budget:
            raise ValueError(
                f"budget ({self.budget}) must cover the final dev evaluation "
                f"({len(dev_examples)} calls)"
            )

        if self.name == "TextGrad" and self.teleprompter_cls.__name__ == "COPRO":
            raise RuntimeError("Refusing to label COPRO results as TextGrad")

        # Reserve enough calls to score the returned instruction after DSPy's
        # search. All requests made inside compile are additionally bounded by
        # this shared budgeted backend, including reflection/proposal calls.
        search_budget = self.budget - len(dev_examples)
        if search_budget == 0:
            candidate = Candidate(task_desc, self._dev_score(task_desc, dev_examples))
            self.history.append(candidate)
            return candidate

        from ..llm_backends import BudgetExceeded, BudgetedBackend
        search_backend = BudgetedBackend(self.backend, search_budget)
        self.dspy_lm.custom_backend = search_backend

        class TaskSignature(dspy.Signature):
            __doc__ = f"{task_desc}\n\nGiven the input, generate the final output."
            input_text: str = dspy.InputField(desc="The input to the task.")
            output: str = dspy.OutputField(desc="The output response.")

        class TaskModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.prog = dspy.Predict(TaskSignature)
            
            def forward(self, input_text):
                return self.prog(input_text=input_text)

        trainset = [dspy.Example(input_text=ex.text, output=ex.label).with_inputs("input_text") for ex in dev_examples]

        def dspy_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
            return self.task.is_correct(pred.output, example.output)

        if self.teleprompter_cls is None:
            raise ValueError("teleprompter_cls must be set on the subclass")
        
        kwargs = dict(self.teleprompter_kwargs)
        if self.name == "GEPA":
            # DSPy requires exactly one of auto/max_metric_calls/max_full_evals.
            kwargs.pop("auto", None)
            # GEPA counts metric evaluations, not model requests. Leave most
            # of the request allowance for reflection; the shared backend is
            # still the authoritative hard cap.
            kwargs["max_metric_calls"] = max(1, search_budget // 4)
            kwargs["num_threads"] = 1
            kwargs["seed"] = self.seed

        # Some teleprompters (like COPRO) require eval_kwargs, while others don't.
        compile_kwargs = {"trainset": trainset}
        if self.name == "MIPROv2":
            # Explicit trials cannot be combined with MIPROv2's auto presets.
            kwargs["auto"] = None
            trials = max(1, min(3, search_budget // max(1, len(dev_examples) * 5)))
            kwargs["num_candidates"] = 2
            kwargs["max_errors"] = max(100, len(dev_examples) * 10)
            kwargs["num_threads"] = 1
            kwargs["seed"] = self.seed
            compile_kwargs["num_trials"] = trials
            compile_kwargs["minibatch"] = False
        teleprompter = self.teleprompter_cls(metric=dspy_metric, **kwargs)
        try:
            try:
                compiled_module = teleprompter.compile(TaskModule(), **compile_kwargs)
            except Exception:
                if search_backend.rejected_calls == 0:
                    raise
                compiled_module = TaskModule()
            if search_backend.rejected_calls:
                # DSPy may swallow the exhausted-call response as an ordinary
                # prediction error. Never accept a partial compile as an
                # optimized result: explicitly evaluate the original prompt.
                optimized_instruction = task_desc
            else:
                try:
                    optimized_instruction = compiled_module.prog.signature.instructions
                except AttributeError:
                    optimized_instruction = task_desc
        finally:
            search_backend.close()
            self.dspy_lm.custom_backend = self.backend
        best_candidate = Candidate(instruction=optimized_instruction)
        best_candidate.dev_score = self._dev_score(best_candidate.instruction, dev_examples)
        self.history.append(best_candidate)
        self.budget_exhausted = search_backend.rejected_calls > 0
        
        return best_candidate

class GEPA(DSPyOptimizerAdapter):
    name = "GEPA"
    def __init__(self, backend, budget: int, seed: int = 0, task=None):
        super().__init__(backend, budget, seed, task)
        self.teleprompter_cls = dspy.GEPA
        self.teleprompter_kwargs = {"auto": "light", "reflection_lm": self.dspy_lm}

class MIPROv2(DSPyOptimizerAdapter):
    name = "MIPROv2"
    def __init__(self, backend, budget: int, seed: int = 0, task=None):
        super().__init__(backend, budget, seed, task)
        self.teleprompter_cls = dspy.teleprompt.MIPROv2
        self.teleprompter_kwargs = {"auto": "light"}

class TextGrad(DSPyOptimizerAdapter):
    name = "TextGrad"
    def __init__(self, backend, budget: int, seed: int = 0, task=None):
        super().__init__(backend, budget, seed, task)
        if hasattr(dspy.teleprompt, "TextGrad"):
            self.teleprompter_cls = dspy.teleprompt.TextGrad
            self.implementation = "DSPy TextGrad"
        else:
            raise ImportError(
                "This DSPy installation has no TextGrad optimizer; refusing to "
                "silently substitute COPRO. Install a TextGrad implementation "
                "or run the experiment without that method."
            )
        self.teleprompter_kwargs = {}
