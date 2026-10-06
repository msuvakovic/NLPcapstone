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
        # We need to map our LLMBackend interface to DSPy's LM interface.
        # This is a bit of a hack since DSPy wants to control the LM directly,
        # but we can wrap our backend to make it look like a DSPy LM to track
        # calls and use our retry logic, or we can just configure DSPy to use
        # a standard LiteLLM backend and live with separate tracking.
        import os
        from ..llm_backends import GroqBackend, OllamaBackend, OpenAIBackend, AnthropicBackend

        # Fallback dummy logic if we aren't using a real API backend (like MockBackend)
        # We create a dummy LM that actually delegates to our mock backend!
        class CustomDSPyLM(dspy.LM):
            def __init__(self, backend, model_name="dummy/dummy"):
                super().__init__(model_name)
                self.custom_backend = backend

            def __call__(self, prompt=None, messages=None, **kwargs):
                if prompt is None and messages is not None:
                    prompt = "\n".join(m.get("content", "") for m in messages)
                if prompt is None:
                    prompt = ""
                response = self.custom_backend.generate(prompt)
                return [response]

        if isinstance(self.backend, GroqBackend):
            lm = dspy.LM(model=f"groq/{self.backend.model}", api_key=os.environ.get("GROQ_API_KEY", ""))
        elif isinstance(self.backend, OllamaBackend):
            lm = dspy.LM(model=f"ollama/{self.backend.model}", api_base="http://localhost:11434")
        elif isinstance(self.backend, OpenAIBackend):
            lm = dspy.LM(model=f"openai/{self.backend.model}", api_key=os.environ.get("OPENAI_API_KEY", ""))
        elif isinstance(self.backend, AnthropicBackend):
            lm = dspy.LM(model=f"anthropic/{self.backend.model}", api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        else:
            lm = CustomDSPyLM(self.backend)
        
        dspy.settings.configure(lm=lm)
        self.dspy_lm = lm

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
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
        
        teleprompter = self.teleprompter_cls(metric=dspy_metric, **self.teleprompter_kwargs)

        # Some teleprompters (like COPRO) require eval_kwargs, while others don't.
        compile_kwargs = {"trainset": trainset}
        if self.name == "TextGrad" and self.teleprompter_cls.__name__ == "COPRO":
            compile_kwargs["eval_kwargs"] = {}

        compiled_module = teleprompter.compile(TaskModule(), **compile_kwargs)

        try:
            optimized_instruction = compiled_module.prog.signature.instructions
        except AttributeError:
            optimized_instruction = task_desc

        best_candidate = Candidate(instruction=optimized_instruction)
        best_candidate.dev_score = self._dev_score(best_candidate.instruction, dev_examples)
        self.history.append(best_candidate)
        
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
        else:
            self.teleprompter_cls = dspy.teleprompt.COPRO
        self.teleprompter_kwargs = {}
