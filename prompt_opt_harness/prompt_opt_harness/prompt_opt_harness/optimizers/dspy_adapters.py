from __future__ import annotations

from .base import Candidate, Optimizer

# Not wired up yet -- GEPA/MIPROv2 are DSPy teleprompters, not raw prompt-string
# optimizers, so they need a dspy.Signature/Module + metric fn, not just this
# Candidate interface. Reference usage (pip install "dspy>=3.2.1,<3.3"):
#
#   import dspy
#   dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key=...))
#
#   from dspy.teleprompt import MIPROv2
#   teleprompter = MIPROv2(metric=your_metric, auto="light")
#   optimized = teleprompter.compile(your_dspy_module, trainset=trainset)
#
#   gepa = dspy.GEPA(metric=your_metric, auto="light", reflection_lm=dspy.LM("openai/gpt-5"))
#   optimized = gepa.compile(your_dspy_module, trainset=trainset, valset=devset)


class DSPyOptimizerAdapter(Optimizer):
    name = "dspy-optimizer"
    teleprompter_cls = None  # e.g. dspy.GEPA or dspy.teleprompt.MIPROv2
    teleprompter_kwargs: dict = {}

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        raise NotImplementedError("wire up dspy.Signature/Module + metric fn, see module comment")
