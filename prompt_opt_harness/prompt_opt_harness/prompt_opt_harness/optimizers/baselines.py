from __future__ import annotations

from .base import Candidate, Optimizer


class FixedPromptBaseline(Optimizer):
    instruction = ""

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        candidate = Candidate(self.instruction, self._dev_score(self.instruction, dev_examples))
        self.history.append(candidate)
        return candidate


class ZeroShotBaseline(FixedPromptBaseline):
    name = "Zero-shot"
    instruction = ""


class HumanWrittenBaseline(FixedPromptBaseline):
    name = "Human-written"
    instruction = (
        "Read the text and decide whether the overall sentiment is Positive or Negative. "
        "Consider the overall tone of the text, not just individual words."
    )


class HumanWrittenReasoningBaseline(FixedPromptBaseline):
    name = "Human-written"
    instruction = (
        "Solve the word problem step by step, showing your work. On the final line, "
        "write the answer in the exact form: Answer: <number>"
    )
