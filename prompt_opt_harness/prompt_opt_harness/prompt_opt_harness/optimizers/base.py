from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..tasks import SENTIMENT_TASK, Task


@dataclass
class Candidate:
    instruction: str
    dev_score: Optional[float] = None


def evaluate_detailed(backend, instruction: str, examples, task: Task = SENTIMENT_TASK) -> Tuple[float, List[dict]]:
    if not examples:
        return 0.0, []
    records = []
    for ex in examples:
        prompt = task.build_prompt(instruction, ex.text)
        response = backend.generate(prompt)
        meta = backend.last_meta
        records.append({
            "id": ex.id,
            "domain": ex.domain,
            "question": ex.text,
            "gold": ex.label,
            "response": response,
            "predicted": task.extract(response),
            "correct": task.is_correct(response, ex.label),
            "well_formed": task.well_formed(response),
            "finish_reason": meta.get("finish_reason"),
            "truncated": meta.get("finish_reason") == "length",
            "output_tokens": meta.get("output_tokens"),
        })
    return sum(r["correct"] for r in records) / len(records), records


def evaluate(backend, instruction: str, examples, task: Task = SENTIMENT_TASK) -> float:
    return evaluate_detailed(backend, instruction, examples, task)[0]


def format_failure(record: dict, limit: int = 400) -> str:
    """One wrong answer as plain text for a reflection or critique prompt."""
    response = (record.get("response") or "").strip()
    tail = response[-limit:] if len(response) > limit else response
    predicted = record.get("predicted")
    note = " (the response was cut off before the final answer)" if record.get("truncated") else ""
    return (
        f"Input: {(record.get('question') or '')[:limit]}\n"
        f"Correct answer: {record.get('gold')}\n"
        f"Model's answer: {predicted if predicted is not None else 'none found'}{note}\n"
        f"End of the model's response: {tail}"
    )


class Optimizer:
    name = "base"
    # True only for oracle upper bounds, which tune on a held-out sample of the
    # new dataset instead of the source tuning set. Nothing else may set this.
    tune_on_target = False
    # True for guarded optimizers: after the search, the harness checks the winner against a separate
    # source sample (dataset.source.confirm) and keeps the starting prompt unless the winner beats it there.
    needs_confirm = False

    def __init__(self, backend, budget: int, seed: int = 0, task: Task = SENTIMENT_TASK,
                 proposal_temperature: Optional[float] = None):
        self.backend = backend
        self.budget = budget  # max LLM calls for this run
        self.seed = seed
        self.task = task
        self.proposal_temperature = proposal_temperature  # None = backend default
        self.history: List[Candidate] = []
        # one entry per dev evaluation, in the same order as history
        self.dev_stats: List[dict] = []
        self.dev_records: List[List[dict]] = []
        self._proposals = 0

    def _propose(self, meta_prompt: str) -> str:
        self._proposals += 1
        return self.backend.generate(
            meta_prompt,
            temperature=self.proposal_temperature,
            seed=self.seed + self._proposals,
        ).strip()

    def _record(self, records: List[dict]) -> float:
        """Log one scored pass over the tuning questions and return its accuracy."""
        tokens = [r["output_tokens"] for r in records if r["output_tokens"] is not None]
        self.dev_stats.append({
            "calls_after": self.backend.stats.calls,
            "n_truncated": sum(r["truncated"] for r in records),
            "n_unformatted": sum(not r["well_formed"] for r in records),
            "mean_output_tokens": sum(tokens) / len(tokens) if tokens else None,
        })
        self.dev_records.append(records)
        return sum(r["correct"] for r in records) / len(records) if records else 0.0

    def _dev_score(self, instruction: str, dev_examples) -> float:
        _, records = evaluate_detailed(self.backend, instruction, dev_examples, task=self.task)
        return self._record(records)

    def _score_cost(self, dev_examples) -> int:
        """Calls one scored prompt costs. Variants that sample more (SAPO, entropy) override this."""
        return len(dev_examples)

    # ---- helpers shared by the newer optimizers ----

    def _score(self, instruction: str, dev_examples) -> Candidate:
        """Score an instruction on the tuning questions and add it to the history."""
        n_before = len(self.dev_records)
        candidate = Candidate(instruction, self._dev_score(instruction, dev_examples))
        # a variant may log extra passes (e.g. SAPO's paraphrases); keep one record per candidate so
        # history, dev_records and dev_stats stay aligned
        del self.dev_records[n_before + 1:]
        del self.dev_stats[n_before + 1:]
        self.history.append(candidate)
        return candidate

    def _best(self) -> Candidate:
        return max(self.history, key=lambda c: c.dev_score)

    def _failures(self, candidate: Candidate) -> List[dict]:
        """The wrong answers recorded when this candidate was scored."""
        index = next(i for i, c in enumerate(self.history) if c is candidate)
        return [r for r in self.dev_records[index] if not r["correct"]]

    def _can_afford(self, proposals: int, scores: int, dev_examples) -> bool:
        """True if proposals + `scores` full passes over the tuning questions fit in the budget.

        Checked before any call is made, so a search never goes over budget.
        """
        needed = proposals + scores * self._score_cost(dev_examples)
        return self.backend.stats.calls + needed <= self.budget

    def optimize(self, task_desc: str, dev_examples) -> Candidate:
        raise NotImplementedError
