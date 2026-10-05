from __future__ import annotations
import random

from ..prompts import OPRO_META_TEMPLATES
from ..budget import BudgetedBackend, CallBudgetExceeded
from .base import Candidate, Optimizer

HISTORY_KEPT = 5
BEAM_SIZE = 2

class OPRO(Optimizer):
    name = "OPRO"

    def _eval_candidate(self, instruction: str, dev_examples, ood_examples: dict = None) -> Candidate:
        self._reject_target_feedback(ood_examples)
        if hasattr(self.backend, 'require_calls'):
            self.backend.require_calls(self._evaluation_calls(dev_examples))
        dev_score = self._dev_score(instruction, dev_examples)
        return Candidate(instruction=instruction, dev_score=dev_score)

    @staticmethod
    def _reject_target_feedback(ood_examples):
        if ood_examples is not None:
            raise ValueError("OPRO accepts source dev data only; ood_examples must be None. "
                             "Evaluate held-out targets after prompt selection.")

    def optimize(self, task_desc: str, dev_examples, ood_examples: dict = None) -> Candidate:
        self._reject_target_feedback(ood_examples)
        self.history = []
        if not dev_examples:
            raise ValueError('OPRO needs nonempty source development data')
        original_backend = self.backend
        self.backend = BudgetedBackend(original_backend, self.budget)
        try:
            return self._search(task_desc, dev_examples)
        except CallBudgetExceeded:
            # Never rank a partially evaluated candidate. If even the initial
            # score cannot complete, fail explicitly rather than invent a score.
            if not self.history:
                raise
            return max(self.history, key=lambda c: c.dev_score)
        finally:
            self.backend = original_backend

    def _search(self, task_desc, dev_examples):
        rng = random.Random(self.seed)
        seed = self._eval_candidate(task_desc, dev_examples)
        self.history.append(seed)

        score_calls = self._evaluation_calls(dev_examples)
        while self.backend.remaining > 0:
            # Candidate feedback, proposal history and selection use source dev only.
            top = sorted(self.history, key=lambda c: c.dev_score, reverse=True)[:HISTORY_KEPT]
            
            history_block = "\n".join(f"Instruction: {c.instruction}\nScore: {c.dev_score:.2f}\n" for c in top)
            
            # Beam search: generate multiple candidates using diverse templates
            candidates_this_step = []
            
            for _ in range(BEAM_SIZE):
                # Reserve a full score for each queued proposal before spending
                # a request generating another one. Retries are capped below.
                if self.backend.remaining < 1 + (len(candidates_this_step) + 1) * score_calls:
                    break
                # Diversify mutation operators
                meta_template = rng.choice(OPRO_META_TEMPLATES)
                meta_prompt = meta_template.format(history_block=history_block)
                
                new_instruction = self.backend.generate(meta_prompt).strip()
                candidates_this_step.append(new_instruction)
            
            if not candidates_this_step:
                break
                
            for candidate_index, new_instruction in enumerate(candidates_this_step):
                if self.backend.remaining == 0 and candidate_index > 0:
                    break
                cand = self._eval_candidate(new_instruction, dev_examples)
                self.history.append(cand)

        best = max(self.history, key=lambda c: c.dev_score)
        return best
