from __future__ import annotations
import random

from ..prompts import OPRO_META_TEMPLATES
from .base import Candidate, Optimizer

HISTORY_KEPT = 5
BEAM_SIZE = 2

class OPRO(Optimizer):
    name = "OPRO"

    def _eval_candidate(self, instruction: str, dev_examples, ood_examples: dict = None) -> Candidate:
        dev_score = self._dev_score(instruction, dev_examples)
        
        ood_accs = None
        if ood_examples:
            ood_accs = {}
            for domain, examples in ood_examples.items():
                ood_accs[domain] = self._dev_score(instruction, examples)
            
        cand = Candidate(instruction=instruction, dev_score=dev_score)
        cand.ood_accs = ood_accs
        return cand

    def optimize(self, task_desc: str, dev_examples, ood_examples: dict = None) -> Candidate:
        seed = self._eval_candidate(task_desc, dev_examples, ood_examples)
        self.history.append(seed)

        while self.backend.stats.calls < self.budget:
            progress = self.backend.stats.calls / self.budget
            
            # Rank candidates using progressive widening score
            for c in self.history:
                if c.ood_accs:
                    c.ood_score = self.progressive_score(c.dev_score, c.ood_accs, progress)
                else:
                    c.ood_score = c.dev_score
            
            # Keep elite prompt pool based on progressive score
            top = sorted(self.history, key=lambda c: c.ood_score, reverse=True)[:HISTORY_KEPT]
            
            history_block = "\n".join(f"Instruction: {c.instruction}\nScore: {c.ood_score:.2f}\n" for c in top)
            
            # Beam search: generate multiple candidates using diverse templates
            candidates_this_step = []
            
            for _ in range(BEAM_SIZE):
                if self.backend.stats.calls >= self.budget:
                    break
                # Diversify mutation operators
                meta_template = random.choice(OPRO_META_TEMPLATES)
                meta_prompt = meta_template.format(history_block=history_block)
                
                new_instruction = self.backend.generate(meta_prompt).strip()
                candidates_this_step.append(new_instruction)
            
            if not candidates_this_step:
                break
                
            for new_instruction in candidates_this_step:
                if self.backend.stats.calls >= self.budget and candidates_this_step.index(new_instruction) > 0:
                    break
                cand = self._eval_candidate(new_instruction, dev_examples, ood_examples)
                self.history.append(cand)

        # Final selection purely based on OOD rank score
        for c in self.history:
            if c.ood_accs:
                c.ood_score = self.progressive_score(c.dev_score, c.ood_accs, 1.0)
            else:
                c.ood_score = c.dev_score
                
        best = max(self.history, key=lambda c: c.ood_score)
        return best
