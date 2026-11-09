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

    def _eval_cost(self, dev_examples, ood_examples) -> int:
        # LLM calls needed to score one candidate (dev + every OOD validation example)
        return len(dev_examples) + sum(len(v) for v in (ood_examples or {}).values())

    def _can_afford(self, extra_calls: int) -> bool:
        return self.backend.stats.calls + extra_calls <= self.budget

    def optimize(self, task_desc: str, dev_examples, ood_examples: dict = None) -> Candidate:
        rnd = random.Random(self.seed)
        eval_cost = self._eval_cost(dev_examples, ood_examples)
        seed = self._eval_candidate(task_desc, dev_examples, ood_examples)
        self.history.append(seed)
        seen = {seed.instruction}

        # each round needs at least one meta-prompt call plus one candidate evaluation
        while self._can_afford(1 + eval_cost):
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
                # only write another candidate if we can also afford to score it and the ones before it
                if not self._can_afford(1 + eval_cost * (len(candidates_this_step) + 1)):
                    break
                # Diversify mutation operators
                meta_template = rnd.choice(OPRO_META_TEMPLATES)
                meta_prompt = meta_template.format(task_name=self.task.name, history_block=history_block)
                
                new_instruction = self.backend.generate(meta_prompt).strip()
                candidates_this_step.append(new_instruction)
            
            if not candidates_this_step:
                break
                
            for new_instruction in candidates_this_step:
                if new_instruction in seen:  # skip duplicates instead of paying to re-score them
                    continue
                if not self._can_afford(eval_cost):
                    break
                seen.add(new_instruction)
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
