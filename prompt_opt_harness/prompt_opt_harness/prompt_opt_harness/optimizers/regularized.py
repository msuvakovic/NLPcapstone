from __future__ import annotations
from typing import Type

from .base import Optimizer, Candidate

def create_regularized_optimizer(base_optimizer_cls: Type[Optimizer], penalty_per_word: float = 0.005) -> Type[Optimizer]:
    """Creates a Length-Regularized optimizer class that penalizes overly long prompts."""
    
    class RegularizedWrapper(base_optimizer_cls):
        name = f"Reg-{base_optimizer_cls.name}"
        
        def _dev_score(self, instruction: str, dev_examples) -> float:
            base_score = super()._dev_score(instruction, dev_examples)
            word_count = len(instruction.split())
            
            # Apply linear penalty for length to encourage conciseness and generalization
            penalty = word_count * penalty_per_word
            
            return max(0.0, base_score - penalty)

    return RegularizedWrapper
