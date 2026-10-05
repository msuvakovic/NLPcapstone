from __future__ import annotations
from typing import Type

from ..prompts import PARAPHRASE_TEMPLATE
from .base import Optimizer

def create_sapo_optimizer(base_optimizer_cls: Type[Optimizer], num_perturbations: int = 2) -> Type[Optimizer]:
    """Creates a Sharpness-Aware Prompt Optimizer that evaluates paraphrased variants of a prompt to ensure flatness."""
    
    class SAPOWrapper(base_optimizer_cls):
        name = f"SAPO-{base_optimizer_cls.name}"

        def _evaluation_calls(self, dev_examples) -> int:
            if not dev_examples:
                return 0
            return (num_perturbations + 1) * super()._evaluation_calls(dev_examples) + num_perturbations
        
        def _dev_score(self, instruction: str, dev_examples) -> float:
            if not dev_examples:
                return 0.0
            
            # Step 1: Score the base instruction
            base_score = super()._dev_score(instruction, dev_examples)
            
            # Step 2: Generate paraphrased variants
            variants = []
            for _ in range(num_perturbations):
                prompt = PARAPHRASE_TEMPLATE.format(instruction=instruction)
                variant = self.backend.generate(prompt).strip()
                variants.append(variant)
                
            # Step 3: Score variants and find the worst-case score
            scores = [base_score]
            for var in variants:
                scores.append(super()._dev_score(var, dev_examples))
                
            return min(scores)

    return SAPOWrapper
