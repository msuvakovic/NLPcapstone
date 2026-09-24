from __future__ import annotations
import math
import random
from typing import Type

from .base import Candidate, Optimizer, evaluate_per_example

def create_dro_optimizer(base_optimizer_cls: Type[Optimizer], num_splits: int = 3, split_ratio: float = 0.5) -> Type[Optimizer]:
    """Creates a DRO-wrapped optimizer class that overrides the dev_score method to compute worst-case accuracy across random splits."""
    
    class DROWrapper(base_optimizer_cls):
        name = f"DRO-{base_optimizer_cls.name}"
        
        def _dev_score(self, instruction: str, dev_examples) -> float:
            if not dev_examples:
                return 0.0
            
            # Evaluate all examples
            per_example_results = evaluate_per_example(self.backend, instruction, dev_examples)
            
            # Create random splits (bootstrapping without replacement for simplicity)
            split_size = max(1, int(len(dev_examples) * split_ratio))
            
            # Use self.seed for reproducible splits per run, combined with some hash of the instruction
            # so different instructions get different splits, but deterministically
            rng = random.Random(self.seed + hash(instruction))
            
            split_scores = []
            for _ in range(num_splits):
                sampled = rng.sample(per_example_results, split_size)
                correct_count = sum(1 for res in sampled if res[2])
                split_scores.append(correct_count / split_size)
                
            return min(split_scores)

    return DROWrapper
