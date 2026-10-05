from __future__ import annotations
from typing import Type
import math
from collections import Counter

from .base import Optimizer, Candidate, parse_label
from ..prompts import build_classification_prompt

def create_entropy_regularized_optimizer(base_optimizer_cls: Type[Optimizer], num_samples: int = 5, entropy_weight: float = 0.2) -> Type[Optimizer]:
    """
    Creates an optimizer class that penalizes prompts with high output entropy (inconsistent answers).
    It simulates high-temperature sampling by appending invisible whitespace to bypass caches,
    forcing the backend to draw independent samples.
    """
    
    class EntropyWrapper(base_optimizer_cls):
        name = f"Entropy-{base_optimizer_cls.name}"

        def _evaluation_calls(self, dev_examples) -> int:
            return num_samples * len(dev_examples)
        
        def _dev_score(self, instruction: str, dev_examples) -> float:
            if not dev_examples:
                return 0.0
                
            correct_count = 0
            total_entropy = 0.0
            
            for ex in dev_examples:
                predictions = []
                for i in range(num_samples):
                    # Add i spaces to the end of the prompt to force a cache miss 
                    # and simulate a new high-temp sample in deterministic/mock backends
                    base_prompt = build_classification_prompt(instruction, ex.text, self.task)
                    prompt = base_prompt + (" " * i)
                    pred = parse_label(self.backend.generate(prompt), self.task)
                    predictions.append(pred)
                
                # Majority vote for correctness
                counts = Counter(predictions)
                best_pred = counts.most_common(1)[0][0]
                if best_pred != 'INVALID' and best_pred == ex.label.strip().casefold():
                    correct_count += 1
                
                # Calculate normalized empirical entropy
                # H = -sum(p * log(p))
                # Max entropy for K unique answers is log(K). We normalize by log(num_samples) if > 1.
                entropy = 0.0
                for count in counts.values():
                    p = count / num_samples
                    entropy -= p * math.log(p)
                    
                max_entropy = math.log(num_samples) if num_samples > 1 else 1.0
                normalized_entropy = entropy / max_entropy
                
                total_entropy += normalized_entropy

            base_score = correct_count / len(dev_examples)
            avg_entropy = total_entropy / len(dev_examples)
            
            # Penalize the score by the average entropy
            return max(0.0, base_score - (entropy_weight * avg_entropy))

    return EntropyWrapper
