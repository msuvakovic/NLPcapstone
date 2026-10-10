from __future__ import annotations
import random
from typing import Type

from .base import Optimizer, evaluate_per_example


def source_subsets(dev_examples, seed, num_splits, split_ratio):
    """The same source IDs face every candidate, independent of process hash seed."""
    ids = sorted(ex.id for ex in dev_examples)
    if len(set(ids)) != len(ids):
        raise ValueError('DRO requires unique source example IDs')
    if not ids:
        return ()
    rng = random.Random(seed)
    size = max(1, int(len(ids) * split_ratio))
    return tuple(tuple(rng.sample(ids, size)) for _ in range(num_splits))

def create_dro_optimizer(base_optimizer_cls: Type[Optimizer], num_splits: int = 3, split_ratio: float = 0.5) -> Type[Optimizer]:
    """Worst accuracy over fixed source subsets; not actual domain-group DRO."""
    if not isinstance(num_splits, int) or isinstance(num_splits, bool) or num_splits < 1:
        raise ValueError('num_splits must be a positive integer')
    if not 0 < split_ratio <= 1:
        raise ValueError('split_ratio must be in (0, 1]')
    
    class DROWrapper(base_optimizer_cls):
        name = f"DRO-{base_optimizer_cls.name}"
        
        def _dev_score(self, instruction: str, dev_examples) -> float:
            if not dev_examples:
                return 0.0
            
            # Validate and choose the subsets before making any model requests.
            self.source_subset_ids = source_subsets(dev_examples, self.seed, num_splits, split_ratio)
            rows = evaluate_per_example(self.backend, instruction, dev_examples, self.task)
            self._rows_cache[instruction] = rows  # reused by failure-driven search
            correctness = {ex.id: correct for ex, _, correct in rows}
            return min(sum(correctness[eid] for eid in group) / len(group)
                       for group in self.source_subset_ids)

    return DROWrapper
