from __future__ import annotations

from .base import Candidate
from .opro import OPRO

class OracleOPRO(OPRO):
    name = "Oracle-OPRO"
    
    def optimize(self, task_desc: str, dev_examples, ood_examples: dict = None) -> Candidate:
        # Combine all data (source + all OOD domains) into one massive dev set
        all_examples = list(dev_examples)
        if ood_examples:
            for examples in ood_examples.values():
                all_examples.extend(examples)
                
        # Call standard OPRO optimize, but passing the combined examples as dev_examples
        # and no OOD examples (so it purely optimizes for accuracy on everything)
        return super().optimize(task_desc, all_examples, None)
