from __future__ import annotations

from .base import Candidate
from .opro import OPRO

class OracleOPRO(OPRO):
    name = "Oracle-OPRO"
    
    def optimize(self, task_desc: str, dev_examples, ood_examples: dict = None) -> Candidate:
        raise ValueError("OracleOPRO is disabled: its old implementation trained on final "
                         "target tests. A target-trained reference requires separate "
                         "target development and test splits. Use OPRO for source-only runs.")
