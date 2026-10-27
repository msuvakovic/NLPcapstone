from __future__ import annotations

from .opro import OPRO


class OracleOPRO(OPRO):
    """Upper bound: OPRO tuned directly on a held-out sample of the NEW dataset.

    The sample is separate from the new-dataset test questions that get reported, but this is
    deliberately not a fair optimizer: it sees the shifted domain. Its Dev column is accuracy
    on that sample. Compare the others against it to see how much of the gap is the shift
    itself and how much is the optimizer overfitting to the source domain.
    """

    name = "Oracle-OPRO"
    tune_on_target = True
