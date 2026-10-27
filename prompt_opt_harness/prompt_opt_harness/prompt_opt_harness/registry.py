"""Every optimizer the project has, by short name, in one place.

The run script, the time estimate in run_all.sh and the tests all read this list.
Groups: core (the six from the first plan), extended (the other branches' ideas), dspy (real GEPA / MIPROv2), all.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type

from .optimizers import (
    EvoPromptLite, GEPALite, OPRO, OracleOPRO, RandomSearch, TextGradLite,
)
from .optimizers.base import Optimizer
from .optimizers.bayesopt import CostBayesOpt
from .optimizers.dspy_adapters import DSPyGEPA, DSPyMIPROv2
from .optimizers.ensemble import EnsembleOptimizer
from .optimizers.pbt import PBT
from .optimizers.wrappers import (
    create_dro_optimizer, create_entropy_regularized_optimizer, create_guarded_optimizer,
    create_regularized_optimizer, create_sapo_optimizer,
)


@dataclass(frozen=True)
class Spec:
    key: str
    cls: Type[Optimizer]
    group: str
    budget_mult: int = 1          # SAPO and entropy score a prompt several times, so they get a proportionally larger budget
    needs: Tuple[str, ...] = ()   # python packages this optimizer needs
    install: str = ""             # the command that installs them
    note: str = ""

    @property
    def name(self) -> str:
        return self.cls.name


_SPECS: List[Spec] = [
    Spec("opro", OPRO, "core", note="asks the model for a better prompt, shown the best earlier ones"),
    Spec("evoprompt", EvoPromptLite, "core", note="mutates and crosses over a small population"),
    Spec("gepa", GEPALite, "core", note="reflects on one wrong answer, rewrites the prompt (simplified GEPA)"),
    Spec("textgrad", TextGradLite, "core", note="critiques a batch of wrong answers, rewrites from the critique (simplified TextGrad)"),
    Spec("random", RandomSearch, "core", note="independent rewrites with no feedback"),
    Spec("oracle", OracleOPRO, "core", note="upper bound: OPRO tuned on a sample of the new dataset"),
    Spec("pbt", PBT, "extended", note="population-based training (Kanika)"),
    Spec("ensemble", EnsembleOptimizer, "extended", note="paraphrases, then combines the best few (Kanika)"),
    Spec("bayesopt", CostBayesOpt, "extended", needs=("sklearn",), install="pip3 install scikit-learn", note="Gaussian-process search, prefers shorter prompts (Kanika)"),
    Spec("reg-opro", create_regularized_optimizer(OPRO), "extended", note="OPRO with a per-word length penalty (Kanika)"),
    Spec("dro-opro", create_dro_optimizer(OPRO), "extended", note="OPRO scored by its worst accuracy over question subsets (Kanika)"),
    Spec("sapo-opro", create_sapo_optimizer(OPRO), "extended", budget_mult=3, note="OPRO scored on the prompt and its paraphrases, worst wins (Kanika)"),
    Spec("entropy-opro", create_entropy_regularized_optimizer(OPRO), "extended", budget_mult=5, note="OPRO penalizing answers that change between samples (Kanika)"),
    Spec("guard-opro", create_guarded_optimizer(OPRO), "extended", note="OPRO whose winner must also beat the starting prompt on separate source questions (Albert)"),
    Spec("guard-gepa", create_guarded_optimizer(GEPALite), "extended", note="the same guard on GEPA-lite (Albert)"),
    Spec("dspy-gepa", DSPyGEPA, "dspy", needs=("dspy",), install='pip3 install "dspy==3.2.1"', note="real DSPy GEPA, instruction-only (Albert, apinero)"),
    Spec("dspy-mipro", DSPyMIPROv2, "dspy", needs=("dspy", "optuna"), install='pip3 install "dspy[optuna]==3.2.1"', note="real DSPy MIPROv2, instruction-only (Albert, apinero)"),
]
OPTIMIZERS: Dict[str, Spec] = {s.key: s for s in _SPECS}
GROUPS = ("core", "extended", "dspy", "all")


def available(spec: Spec) -> Tuple[bool, str]:
    missing = [m for m in spec.needs if importlib.util.find_spec(m) is None]
    if missing:
        return False, f"needs {', '.join(missing)}: {spec.install}"
    return True, ""


def resolve(names: str) -> List[Spec]:
    """'core', 'all', or a comma list like 'opro,gepa,dspy-gepa' (groups and names can be mixed)."""
    chosen: List[Spec] = []
    for token in (t.strip().lower() for t in names.split(",") if t.strip()):
        if token in GROUPS:
            chosen += [s for s in _SPECS if token == "all" or s.group == token]
        elif token in OPTIMIZERS:
            chosen.append(OPTIMIZERS[token])
        else:
            raise ValueError(f"Unknown optimizer '{token}'. Choose from: {', '.join(list(OPTIMIZERS) + list(GROUPS))}")
    seen, unique = set(), []
    for s in chosen:
        if s.key not in seen:
            seen.add(s.key)
            unique.append(s)
    return unique
