from .base import Candidate, Optimizer
from .opro import OPRO
from .evoprompt import EvoPromptLite
from .baselines import ZeroShotBaseline, HumanWrittenBaseline
from .gepa import GEPA
from .dro import create_dro_optimizer
from .regularized import create_regularized_optimizer
from .sapo import create_sapo_optimizer
from .textgrad import TextGrad
from .pbt import PBT
from .reflective import ReflectiveOptimizer
from .ensemble import EnsembleOptimizer
from .oracle import OracleOPRO
from .entropy import create_entropy_regularized_optimizer
from .dspy_adapters import DSPyGEPA, DSPyMIPROv2, DSPyCOPRO


def __getattr__(name):
    # Optional Bayesian dependencies must not be required to import the harness
    # or install only the DSPy extra. Existing explicit imports still work.
    modules = {'CostAwareBayesOpt': '.bayesopt', 'EmbeddingBayesOpt': '.bayesopt_emb'}
    if name not in modules:
        raise AttributeError(name)
    from importlib import import_module
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "Candidate",
    "DSPyGEPA",
    "DSPyMIPROv2",
    "DSPyCOPRO",
    "Optimizer",
    "OPRO",
    "EvoPromptLite",
    "ZeroShotBaseline",
    "HumanWrittenBaseline",
    "GEPA",
    "create_dro_optimizer",
    "create_regularized_optimizer",
    "create_sapo_optimizer",
    "TextGrad",
    "PBT",
    "ReflectiveOptimizer",
    "EnsembleOptimizer",
    "CostAwareBayesOpt",
    "EmbeddingBayesOpt",
    "OracleOPRO",
    "create_entropy_regularized_optimizer",
]
