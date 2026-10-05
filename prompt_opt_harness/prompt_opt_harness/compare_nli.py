from __future__ import annotations

from prompt_opt_harness.datasets import load_nli_demo_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import MockBackend
from prompt_opt_harness.optimizers import EvoPromptLite, OPRO, ZeroShotBaseline, GEPA, create_dro_optimizer
from prompt_opt_harness.report import build_report
from prompt_opt_harness.optimizers.entropy import create_entropy_regularized_optimizer

BUDGET = 200
SEED = 42

def run_nli_comparison():
    dataset = load_nli_demo_dataset()

    def backend_factory():
        return MockBackend(dataset.lookup(), source_domain=dataset.source_domain)
        
    EntropyOPRO = create_entropy_regularized_optimizer(OPRO, num_samples=3, entropy_weight=0.3)

    methods = [
        ("Zero-shot", ZeroShotBaseline, 1),
        ("OPRO", OPRO, BUDGET),
        ("Entropy-OPRO", EntropyOPRO, BUDGET),
        ("GEPA", GEPA, BUDGET),
        ("EvoPrompt", EvoPromptLite, BUDGET),
        ("DRO", create_dro_optimizer(OPRO), BUDGET),
    ]

    print("=== Mock NLI pipeline check: source-only selection; not model performance ===")
    
    results = []
    for name, optimizer_cls, budget in methods:
        res = run_experiment(name, optimizer_cls, backend_factory, dataset, budget=budget, seed=SEED)
        results.append(res)
        
    print(build_report(results))
    
if __name__ == '__main__':
    run_nli_comparison()
