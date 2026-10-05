from __future__ import annotations

from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import MockBackend
from prompt_opt_harness.optimizers import EvoPromptLite, OPRO, ZeroShotBaseline, GEPA, create_dro_optimizer
from prompt_opt_harness.report import build_report

BUDGET = 200
SEED = 42

def run_comparison():
    dataset = load_demo_dataset()

    def backend_factory():
        return MockBackend(dataset.lookup(), source_domain=dataset.source_domain)
        
    methods = [
        ("Zero-shot", ZeroShotBaseline, 1),
        ("OPRO", OPRO, BUDGET),
        ("GEPA", GEPA, BUDGET),
        ("EvoPrompt", EvoPromptLite, BUDGET),
        ("DRO", create_dro_optimizer(OPRO), BUDGET),
    ]

    print("=== Mock pipeline check: source-only selection; not model performance ===")
    
    results_after = []
    for name, optimizer_cls, budget in methods:
        res = run_experiment(name, optimizer_cls, backend_factory, dataset, budget=budget, seed=SEED)
        results_after.append(res)
        
    print(build_report(results_after))
    
if __name__ == '__main__':
    run_comparison()
