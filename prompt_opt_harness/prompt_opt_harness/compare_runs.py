from __future__ import annotations
import copy

from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import MockBackend
from prompt_opt_harness.optimizers import EvoPromptLite, OPRO, ZeroShotBaseline, GEPA, create_dro_optimizer, OracleOPRO
from prompt_opt_harness.report import build_report
import prompt_opt_harness.optimizers.base as base

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
        ("Oracle Upper-Bound", OracleOPRO, BUDGET),
    ]

    # --- Run 1: Old Static Objective ---
    print("=== RUN 1: Old Static Objective (Before) ===")
    
    original_progressive_score = base.Optimizer.progressive_score
    def static_score(self, dev, ood, prog):
        return self.ood_rank_score(dev, ood)
    base.Optimizer.progressive_score = static_score
    
    results_before = []
    for name, optimizer_cls, budget in methods:
        res = run_experiment(name, optimizer_cls, backend_factory, dataset, budget=budget, seed=SEED)
        results_before.append(res)
        
    print(build_report(results_before))
    print()
    
    # --- Run 2: New Progressive Objective ---
    print("=== RUN 2: Progressive Widening Objective (After) ===")
    base.Optimizer.progressive_score = original_progressive_score
    
    results_after = []
    for name, optimizer_cls, budget in methods:
        res = run_experiment(name, optimizer_cls, backend_factory, dataset, budget=budget, seed=SEED)
        results_after.append(res)
        
    print(build_report(results_after))
    
if __name__ == '__main__':
    run_comparison()
