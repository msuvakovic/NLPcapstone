from __future__ import annotations

from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import MockBackend
from prompt_opt_harness.logger import save_run
from prompt_opt_harness.optimizers import EvoPromptLite, HumanWrittenBaseline, OPRO, ZeroShotBaseline, GEPA, create_dro_optimizer, create_regularized_optimizer, create_sapo_optimizer, TextGrad
from prompt_opt_harness.report import build_report

BUDGET = 40
SEED = 42


def main() -> None:
    dataset = load_demo_dataset()

    def backend_factory():
        return MockBackend(dataset.lookup(), source_domain=dataset.source_domain)
        
    def cross_backend_factory():
        # Simulates a different model that doesn't have the "DOMAIN_CUE_BONUS" overfitting quirks
        class StrictMock(MockBackend):
            def _classify(self, full_prompt: str, text: str) -> str:
                # Less susceptible to domain memorization
                label, domain = self.lookup.get(text, ("Positive", "unknown"))
                prob_correct = 0.65
                from prompt_opt_harness.llm_backends import QUALITY_BONUS, QUALITY_FRAGMENTS, _pseudo_random
                instruction = full_prompt.split("\n\nText:")[0].lower()
                prob_correct += QUALITY_BONUS * sum(f.lower() in instruction for f in QUALITY_FRAGMENTS)
                
                correct = _pseudo_random(f"{text}::{full_prompt}") < prob_correct
                return label if correct else ("Negative" if label == "Positive" else "Positive")
        return StrictMock(dataset.lookup(), source_domain=dataset.source_domain)

    methods = [
        ("Zero-shot", ZeroShotBaseline, 1),
        ("Human-written", HumanWrittenBaseline, 1),
        ("OPRO", OPRO, BUDGET),
        ("EvoPrompt-lite", EvoPromptLite, BUDGET),
        ("GEPA", GEPA, BUDGET),
        ("TextGrad", TextGrad, BUDGET),
        ("DRO-OPRO", create_dro_optimizer(OPRO), BUDGET),
        ("Reg-OPRO", create_regularized_optimizer(OPRO), BUDGET),
        ("SAPO-OPRO", create_sapo_optimizer(OPRO), BUDGET),
    ]

    results = []
    for name, optimizer_cls, budget in methods:
        result = run_experiment(name, optimizer_cls, backend_factory, dataset, budget=budget, seed=SEED)
        results.append(result)
        log_path = save_run(result)
        print(f"[{name}] logged to {log_path}")

    # Add Cross-Model Test
    cross_result = run_experiment(
        "Cross-Model (OPRO)", 
        OPRO, 
        backend_factory, 
        dataset, 
        budget=BUDGET, 
        seed=SEED,
        eval_backend_factory=cross_backend_factory
    )
    results.append(cross_result)
    log_path = save_run(cross_result)
    print(f"[Cross-Model] logged to {log_path}")

    print()
    print(build_report(results))

    print()
    print("Best OPRO instruction found:")
    print(f"  {next(r for r in results if r.name == 'OPRO').best_instruction}")


if __name__ == "__main__":
    main()
