from __future__ import annotations

from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import OpenAIBackend
from prompt_opt_harness.logger import save_run
from prompt_opt_harness.optimizers import EvoPromptLite, HumanWrittenBaseline, OPRO, ZeroShotBaseline, GEPA, create_dro_optimizer
from prompt_opt_harness.report import build_report

BUDGET = 40
SEED = 42


def main() -> None:
    dataset = load_demo_dataset()

    def backend_factory():
        return OpenAIBackend(model="gpt-4o-mini")

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

    results = [
        run_experiment(name, cls, backend_factory, dataset, budget=budget, seed=SEED)
        for name, cls, budget in methods
    ]
    for r in results:
        print(f"[{r.name}] logged to {save_run(r)}")

    print()
    print(build_report(results))


if __name__ == "__main__":
    main()
