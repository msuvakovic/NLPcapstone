from __future__ import annotations

from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import MockBackend
from prompt_opt_harness.logger import save_run
from prompt_opt_harness.optimizers import EvoPromptLite, HumanWrittenBaseline, OPRO, ZeroShotBaseline, GEPA, MIPROv2, TextGrad
from prompt_opt_harness.report import build_report
import os

# Set dummy key for DSPy
os.environ["OPENAI_API_KEY"] = "dummy"

BUDGET = 40
SEED = 42


def main() -> None:
    dataset = load_demo_dataset()

    def backend_factory():
        return MockBackend(dataset.lookup(), source_domain=dataset.source_domain)

    methods = [
        ("Zero-shot", ZeroShotBaseline, len(dataset.source.dev)),
        ("Human-written", HumanWrittenBaseline, len(dataset.source.dev)),
        ("OPRO", OPRO, BUDGET),
        ("EvoPrompt-lite", EvoPromptLite, BUDGET),
        ("GEPA", GEPA, BUDGET),
        ("MIPROv2", MIPROv2, BUDGET),
        ("TextGrad", TextGrad, BUDGET),
    ]

    results = []
    for name, optimizer_cls, budget in methods:
        result = run_experiment(name, optimizer_cls, backend_factory, dataset, budget=budget, seed=SEED)
        results.append(result)
        log_path = save_run(result)
        print(f"[{name}] logged to {log_path}")

    print()
    print(build_report(results))

    print()
    print("Best OPRO instruction found:")
    print(f"  {next(r for r in results if r.name == 'OPRO').best_instruction}")


if __name__ == "__main__":
    main()

