from __future__ import annotations

from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import GroqBackend
from prompt_opt_harness.logger import save_run
from prompt_opt_harness.optimizers import EvoPromptLite, HumanWrittenBaseline, OPRO, ZeroShotBaseline
from prompt_opt_harness.report import build_report

MODEL = "openai/gpt-oss-20b"
BUDGET = 40
SEED = 42
EVAL_MODELS = ["openai/gpt-oss-20b","openai/gpt-oss-120b","llama-3.3-70b-versatile","llama-3.1-8b-instant"]


def main() -> None:
    dataset = load_demo_dataset()

    def backend_factory():
        return GroqBackend(model=MODEL)
    def eval_backend_factory(Evals):
        returndict = {}
        for x in Evals:
            toinsert = {x,GroqBackend(model=x)}
            returndict.update(toinsert)
        return returndict
    methods = [
        ("Zero-shot", ZeroShotBaseline, 1),
        ("Human-written", HumanWrittenBaseline, 1),
        ("OPRO", OPRO, BUDGET),
        ("EvoPrompt-lite", EvoPromptLite, BUDGET),
    ]

    results = []
    for name, optimizer_cls, budget in methods:
        print(f"Running {name}...")
        result = run_experiment(name, optimizer_cls, backend_factory, dataset, budget=budget, seed=SEED)
        results.append(result)
        print(f"  logged to {save_run(result)}")

    print()
    print(f"Model: {MODEL} (via Groq)")
    print(build_report(results))


if __name__ == "__main__":
    main()
