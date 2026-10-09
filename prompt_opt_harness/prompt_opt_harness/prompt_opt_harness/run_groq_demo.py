from __future__ import annotations

import argparse

from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import GroqBackend, OllamaBackend
from prompt_opt_harness.logger import save_run
from prompt_opt_harness.optimizers import EvoPromptLite, HumanWrittenBaseline, OPRO, ZeroShotBaseline
from prompt_opt_harness.report import build_report

DEFAULT_MODEL = {"groq": "openai/gpt-oss-20b", "ollama": "llama3.1"}
BUDGET = 180
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["groq", "ollama"], default="groq")
    parser.add_argument("--model", default=None)
    parser.add_argument("--budget", type=int, default=BUDGET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model or DEFAULT_MODEL[args.backend]
    dataset = load_demo_dataset()

    def backend_factory():
        return OllamaBackend(model=model) if args.backend == "ollama" else GroqBackend(model=model)

    methods = [
        ("Zero-shot", ZeroShotBaseline, len(dataset.source.dev)),
        ("Human-written", HumanWrittenBaseline, len(dataset.source.dev)),
        ("OPRO", OPRO, args.budget),
        ("EvoPrompt-lite", EvoPromptLite, args.budget),
    ]

    results = []
    for name, optimizer_cls, budget in methods:
        print(f"Running {name}...")
        result = run_experiment(name, optimizer_cls, backend_factory, dataset, budget=budget, seed=SEED)
        results.append(result)
        print(f"  logged to {save_run(result)}")

    print()
    print(f"Model: {model} (via {args.backend})")
    print(build_report(results))


if __name__ == "__main__":
    main()