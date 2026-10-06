from __future__ import annotations

import argparse

from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import GroqBackend, OllamaBackend
from prompt_opt_harness.logger import save_run
from prompt_opt_harness.optimizers import EvoPromptLite, HumanWrittenReasoningBaseline, OPRO, ZeroShotBaseline
from prompt_opt_harness.real_datasets_reasoning import load_real_reasoning_dataset
from prompt_opt_harness.report import build_report
from prompt_opt_harness.tasks import REASONING_TASK

DEFAULT_MODEL = {"groq": "openai/gpt-oss-20b", "ollama": "llama3.1"}
BUDGET = 180
SEED = 42
N_PER_SPLIT = 30


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["groq", "ollama"], default="groq")
    parser.add_argument("--model", default=None)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--n-per-split", type=int, default=N_PER_SPLIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model or DEFAULT_MODEL[args.backend]

    print(f"Loading real data ({args.n_per_split} examples/split)...")
    dataset = load_real_reasoning_dataset(n_per_split=args.n_per_split, seed=SEED)

    def backend_factory():
        return OllamaBackend(model=model) if args.backend == "ollama" else GroqBackend(model=model)

    methods = [
        ("Zero-shot", ZeroShotBaseline, 1),
        ("Human-written", HumanWrittenReasoningBaseline, 1),
        ("OPRO", OPRO, args.budget),
        ("EvoPrompt-lite", EvoPromptLite, args.budget),
    ]

    results = []
    for name, optimizer_cls, budget in methods:
        print(f"Running {name}...")
        result = run_experiment(name, optimizer_cls, backend_factory, dataset, budget=budget, seed=SEED, task=REASONING_TASK)
        results.append(result)
        print(f"  logged to {save_run(result)}")

    print()
    print(f"Model: {model} (via {args.backend}) | data: GSM8K -> SVAMP")
    print(build_report(results))


if __name__ == "__main__":
    main()