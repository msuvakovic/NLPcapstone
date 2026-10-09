from __future__ import annotations

import argparse

from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import GroqBackend, OllamaBackend
from prompt_opt_harness.logger import save_run
from prompt_opt_harness.optimizers import EvoPromptLite, HumanWrittenReasoningBaseline, OPRO, ZeroShotBaseline, GEPA, MIPROv2, TextGrad
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
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["Zero-shot", "Human-written", "OPRO", "EvoPrompt-lite", "GEPA", "MIPROv2", "TextGrad"],
        default=None,
        help="Run only these methods; defaults to all methods available in the installed DSPy version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model or DEFAULT_MODEL[args.backend]

    print(f"Loading real data ({args.n_per_split} examples/split)...")
    dataset = load_real_reasoning_dataset(n_per_split=args.n_per_split, seed=SEED)

    def backend_factory():
        return OllamaBackend(model=model) if args.backend == "ollama" else GroqBackend(model=model)

    methods = [
        ("Zero-shot", ZeroShotBaseline, len(dataset.source.dev)),
        ("Human-written", HumanWrittenReasoningBaseline, len(dataset.source.dev)),
        ("OPRO", OPRO, args.budget),
        ("EvoPrompt-lite", EvoPromptLite, args.budget),
        ("GEPA", GEPA, args.budget),
        ("MIPROv2", MIPROv2, args.budget),
    ]

    try:
        import dspy
        has_textgrad = hasattr(dspy.teleprompt, "TextGrad")
    except ImportError:
        has_textgrad = False
    if has_textgrad:
        methods.append(("TextGrad", TextGrad, args.budget))
    else:
        print("Skipping TextGrad: installed DSPy has no native TextGrad optimizer; COPRO is not mislabeled as TextGrad.")

    if args.methods:
        requested = set(args.methods)
        unavailable = requested - {name for name, _, _ in methods}
        if unavailable:
            raise ValueError(f"Requested method(s) unavailable in this environment: {', '.join(sorted(unavailable))}")
        methods = [entry for entry in methods if entry[0] in requested]

    results = []
    failures = []
    for name, optimizer_cls, budget in methods:
        print(f"Running {name}...")
        try:
            result = run_experiment(name, optimizer_cls, backend_factory, dataset, budget=budget, seed=SEED, task=REASONING_TASK)
        except Exception as exc:
            failures.append((name, exc))
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        results.append(result)
        print(f"  logged to {save_run(result)}")
        print(f"  optimization calls: {result.optimization_calls}/{result.optimization_budget}; total calls including held-out evaluation: {result.api_calls}")
        if result.optimizer_budget_exhausted:
            print("  NOTE: optimizer search reached the hard call cap; the returned prompt may be the unoptimized starting prompt.")

    print()
    print(f"Model: {model} (via {args.backend}) | data: GSM8K -> SVAMP")
    print(build_report(results))
    if failures:
        print("\nFailed methods:")
        for name, exc in failures:
            print(f"  {name}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()