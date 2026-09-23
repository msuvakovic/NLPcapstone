from __future__ import annotations

from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.harness import run_experiment_cross_train
from prompt_opt_harness.llm_backends import GroqBackend
from prompt_opt_harness.logger import save_run
from prompt_opt_harness.optimizers import EvoPromptLite, HumanWrittenBaseline, OPRO, ZeroShotBaseline
from prompt_opt_harness.report import build_report

MODEL = "openai/gpt-oss-20b"
BUDGET = 40
SEED = 42
EVAL_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "qwen/qwen3.6-27b"]



def main() -> None:
    dataset = load_demo_dataset()

    def backend_factory():
        return GroqBackend(model=MODEL)
    def eval_backend_factory():
        return {m: GroqBackend(model=m) for m in EVAL_MODELS}

    methods = [
        ("Zero-shot", ZeroShotBaseline, 1),
        ("Human-written", HumanWrittenBaseline, 1),
        ("OPRO", OPRO, BUDGET),
        ("EvoPrompt-lite", EvoPromptLite, BUDGET),

    ]

    results = []
    for name, optimizer_cls, budget in methods:
        print(f"Running {name}...")
        result = run_experiment_cross_train(name, optimizer_cls, backend_factory, eval_backend_factory,dataset, budget=budget, seed=SEED)
        for r in result.values():
            results.append(r)
            print(f"  logged to {save_run(r)}")
    print()
    print(f"Model: {MODEL} (via Groq)")
    print(build_report(results))


if __name__ == "__main__":
    main()
