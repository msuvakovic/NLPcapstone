from __future__ import annotations

from prompt_opt_harness.harness import run_experiment_cross_train
from prompt_opt_harness.llm_backends import GroqBackend
from prompt_opt_harness.logger import save_run
from prompt_opt_harness.optimizers import EvoPromptLite, HumanWrittenBaseline, OPRO, ZeroShotBaseline
from prompt_opt_harness.real_datasets import load_real_dataset
from prompt_opt_harness.report import build_report

MODEL = "openai/gpt-oss-20b"
BUDGET = 40
SEED = 42
N_PER_SPLIT = 30
EVAL_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "qwen/qwen3.6-27b"]


def main() -> None:
    print(f"Loading real data ({N_PER_SPLIT} examples/split)...")
    dataset = load_real_dataset(n_per_split=N_PER_SPLIT, seed=SEED)

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
    print(f"Model: {MODEL} (via Groq) | data: SST-2 -> amazon_polarity / tweet_eval")
    print(build_report(results))


if __name__ == "__main__":
    main()
