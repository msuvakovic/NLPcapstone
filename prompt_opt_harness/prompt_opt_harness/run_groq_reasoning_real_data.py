from __future__ import annotations

from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import GroqBackend
from prompt_opt_harness.logger import save_run
from prompt_opt_harness.optimizers import EvoPromptLite, HumanWrittenReasoningBaseline, OPRO, ZeroShotBaseline
from prompt_opt_harness.real_datasets_reasoning import load_real_reasoning_dataset
from prompt_opt_harness.report import build_report
from prompt_opt_harness.tasks import REASONING_TASK

MODEL = "openai/gpt-oss-20b"
BUDGET = 180
SEED = 42
N_PER_SPLIT = 30


def main() -> None:
    print(f"Loading real data ({N_PER_SPLIT} examples/split)...")
    dataset = load_real_reasoning_dataset(n_per_split=N_PER_SPLIT, seed=SEED)

    def backend_factory():
        return GroqBackend(model=MODEL)

    methods = [
        ("Zero-shot", ZeroShotBaseline, 1),
        ("Human-written", HumanWrittenReasoningBaseline, 1),
        ("OPRO", OPRO, BUDGET),
        ("EvoPrompt-lite", EvoPromptLite, BUDGET),
    ]

    results = []
    for name, optimizer_cls, budget in methods:
        print(f"Running {name}...")
        result = run_experiment(name, optimizer_cls, backend_factory, dataset, budget=budget, seed=SEED, task=REASONING_TASK)
        results.append(result)
        print(f"  logged to {save_run(result)}")

    print()
    print(f"Model: {MODEL} (via Groq) | data: GSM8K -> SVAMP")
    print(build_report(results))


if __name__ == "__main__":
    main()
