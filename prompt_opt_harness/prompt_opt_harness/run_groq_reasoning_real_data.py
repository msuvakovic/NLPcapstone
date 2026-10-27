from __future__ import annotations

import argparse

from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import GroqBackend, OllamaBackend
from prompt_opt_harness.logger import find_run, load_run, save_run
from prompt_opt_harness.optimizers import HumanWrittenReasoningBaseline, ZeroShotBaseline
from prompt_opt_harness.registry import GROUPS, OPTIMIZERS, available, resolve
from prompt_opt_harness.prompts import REASONING_BARE_INSTRUCTION
from prompt_opt_harness.real_datasets_reasoning import OOD_SETS, load_real_reasoning_dataset
from prompt_opt_harness.report import build_multiseed_report, build_report
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
    parser.add_argument("--seeds", type=int, default=1, help="number of seeds; each one draws a different sample of questions")
    parser.add_argument("--split-mode", choices=["same", "train_test"], default="same")
    parser.add_argument("--start", choices=["default", "bare"], default="default", help="bare = optimizers start from 'Solve the math word problem.'")
    parser.add_argument("--proposal-temp", type=float, default=None, help="temperature for the optimizers' proposal calls")
    parser.add_argument("--optimizers", default="opro,evoprompt",
                        help="comma-separated names and/or groups. Names: " + ", ".join(OPTIMIZERS) + ". Groups: " + ", ".join(GROUPS)
                             + ". 'oracle' tunes on a sample of the new dataset (an upper bound). Ones whose package is missing are skipped with a note.")
    parser.add_argument("--ood", default="svamp",
                        help="comma-separated new datasets to evaluate on, from: " + ", ".join(OOD_SETS)
                             + ". Two or more are needed for cross-domain variance and a meaningful worst case.")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="cap on each answer's length; answers that hit it are logged as truncated (use analyze.py to check whether that matters)")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--resume", action="store_true", help="skip runs already saved in --log-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model or DEFAULT_MODEL[args.backend]

    def backend_factory():
        if args.backend == "ollama":
            return OllamaBackend(model=model, max_tokens=args.max_tokens)
        return GroqBackend(model=model, max_tokens=args.max_tokens)

    try:
        specs = resolve(args.optimizers)
    except ValueError as exc:
        raise SystemExit(str(exc))
    runnable = []
    for spec in specs:
        ok, why = available(spec)
        if ok:
            runnable.append(spec)
        else:
            print(f"Skipping {spec.name}: {why}")
    methods = [("Zero-shot", ZeroShotBaseline, 1), ("Human-written", HumanWrittenReasoningBaseline, 1)]
    # SAPO and entropy score each prompt several times, so they get a proportionally larger budget
    methods += [(spec.name, spec.cls, args.budget * spec.budget_mult) for spec in runnable]
    ood_sets = [k.strip().lower() for k in args.ood.split(",") if k.strip()]
    bad = [k for k in ood_sets if k not in OOD_SETS]
    if bad or not ood_sets:
        raise SystemExit(f"Unknown --ood value(s): {', '.join(bad) or '(none given)'}. Choose from: {', '.join(OOD_SETS)}")
    config = {
        "dataset": "gsm8k -> " + " + ".join(ood_sets),
        "n_per_split": args.n_per_split,
        "split_mode": args.split_mode,
        "start": args.start,
    }

    by_method = {name: [] for name, _, _ in methods}
    failures = []
    for seed in range(SEED, SEED + args.seeds):
        print(f"\nSeed {seed}: loading data ({args.n_per_split} examples/split, {args.split_mode})...")
        dataset = load_real_reasoning_dataset(n_per_split=args.n_per_split, seed=seed, split_mode=args.split_mode, ood_sets=ood_sets)
        if args.start == "bare":
            dataset.base_instruction = REASONING_BARE_INSTRUCTION

        results = []
        for name, optimizer_cls, budget in methods:
            existing = find_run(name, seed, args.log_dir) if args.resume else None
            if existing:
                print(f"Skipping {name}, found {existing}")
                result = load_run(existing)
            else:
                print(f"Running {name}...")
                try:
                    result = run_experiment(
                        name, optimizer_cls, backend_factory, dataset, budget=budget, seed=seed,
                        task=REASONING_TASK, proposal_temperature=args.proposal_temp, config=config,
                    )
                except Exception as exc:  # one failing method must not lose hours of other runs; --resume retries it
                    failures.append((name, seed, f"{type(exc).__name__}: {exc}"))
                    print(f"  FAILED: {type(exc).__name__}: {exc}")
                    continue
                print(f"  logged to {save_run(result, args.log_dir)}")
            results.append(result)
            by_method[name].append(result)

        print()
        print(f"Model: {model} (via {args.backend}) | data: GSM8K -> {' + '.join(o.upper() for o in ood_sets)} | seed {seed}")
        print(build_report(results))

    if failures:
        print("\nFailed (not logged; run again with --resume to retry them):")
        for name, seed, why in failures:
            print(f"  {name} (seed {seed}): {why}")

    if args.seeds > 1:
        print()
        print(f"Mean +/- sd over {args.seeds} seeds")
        print(build_multiseed_report({k: v for k, v in by_method.items() if v}))


if __name__ == "__main__":
    main()
