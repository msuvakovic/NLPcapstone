from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from statistics import mean, stdev
import sys
from pathlib import Path

# Ensure the package root is on sys.path (match tests' import strategy)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import MockBackend
from prompt_opt_harness.optimizers import OPRO, EvoPromptLite, PBT, ReflectiveOptimizer, EnsembleOptimizer, CostAwareBayesOpt
import os


def price_per_1k_for(backend_name: str, model: str | None) -> float:
    """Return an estimated USD price per 1k tokens for the given backend.

    Can be overridden with env vars like PRICE_GROQ, PRICE_OPENAI, PRICE_DEFAULT.
    These are estimates for comparison only.
    """
    env_map = {
        "GROQ": float(os.environ.get("PRICE_GROQ", "0.003")),
        "OPENAI": float(os.environ.get("PRICE_OPENAI", "0.03")),
        # default small non-zero Ollama price estimate (can be overridden)
        "OLLAMA": float(os.environ.get("PRICE_OLLAMA", "0.001")),
        "MOCK": float(os.environ.get("PRICE_MOCK", "0.0")),
    }
    if backend_name is None:
        return float(os.environ.get("PRICE_DEFAULT", "0.01"))
    key = backend_name.upper()
    return env_map.get(key, float(os.environ.get("PRICE_DEFAULT", "0.01")))


def backend_factory_for(dataset):
    return lambda: MockBackend(dataset.lookup(), source_domain=dataset.source_domain)


def main():
    dataset = load_demo_dataset()
    backend_factory = backend_factory_for(dataset)

    experiments = [
        (OPRO, "OPRO"),
        (EvoPromptLite, "EvoPrompt-lite"),
        (PBT, "PBT"),
        (ReflectiveOptimizer, "Reflective"),
        (EnsembleOptimizer, "Ensemble"),
        (CostAwareBayesOpt, "CostBayesOpt"),
    ]

    seeds = [0, 1, 2, 3, 4]
    budgets = [10, 20, 40]

    detailed_results = []
    summary = {}

    for budget in budgets:
        for cls, name in experiments:
            print(f"Running {name} with budget={budget} across seeds {seeds}...")
            per_seed = []
            for seed in seeds:
                r = run_experiment(name, cls, backend_factory, dataset, budget=budget, seed=seed)
                rd = asdict(r)
                rd["optimizer_class"] = cls.__name__
                rd["budget"] = budget
                rd["seed"] = seed
                # estimate cost (USD) from token counts and per-backend pricing
                backend_instance = backend_factory()
                backend_name = backend_instance.__class__.__name__
                model = getattr(backend_instance, "model", None)
                price_per_1k = price_per_1k_for(backend_name, model)
                tokens = rd.get("input_tokens", 0) + rd.get("output_tokens", 0)
                cost_usd = tokens / 1000.0 * price_per_1k
                rd["estimated_cost_usd"] = cost_usd
                # cost-normalized gain relative to zero-shot baseline will be computed later
                per_seed.append(rd)
                detailed_results.append(rd)

            # aggregate per (optimizer,budget)
            src_accs = [p["source_test_acc"] for p in per_seed]
            ood_gaps = [p["ood_gap"] for p in per_seed]
            summary_key = f"{name}@{budget}"
            summary[summary_key] = {
                "n_runs": len(per_seed),
                "source_test_acc_mean": mean(src_accs),
                "source_test_acc_std": stdev(src_accs) if len(src_accs) > 1 else 0.0,
                "ood_gap_mean": mean(ood_gaps),
                "ood_gap_std": stdev(ood_gaps) if len(ood_gaps) > 1 else 0.0,
            }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    logs_dir = Path(__file__).resolve().parents[1] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = logs_dir / f"compare_detailed_{ts}.json"
    summary_path = logs_dir / f"compare_summary_{ts}.json"
    with open(detailed_path, "w", encoding="utf-8") as fh:
        json.dump(detailed_results, fh, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"Saved detailed results to {detailed_path}")
    print(f"Saved summary to {summary_path}")
    for name, s in summary.items():
        print(f"{name}: src_mean={s['source_test_acc_mean']:.3f} ± {s['source_test_acc_std']:.3f}, ood_gap_mean={s['ood_gap_mean']:.3f} ± {s['ood_gap_std']:.3f}")

    # Compute cost-normalized gains (compare each optimizer@budget to Zero-shot baseline if present)
    zero_shot = [r for r in detailed_results if r.get("name", "").lower().startswith("zero")]
    if zero_shot:
        # pick same budget=1 zero-shot run mean cost and source_acc
        zs = zero_shot[0]
        zs_acc = zs.get("source_test_acc", 0.0)
        for rd in detailed_results:
            gain = rd.get("source_test_acc", 0.0) - zs_acc
            cost = rd.get("estimated_cost_usd", 0.0)
            rd["cost_normalized_gain_per_usd"] = (gain / cost) if cost else 0.0

        # write updated detailed results
        with open(detailed_path, "w", encoding="utf-8") as fh:
            json.dump(detailed_results, fh, indent=2, ensure_ascii=False)

    # Optionally run on a real dataset if GROQ_API_KEY is present (small sweep)
    if os.environ.get("GROQ_API_KEY"):
        try:
            from prompt_opt_harness.real_datasets import load_real_dataset
            from prompt_opt_harness.llm_backends import GroqBackend, OpenAICompatibleBackend, OllamaBackend

            print("\nGROQ_API_KEY detected — running a small real-data comparison (SST-2 -> OOD)...")
            real_dataset = load_real_dataset(n_per_split=30, seed=42)

            # Build a list of backend factories to try. Default: Groq if key present.
            backend_factories = []

            def make_groq_factory():
                return lambda: GroqBackend(model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"))

            backend_factories.append(("Groq", make_groq_factory()))

            # Optional: include Ollama cloud (gemma4-31b) when configured
            ollama_url = os.environ.get("OLLAMA_API_URL")
            ollama_key = os.environ.get("OLLAMA_API_KEY")
            if ollama_url and ollama_key:
                def make_ollama_factory():
                    return lambda: OpenAICompatibleBackend(model=os.environ.get("OLLAMA_MODEL", "gemma4-31b"), base_url=ollama_url, api_key_env="OLLAMA_API_KEY")

                backend_factories.append(("Ollama", make_ollama_factory()))
            else:
                # Detect a locally pulled Ollama model in the user's Downloads (e.g., 'yearn_for_mines')
                local_candidate = Path.home() / "Downloads" / "yearn_for_mines"
                if local_candidate.exists():
                    def make_local_ollama_factory():
                        # Ollama local server typically runs at http://localhost:11434
                        return lambda: OllamaBackend(model="yearn_for_mines")

                    backend_factories.append(("OllamaLocal", make_local_ollama_factory()))

            real_methods = [
                (OPRO, "OPRO"),
                (EvoPromptLite, "EvoPrompt-lite"),
                (PBT, "PBT"),
                (ReflectiveOptimizer, "Reflective"),
                (EnsembleOptimizer, "Ensemble"),
            ]

            real_results = []
            for backend_name, factory in backend_factories:
                print(f"\nRunning real-data experiments against backend: {backend_name}")
                real_backend_factory = factory
                for cls, name in real_methods:
                    print(f"Running {name} on real data with budget=40 against {backend_name}...")
                    r = run_experiment(name, cls, real_backend_factory, real_dataset, budget=40, seed=42)
                    rd = asdict(r)
                    rd["optimizer_class"] = cls.__name__
                    rd["budget"] = 40
                    rd["seed"] = 42
                    # estimate cost
                    backend_inst = real_backend_factory()
                    price_per_1k = price_per_1k_for(backend_inst.__class__.__name__, getattr(backend_inst, "model", None))
                    tokens = rd.get("input_tokens", 0) + rd.get("output_tokens", 0)
                    rd["estimated_cost_usd"] = tokens / 1000.0 * price_per_1k
                    rd["real_backend"] = backend_name
                    real_results.append(rd)

            real_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            real_out = logs_dir / f"compare_real_{real_ts}.json"
            with open(real_out, "w", encoding="utf-8") as fh:
                json.dump(real_results, fh, indent=2, ensure_ascii=False)
            print(f"Saved real-data results to {real_out}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Failed to run real-data experiments:", e)


if __name__ == "__main__":
    main()
