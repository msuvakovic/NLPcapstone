from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join("prompt_opt_harness", "prompt_opt_harness")))

from prompt_opt_harness.real_datasets import load_real_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.optimizers import OPRO, EvoPromptLite, PBT, ReflectiveOptimizer, EnsembleOptimizer
from prompt_opt_harness.llm_backends import OpenAICompatibleBackend, OllamaBackend


def make_ollama_factory():
    url = (os.environ.get("OLLAMA_API_URL") or "").strip()
    key = (os.environ.get("OLLAMA_API_KEY") or "").strip()
    model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    use_remote = os.environ.get("OLLAMA_USE_REMOTE", "0").strip().lower() in {"1", "true", "yes", "y"}
    if use_remote and url and key:
        return lambda: OpenAICompatibleBackend(model=model, base_url=url, api_key_env="OLLAMA_API_KEY")
    return lambda: OllamaBackend(model=model)


def run_sweep(seeds=(0, 1, 2, 3, 4), budget=40):
    backend_factory = make_ollama_factory()
    real_methods = [
        (OPRO, "OPRO"),
        (EvoPromptLite, "EvoPrompt-lite"),
        (PBT, "PBT"),
        (ReflectiveOptimizer, "Reflective"),
        (EnsembleOptimizer, "Ensemble"),
    ]

    all_results = []
    for seed in seeds:
        dataset = load_real_dataset(n_per_split=30, seed=seed)
        for cls, name in real_methods:
            print(f"Running {name} seed={seed} budget={budget}...")
            r = run_experiment(name, cls, backend_factory, dataset, budget=budget, seed=seed)
            rd = asdict(r)
            rd["optimizer_class"] = cls.__name__
            rd["budget"] = budget
            rd["seed"] = seed
            backend_inst = backend_factory()
            price_per_1k = float(os.environ.get("PRICE_OLLAMA", "0.0"))
            tokens = rd.get("input_tokens", 0) + rd.get("output_tokens", 0)
            rd["estimated_cost_usd"] = tokens / 1000.0 * price_per_1k
            rd["real_backend"] = "Ollama"
            all_results.append(rd)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = Path(__file__).resolve().parents[1] / "prompt_opt_harness" / "logs" / f"compare_real_ollama_multiseed_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    print(f"Saved multi-seed results to {out}")


if __name__ == '__main__':
    # seeds and budget can be overridden via env vars
    seeds_env = os.environ.get("MULTI_SEED_LIST")
    if seeds_env:
        seeds = [int(s) for s in seeds_env.split(",")]
    else:
        seeds = list(range(5))
    budget = int(os.environ.get("MULTI_SEED_BUDGET", "40"))
    run_sweep(seeds=seeds, budget=budget)
