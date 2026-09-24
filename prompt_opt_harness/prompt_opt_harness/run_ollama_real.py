from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompt_opt_harness.real_datasets import load_real_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.optimizers import OPRO, EvoPromptLite, PBT, ReflectiveOptimizer, EnsembleOptimizer
from prompt_opt_harness.llm_backends import OpenAICompatibleBackend, OllamaBackend


def make_ollama_factory():
    # prefer explicit cloud URL/key if set, otherwise assume local Ollama server
    url = os.environ.get("OLLAMA_API_URL")
    key = os.environ.get("OLLAMA_API_KEY")
    model = os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud")
    if url and key:
        return lambda: OpenAICompatibleBackend(model=model, base_url=url, api_key_env="OLLAMA_API_KEY")
    # local default
    return lambda: OllamaBackend(model=model)


def main():
    dataset = load_real_dataset(n_per_split=30, seed=42)
    backend_factory = make_ollama_factory()

    real_methods = [
        (OPRO, "OPRO"),
        (EvoPromptLite, "EvoPrompt-lite"),
        (PBT, "PBT"),
        (ReflectiveOptimizer, "Reflective"),
        (EnsembleOptimizer, "Ensemble"),
    ]

    results = []
    for cls, name in real_methods:
        print(f"Running {name} on real data with budget=40...")
        r = run_experiment(name, cls, backend_factory, dataset, budget=40, seed=42)
        rd = asdict(r)
        rd["optimizer_class"] = cls.__name__
        rd["budget"] = 40
        rd["seed"] = 42
        # estimate cost using model name and price override
        backend_inst = backend_factory()
        price_per_1k = float(os.environ.get("PRICE_OLLAMA", "0.0"))
        tokens = rd.get("input_tokens", 0) + rd.get("output_tokens", 0)
        rd["estimated_cost_usd"] = tokens / 1000.0 * price_per_1k
        rd["real_backend"] = "Ollama"
        results.append(rd)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = Path(__file__).resolve().parents[1] / "logs" / f"compare_real_ollama_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"Saved real-data Ollama results to {out}")


if __name__ == '__main__':
    main()
