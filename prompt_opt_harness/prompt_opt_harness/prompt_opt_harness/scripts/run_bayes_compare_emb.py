from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import MockBackend
from prompt_opt_harness.optimizers import PBT, CostAwareBayesOpt, EmbeddingBayesOpt, ZeroShotBaseline


def backend_factory_for(dataset):
    return lambda: MockBackend(dataset.lookup(), source_domain=dataset.source_domain)


def main():
    dataset = load_demo_dataset()
    backend_factory = backend_factory_for(dataset)

    experiments = [
        (ZeroShotBaseline, "ZeroShot"),
        (PBT, "PBT"),
        (CostAwareBayesOpt, "CostBayesOpt"),
        (EmbeddingBayesOpt, "EmbeddingBayesOpt"),
    ]

    seeds = [0, 1]
    budgets = [10, 20]

    results = []
    for budget in budgets:
        for cls, name in experiments:
            for seed in seeds:
                print(f"Running {name} budget={budget} seed={seed}...")
                r = run_experiment(name, cls, backend_factory, dataset, budget=budget, seed=seed)
                rd = asdict(r)
                rd.update({"optimizer_class": cls.__name__, "budget": budget, "seed": seed})
                results.append(rd)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = Path(__file__).resolve().parents[1] / 'logs' / f'bayes_emb_compare_{ts}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print('Saved results to', out)


if __name__ == '__main__':
    main()
