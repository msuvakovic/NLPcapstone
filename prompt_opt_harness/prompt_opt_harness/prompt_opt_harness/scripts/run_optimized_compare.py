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


def make_factory(cls, init_random, kappa, candidate_pool):
    # returns a callable that when called like a class returns an optimizer instance
    def factory(backend, budget, seed):
        return cls(backend, budget=budget, seed=seed, init_random=init_random, kappa=kappa, candidate_pool=candidate_pool)
    return factory


def main():
    dataset = load_demo_dataset()
    backend_factory = backend_factory_for(dataset)

    # Optimized configs (from HPO)
    cost_cfg = {"init_random": 6, "kappa": 1.0, "candidate_pool": 40}
    emb_cfg = {"init_random": 6, "kappa": 1.0, "candidate_pool": 40}

    experiments = [
        (ZeroShotBaseline, "ZeroShot"),
        (PBT, "PBT"),
        (lambda b, budget, seed: CostAwareBayesOpt(b, budget=budget, seed=seed, **cost_cfg), "CostBayesOpt_opt"),
        (lambda b, budget, seed: EmbeddingBayesOpt(b, budget=budget, seed=seed, **emb_cfg), "EmbeddingBayesOpt_opt"),
    ]

    seeds = [0, 1, 2, 3, 4]
    budgets = [10, 20, 40]

    results = []
    for budget in budgets:
        for cls, name in experiments:
            print(f"Running {name} with budget={budget} across seeds {seeds}...")
            for seed in seeds:
                r = run_experiment(name, cls, backend_factory, dataset, budget=budget, seed=seed)
                rd = asdict(r)
                rd.update({"optimizer_class": rd.get("name", name), "budget": budget, "seed": seed})
                results.append(rd)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = Path(__file__).resolve().parents[1] / 'logs' / f'optimized_compare_{ts}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print('Saved optimized comparison to', out)


if __name__ == '__main__':
    main()
