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
from prompt_opt_harness.optimizers import CostAwareBayesOpt, EmbeddingBayesOpt, PBT, ZeroShotBaseline


def backend_factory_for(dataset):
    return lambda: MockBackend(dataset.lookup(), source_domain=dataset.source_domain)


def main():
    dataset = load_demo_dataset()
    backend_factory = backend_factory_for(dataset)

    # Hyperparameter grid
    init_random_vals = [4, 6]
    kappa_vals = [1.0, 1.96]
    candidate_pool_vals = [20, 40]

    optimizers = [
        (CostAwareBayesOpt, "CostBayesOpt"),
        (EmbeddingBayesOpt, "EmbeddingBayesOpt"),
    ]

    seeds = [0, 1]
    budget = 20

    results = []
    for cls, name in optimizers:
        for init_random in init_random_vals:
            for kappa in kappa_vals:
                for candidate_pool in candidate_pool_vals:
                    config = {"init_random": init_random, "kappa": kappa, "candidate_pool": candidate_pool}
                    print(f"Running {name} with {config} across seeds {seeds}...")
                    for seed in seeds:
                        # instantiate optimizer via run_experiment which expects a class, but we need to pass params
                        # We'll pass a small wrapper class that binds these defaults
                        def factory_cls(backend, budget=budget, seed=seed, cls=cls, init_random=init_random, kappa=kappa, candidate_pool=candidate_pool):
                            return cls(backend, budget=budget, seed=seed, init_random=init_random, kappa=kappa, candidate_pool=candidate_pool)

                        r = run_experiment(name, factory_cls, backend_factory, dataset, budget=budget, seed=seed)
                        rd = asdict(r)
                        rd.update({"optimizer_class": cls.__name__, "init_random": init_random, "kappa": kappa, "candidate_pool": candidate_pool, "seed": seed, "budget": budget})
                        results.append(rd)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = Path(__file__).resolve().parents[1] / 'logs' / f'bayes_hpo_{ts}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print('Saved HPO results to', out)


if __name__ == '__main__':
    main()
