from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import MockBackend
from prompt_opt_harness.optimizers import OPRO

dataset = load_demo_dataset()
backend = MockBackend(dataset.lookup(), source_domain=dataset.source_domain)
res = run_experiment('OPRO', OPRO, lambda: backend, dataset, budget=40, seed=42)

for c, score in res.candidate_pool:
    print(f"{score:.2f} | {c}")
