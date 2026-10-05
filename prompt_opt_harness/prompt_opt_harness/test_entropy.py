from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import MockBackend
from prompt_opt_harness.optimizers import OPRO
from prompt_opt_harness.optimizers.entropy import create_entropy_regularized_optimizer
from prompt_opt_harness.report import build_report

dataset = load_demo_dataset()
def backend_factory():
    return MockBackend(dataset.lookup(), source_domain=dataset.source_domain)

EntropyOPRO = create_entropy_regularized_optimizer(OPRO)

methods = [
    ("OPRO", OPRO, 100),
    ("Entropy-OPRO", EntropyOPRO, 100),
]

results = []
for name, cls, budget in methods:
    res = run_experiment(name, cls, backend_factory, dataset, budget, seed=42)
    results.append(res)

print(build_report(results))
