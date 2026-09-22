import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import MockBackend
from prompt_opt_harness.metrics import cross_domain_variance, ood_gap, worst_case_accuracy
from prompt_opt_harness.optimizers import OPRO, ZeroShotBaseline


def backend_factory(dataset):
    return lambda: MockBackend(dataset.lookup(), source_domain=dataset.source_domain)


def test_metrics_math():
    assert ood_gap(0.9, [0.8, 0.7]) == 0.9 - 0.75
    assert ood_gap(0.9, []) == 0.0
    assert worst_case_accuracy([0.9, 0.4, 0.7]) == 0.4
    assert cross_domain_variance([0.5]) == 0.0
    assert cross_domain_variance([0.5, 0.5]) == 0.0
    print("test_metrics_math: OK")


def test_determinism():
    dataset = load_demo_dataset()
    r1 = run_experiment("OPRO", OPRO, backend_factory(dataset), dataset, budget=20, seed=1)
    r2 = run_experiment("OPRO", OPRO, backend_factory(dataset), dataset, budget=20, seed=1)
    assert r1.best_instruction == r2.best_instruction
    assert r1.source_test_acc == r2.source_test_acc
    assert r1.ood_accs == r2.ood_accs
    print("test_determinism: OK")


def test_budget_is_respected():
    dataset = load_demo_dataset()
    for budget in (10, 20, 40):
        r = run_experiment("OPRO", OPRO, backend_factory(dataset), dataset, budget=budget, seed=0)
        search_calls = r.api_calls - 22  # 22 = fixed post-hoc eval (6 test + 8 + 8 OOD)
        assert search_calls <= budget + len(dataset.source.dev), (budget, search_calls)
    print("test_budget_is_respected: OK")


def test_accuracy_in_range():
    dataset = load_demo_dataset()
    r = run_experiment("Zero-shot", ZeroShotBaseline, backend_factory(dataset), dataset, budget=1, seed=0)
    assert 0.0 <= r.source_test_acc <= 1.0
    for acc in r.ood_accs.values():
        assert 0.0 <= acc <= 1.0
    print("test_accuracy_in_range: OK")


def test_reasoning_extraction():
    from prompt_opt_harness.tasks import extract_number, REASONING_TASK

    assert extract_number("Let's see... Answer: 42") == 42.0
    assert extract_number("The total is 1,204 apples. Answer: 1,204") == 1204.0
    assert extract_number("no numbers here") is None
    assert REASONING_TASK.is_correct("blah blah Answer: 8", "8")
    assert not REASONING_TASK.is_correct("blah blah Answer: 9", "8")
    print("test_reasoning_extraction: OK")


def test_reasoning_dataset_shape():
    from prompt_opt_harness.datasets import load_demo_reasoning_dataset

    d = load_demo_reasoning_dataset()
    assert len(d.source.dev) == 6
    assert len(d.source.test) == 6
    assert set(d.ood.keys()) == {"svamp_style", "multistep"}
    for ex in d.source.dev + d.source.test:
        assert ex.label.lstrip("-").isdigit()
    print("test_reasoning_dataset_shape: OK")


if __name__ == "__main__":
    test_metrics_math()
    test_determinism()
    test_budget_is_respected()
    test_accuracy_in_range()
    test_reasoning_extraction()
    test_reasoning_dataset_shape()
    print("\nAll smoke tests passed.")
