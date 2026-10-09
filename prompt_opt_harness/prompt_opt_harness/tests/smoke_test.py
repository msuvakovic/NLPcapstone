import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompt_opt_harness.datasets import load_demo_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import BudgetExceeded, BudgetedBackend, MockBackend
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
        assert r.optimization_calls <= budget, (budget, r.optimization_calls)
        assert r.api_calls == r.optimization_calls + 22  # fixed post-hoc evaluation
    print("test_budget_is_respected: OK")


def test_budgeted_backend_stops_before_overrun():
    dataset = load_demo_dataset()
    backend = MockBackend(dataset.lookup(), source_domain=dataset.source_domain)
    limited = BudgetedBackend(backend, budget=2)
    limited.generate("first")
    limited.generate("second")
    try:
        limited.generate("must not be sent")
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("expected hard budget to reject a third request")
    assert limited.calls == 2
    assert backend.stats.calls == 2


def test_deepcopied_budgeted_backend_shares_hard_cap():
    dataset = load_demo_dataset()
    backend = MockBackend(dataset.lookup(), source_domain=dataset.source_domain)
    limited = BudgetedBackend(backend, budget=1)
    clone = copy.deepcopy(limited)
    assert clone is not limited
    assert clone._state is limited._state
    assert clone._state.lock is limited._state.lock
    clone.generate("one call")
    try:
        limited.generate("over budget")
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("deep-copied adapter bypassed shared call cap")
    assert limited.calls == 1
    assert limited.rejected_calls == 1


def test_nested_budget_enforces_both_limits_and_counts_requests_once():
    dataset = load_demo_dataset()
    backend = MockBackend(dataset.lookup(), source_domain=dataset.source_domain)
    outer = BudgetedBackend(backend, budget=8)
    inner = BudgetedBackend(outer, budget=3)
    for _ in range(3):
        inner.generate("nested request")
    assert inner.calls == 3
    assert outer.calls == 3
    assert outer.total_calls == 3
    assert backend.stats.calls == 3
    inner.close()
    outer.close()


def test_budgeted_backend_disables_billable_retries_during_optimization():
    class RetryBackend:
        def __init__(self):
            self.stats = type("Stats", (), {"calls": 0})()
            self.max_retries = 5

        def generate(self, prompt):
            return str(self.max_retries)

    backend = RetryBackend()
    limited = BudgetedBackend(backend, budget=1)
    assert backend.max_retries == 5
    assert limited.generate("one") == "5"
    limited.close()
    assert backend.max_retries == 5


def test_nested_optimizer_budget_respects_parent_and_final_eval_reserve():
    dataset = load_demo_dataset()
    backend = MockBackend(dataset.lookup(), source_domain=dataset.source_domain)
    parent = BudgetedBackend(backend, budget=12)
    child = BudgetedBackend(parent, budget=6)
    for _ in range(6):
        child.generate("nested request")
    assert child.calls == 6
    assert parent.calls == 6
    assert parent.remaining_calls == 6
    parent.close()
    child.close()


def test_evolutionary_optimizers_reserve_full_candidate_evaluation():
    from prompt_opt_harness.optimizers import EvoPromptLite

    dataset = load_demo_dataset()
    for optimizer in (OPRO, EvoPromptLite):
        result = run_experiment(
            optimizer.__name__, optimizer, backend_factory(dataset), dataset, budget=20, seed=7
        )
        assert result.optimization_calls <= 20
        assert result.api_calls == result.optimization_calls + 22


def test_dspy_optimizers_obey_hard_call_budget():
    from prompt_opt_harness.optimizers import GEPA, MIPROv2

    dataset = load_demo_dataset()
    for optimizer in (GEPA, MIPROv2):
        result = run_experiment(
            optimizer.__name__, optimizer, backend_factory(dataset), dataset,
            budget=12, seed=7,
        )
        assert result.optimization_calls <= 12
        assert result.api_calls == result.optimization_calls + 22


def test_baseline_budget_counts_dev_scoring_separately_from_heldout():
    dataset = load_demo_dataset()
    result = run_experiment(
        "Zero-shot", ZeroShotBaseline, backend_factory(dataset), dataset,
        budget=len(dataset.source.dev), seed=0,
    )
    assert result.optimization_calls == len(dataset.source.dev)
    assert result.api_calls == result.optimization_calls + 22


def test_accuracy_in_range():
    dataset = load_demo_dataset()
    r = run_experiment(
        "Zero-shot", ZeroShotBaseline, backend_factory(dataset), dataset,
        budget=len(dataset.source.dev), seed=0,
    )
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
