import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompt_opt_harness.datasets import Dataset, DomainSplit, load_demo_dataset
from prompt_opt_harness.harness import run_experiment
from prompt_opt_harness.llm_backends import MockBackend
from prompt_opt_harness.logger import save_run
from prompt_opt_harness.optimizers import RandomSearch, ZeroShotBaseline


def backend_factory(dataset, calls=None):
    def factory():
        if calls is not None:
            calls.append("constructed")
        return MockBackend(dataset.lookup(), source_domain=dataset.source_domain)
    return factory


def test_duplicate_text_is_rejected_before_backend_construction():
    dataset = load_demo_dataset()
    duplicate = replace(dataset.source.test[0], text=dataset.source.dev[0].text)
    invalid = Dataset(
        source_domain=dataset.source_domain,
        source=DomainSplit(dataset.source.dev, [duplicate] + dataset.source.test[1:]),
        ood=dataset.ood,
        base_instruction=dataset.base_instruction,
    )
    constructed = []

    with pytest.raises(ValueError, match="duplicate text"):
        run_experiment(
            "Zero-shot", ZeroShotBaseline, backend_factory(invalid, constructed),
            invalid, budget=len(invalid.source.dev),
        )
    assert constructed == []


def test_duplicate_id_and_wrong_domain_are_rejected():
    dataset = load_demo_dataset()
    duplicate_id = replace(dataset.source.test[0], id=dataset.source.dev[0].id)
    invalid_id = Dataset(
        dataset.source_domain,
        DomainSplit(dataset.source.dev, [duplicate_id] + dataset.source.test[1:]),
        dataset.ood,
        dataset.base_instruction,
    )
    with pytest.raises(ValueError, match="duplicate ID"):
        run_experiment(
            "Zero-shot", ZeroShotBaseline, backend_factory(invalid_id), invalid_id,
            budget=len(invalid_id.source.dev),
        )

    wrong_domain = replace(dataset.ood["amazon"][0], domain="tweets")
    invalid_domain = Dataset(
        dataset.source_domain,
        dataset.source,
        {**dataset.ood, "amazon": [wrong_domain] + dataset.ood["amazon"][1:]},
        dataset.base_instruction,
    )
    with pytest.raises(ValueError, match="expected 'amazon'"):
        run_experiment(
            "Zero-shot", ZeroShotBaseline, backend_factory(invalid_domain),
            invalid_domain, budget=len(invalid_domain.source.dev),
        )


def test_random_search_is_seeded_and_respects_budget():
    dataset = load_demo_dataset()
    first = run_experiment(
        "Random-search", RandomSearch, backend_factory(dataset), dataset,
        budget=20, seed=9,
    )
    second = run_experiment(
        "Random-search", RandomSearch, backend_factory(dataset), dataset,
        budget=20, seed=9,
    )
    assert first.best_instruction == second.best_instruction
    assert first.candidate_pool == second.candidate_pool
    assert first.optimization_calls <= 20
    assert first.api_calls == first.optimization_calls + 22
    assert len(first.candidate_pool) >= 2


def test_log_metadata_labels_approximate_usage(tmp_path):
    dataset = load_demo_dataset()
    result = run_experiment(
        "Zero-shot", ZeroShotBaseline, backend_factory(dataset), dataset,
        budget=len(dataset.source.dev), seed=17,
    )
    path = save_run(result, str(tmp_path))
    payload = json.loads(Path(path).read_text())

    assert payload["seed"] == 17
    assert payload["valid"] is True
    assert payload["stop_reason"] == "completed"
    assert payload["input_tokens"] is None
    assert payload["output_tokens"] is None
    assert payload["usage_source"] == "unavailable"
    assert payload["approx_input_words"] > 0
    assert payload["approx_count_method"] == "whitespace_split_approximation"