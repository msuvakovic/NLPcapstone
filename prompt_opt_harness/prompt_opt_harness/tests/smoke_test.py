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
        assert r.search_calls <= budget, (budget, r.search_calls)
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


class ScriptedReasoningBackend:
    # offline stand-in for a real model on the reasoning task: right 70% of the time,
    # cuts off ~15% of answers before the Answer line
    def __init__(self, lookup):
        from prompt_opt_harness.llm_backends import LLMCallStats
        self.stats = LLMCallStats()
        self.lookup = lookup
        self.last_meta = {}
        self.n_proposals = 0

    def info(self):
        return {"backend": "ScriptedReasoningBackend", "model": "scripted", "temperature": 0.0, "max_tokens": 512}

    def generate(self, prompt, temperature=None, seed=None):
        import hashlib
        import re
        match = re.search(r"Question: (.*)\nEnd your response", prompt, re.S)
        if not match:
            self.n_proposals += 1
            text = f"Solve the problem step by step, variant {self.n_proposals}."
            self.last_meta = {"finish_reason": "stop", "output_tokens": 12}
        else:
            gold = int(self.lookup[match.group(1)])
            r = int(hashlib.sha256((match.group(1) + prompt).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            if r < 0.15:
                text = "First we compute the first part and then"
                self.last_meta = {"finish_reason": "length", "output_tokens": 512}
            else:
                value = gold if r < 0.75 else gold + 1
                text = f"Working it out.\nAnswer: {value}"
                self.last_meta = {"finish_reason": "stop", "output_tokens": 40}
        self.stats.record(prompt, text, 0.01, output_tokens=self.last_meta["output_tokens"])
        return text


def _reasoning_setup():
    from prompt_opt_harness.datasets import load_demo_reasoning_dataset

    dataset = load_demo_reasoning_dataset()
    lookup = _lookup_for(dataset)
    return dataset, (lambda: ScriptedReasoningBackend(lookup))


def test_detailed_records_and_truncation():
    from prompt_opt_harness.tasks import REASONING_TASK

    dataset, factory = _reasoning_setup()
    r = run_experiment("OPRO", OPRO, factory, dataset, budget=60, seed=3, task=REASONING_TASK)
    assert set(r.eval_records) == {"source_test", "ood:svamp_style", "ood:multistep"}
    rec = r.eval_records["source_test"][0]
    for key in ("id", "question", "gold", "response", "predicted", "correct", "well_formed", "finish_reason", "truncated"):
        assert key in rec, key
    flat = [x for rs in r.eval_records.values() for x in rs]
    assert any(x["truncated"] for x in flat)
    assert all(x["well_formed"] is False for x in flat if x["truncated"])
    assert r.eval_summary["n"] == len(flat)
    assert len(r.candidate_stats) == len(r.candidate_pool)
    assert all("calls_after" in c and "n_truncated" in c for c in r.candidate_stats)
    assert r.search_calls < r.api_calls
    assert r.config["model"] == "scripted" and r.config["n_dev"] == 6
    print("test_detailed_records_and_truncation: OK")


def test_log_roundtrip_and_reports():
    import json
    import tempfile

    from prompt_opt_harness.logger import find_run, load_run, save_run
    from prompt_opt_harness.report import build_multiseed_report, build_report
    from prompt_opt_harness.tasks import REASONING_TASK

    dataset, factory = _reasoning_setup()
    results = [
        run_experiment("Zero-shot", ZeroShotBaseline, factory, dataset, budget=1, seed=1, task=REASONING_TASK),
        run_experiment("OPRO", OPRO, factory, dataset, budget=60, seed=1, task=REASONING_TASK),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = save_run(results[1], tmp)
        payload = json.load(open(path))
        assert payload["schema_version"] == 2 and payload["seed"] == 1
        assert payload["eval_records"]["source_test"] and payload["dev_records"]
        assert find_run("OPRO", 1, tmp) == path and find_run("OPRO", 2, tmp) is None
        loaded = load_run(path)
        assert loaded.source_test_acc == results[1].source_test_acc
        assert loaded.candidate_pool == results[1].candidate_pool
    table = build_report(results)
    assert "Dev-Test" in table
    multi = build_multiseed_report({"Zero-shot": [results[0], results[0]], "OPRO": [results[1], results[1]]})
    assert "+/-" in multi
    print("test_log_roundtrip_and_reports: OK")


def test_proposal_temperature_reaches_backend():
    class Spy(ScriptedReasoningBackend):
        def __init__(self, lookup):
            super().__init__(lookup)
            self.temps = []

        def generate(self, prompt, temperature=None, seed=None):
            if "Question:" not in prompt:
                self.temps.append((temperature, seed))
            return super().generate(prompt, temperature, seed)

    from prompt_opt_harness.tasks import REASONING_TASK

    dataset, _ = _reasoning_setup()
    lookup = {ex.text: ex.label for ex in dataset.source.dev + dataset.source.test + dataset.source.confirm}
    for examples in dataset.ood.values():
        lookup.update({ex.text: ex.label for ex in examples})
    spy = Spy(lookup)
    run_experiment("OPRO", OPRO, lambda: spy, dataset, budget=40, seed=5, task=REASONING_TASK, proposal_temperature=0.8)
    assert spy.temps and all(t == 0.8 for t, _ in spy.temps)
    assert len({s for _, s in spy.temps}) == len(spy.temps)
    print("test_proposal_temperature_reaches_backend: OK")


def test_same_split_loader():
    import random
    import types

    class FakeDS:
        def __init__(self, rows):
            self.rows = rows

        def shuffle(self, seed):
            rows = list(self.rows)
            random.Random(seed).shuffle(rows)
            return FakeDS(rows)

        def select(self, indices):
            return FakeDS([self.rows[i] for i in indices])

        def __iter__(self):
            return iter(self.rows)

        def __len__(self):
            return len(self.rows)

    def fake_load_dataset(name, config, split):
        if name == "openai/gsm8k":
            n = 200 if split == "train" else 100
            return FakeDS([{"question": f"{split} q{i}", "answer": f"work #### {i}"} for i in range(n)])
        return FakeDS([{"question": f"svamp q{i}", "result": i} for i in range(100)])

    fake = types.ModuleType("datasets")
    fake.load_dataset = fake_load_dataset
    saved = sys.modules.get("datasets")
    sys.modules["datasets"] = fake
    try:
        from prompt_opt_harness.real_datasets_reasoning import load_real_reasoning_dataset

        same = load_real_reasoning_dataset(n_per_split=30, seed=1, split_mode="same")
        dev, test = {e.text for e in same.source.dev}, {e.text for e in same.source.test}
        assert len(dev) == len(test) == 30 and not dev & test
        assert all(t.startswith("test ") for t in dev | test)
        confirm = {e.text for e in same.source.confirm}
        assert len(confirm) == 30 and not confirm & (dev | test), "confirmation questions are separate from dev and test"

        legacy = load_real_reasoning_dataset(n_per_split=30, seed=1, split_mode="train_test")
        assert all(e.text.startswith("train ") for e in legacy.source.dev)
        assert all(e.text.startswith("test ") for e in legacy.source.test)

        test_texts = {e.text for e in same.ood["svamp"]}
        val_texts = {e.text for e in same.ood_val["svamp"]}
        assert len(test_texts) == len(val_texts) == 30 and not test_texts & val_texts
        rows = [f"svamp q{i}" for i in range(100)]
        random.Random(1).shuffle(rows)
        assert test_texts == set(rows[:30]), "reported new-dataset questions must not change"

        other = load_real_reasoning_dataset(n_per_split=30, seed=2, split_mode="same")
        assert {e.text for e in other.source.dev} != dev
    finally:
        if saved is None:
            del sys.modules["datasets"]
        else:
            sys.modules["datasets"] = saved
    print("test_same_split_loader: OK")


def test_duplicate_candidates_keep_their_own_stats():
    class Repeater(ScriptedReasoningBackend):
        def generate(self, prompt, temperature=None, seed=None):
            if "Question:" not in prompt:
                self.last_meta = {"finish_reason": "stop", "output_tokens": 5}
                self.stats.record(prompt, "same", 0.01, output_tokens=5)
                return "Solve it the same way every time."
            return super().generate(prompt, temperature, seed)

    from prompt_opt_harness.tasks import REASONING_TASK

    dataset, _ = _reasoning_setup()
    lookup = {ex.text: ex.label for ex in dataset.source.dev + dataset.source.test + dataset.source.confirm}
    for examples in dataset.ood.values():
        lookup.update({ex.text: ex.label for ex in examples})
    r = run_experiment("OPRO", OPRO, lambda: Repeater(lookup), dataset, budget=80, seed=1, task=REASONING_TASK)
    calls = [c["calls_after"] for c in r.candidate_stats]
    assert len(calls) >= 3 and calls == sorted(set(calls)), calls
    print("test_duplicate_candidates_keep_their_own_stats: OK")


def _lookup_for(dataset):
    lookup = {ex.text: ex.label for ex in dataset.source.dev + dataset.source.test + dataset.source.confirm}
    for rows in list(dataset.ood.values()) + list(dataset.ood_val.values()):
        lookup.update({ex.text: ex.label for ex in rows})
    return lookup


ALL_OPTIMIZERS = None


def _all_optimizers():
    from prompt_opt_harness.optimizers import EvoPromptLite, GEPALite, OPRO, OracleOPRO, RandomSearch, TextGradLite

    return [OPRO, EvoPromptLite, GEPALite, TextGradLite, RandomSearch, OracleOPRO]


def _registry_optimizers():
    """Every registry optimizer that runs without DSPy and has its packages installed, with its budget multiplier."""
    from prompt_opt_harness.registry import OPTIMIZERS, available

    return [s for s in OPTIMIZERS.values() if s.group != "dspy" and available(s)[0]]


def test_every_optimizer_stays_within_budget():
    from prompt_opt_harness.tasks import REASONING_TASK

    dataset, factory = _reasoning_setup()
    for cls in _all_optimizers():
        for budget in (14, 30, 61, 100):
            r = run_experiment(cls.name, cls, factory, dataset, budget=budget, seed=2, task=REASONING_TASK)
            assert r.search_calls <= budget, (cls.name, budget, r.search_calls)
    print("test_every_optimizer_stays_within_budget: OK")


def test_new_optimizers_log_candidates_and_use_failures():
    from prompt_opt_harness.optimizers import GEPALite, RandomSearch, TextGradLite
    from prompt_opt_harness.tasks import REASONING_TASK

    dataset, _ = _reasoning_setup()
    lookup = _lookup_for(dataset)

    class Spy(ScriptedReasoningBackend):
        def __init__(self, lookup):
            super().__init__(lookup)
            self.proposals = []

        def generate(self, prompt, temperature=None, seed=None):
            if "Question:" not in prompt:
                self.proposals.append(prompt)
            return super().generate(prompt, temperature, seed)

    for cls in (GEPALite, TextGradLite, RandomSearch):
        spy = Spy(lookup)
        r = run_experiment(cls.name, cls, lambda: spy, dataset, budget=120, seed=4, task=REASONING_TASK)
        assert len(r.candidate_pool) >= 3, (cls.name, len(r.candidate_pool))
        assert len(r.candidate_stats) == len(r.candidate_pool)
        assert r.dev_score == max(score for _, score in r.candidate_pool)
        assert r.config["tuned_on"] == "source tuning set"
        if cls is RandomSearch:
            assert not any("Correct answer:" in p for p in spy.proposals)
        else:
            assert any("Correct answer:" in p for p in spy.proposals), cls.name
    print("test_new_optimizers_log_candidates_and_use_failures: OK")


def test_random_search_is_reproducible():
    from prompt_opt_harness.optimizers import RandomSearch
    from prompt_opt_harness.tasks import REASONING_TASK

    dataset, factory = _reasoning_setup()
    a = run_experiment("Random-search", RandomSearch, factory, dataset, budget=80, seed=7, task=REASONING_TASK)
    b = run_experiment("Random-search", RandomSearch, factory, dataset, budget=80, seed=7, task=REASONING_TASK)
    assert a.candidate_pool == b.candidate_pool
    print("test_random_search_is_reproducible: OK")


def test_check_splits_rejects_overlap():
    import dataclasses

    from prompt_opt_harness.datasets import DomainSplit, load_demo_reasoning_dataset
    from prompt_opt_harness.harness import check_splits

    clean = load_demo_reasoning_dataset()
    check_splits(clean)

    leaky_test = dataclasses.replace(clean, ood={**clean.ood, "svamp_style": clean.ood["svamp_style"] + clean.source.dev[:1]})
    try:
        check_splits(leaky_test)
        raise AssertionError("tuning question inside a test set was not caught")
    except ValueError as exc:
        assert "Data leak" in str(exc)

    shouted = clean.source.dev[0]
    leaky_val = dataclasses.replace(clean, ood_val={**clean.ood_val, "multistep": clean.ood_val["multistep"] + [
        dataclasses.replace(clean.ood["multistep"][0])]})
    try:
        check_splits(leaky_val)
        raise AssertionError("oracle tuning sample overlapping a test set was not caught")
    except ValueError as exc:
        assert "Data leak" in str(exc)

    # spacing and case differences must not hide a duplicate
    sneaky = dataclasses.replace(clean, source=DomainSplit(
        dev=clean.source.dev, test=clean.source.test + [dataclasses.replace(shouted, text="  " + shouted.text.upper() + " ")]))
    try:
        check_splits(sneaky)
        raise AssertionError("re-cased duplicate was not caught")
    except ValueError:
        pass
    print("test_check_splits_rejects_overlap: OK")


def test_only_the_oracle_sees_the_new_dataset():
    from prompt_opt_harness.tasks import REASONING_TASK

    dataset, _ = _reasoning_setup()
    lookup = _lookup_for(dataset)
    test_texts = {ex.text for rows in dataset.ood.values() for ex in rows} | {ex.text for ex in dataset.source.test}
    val_texts = {ex.text for rows in dataset.ood_val.values() for ex in rows}

    for cls in _all_optimizers():
        seen = []

        class Spy(ScriptedReasoningBackend):
            def generate(self, prompt, temperature=None, seed=None):
                seen.append(prompt)
                return super().generate(prompt, temperature, seed)

        r = run_experiment(cls.name, cls, lambda: Spy(lookup), dataset, budget=80, seed=1, task=REASONING_TASK)
        during_search = seen[: r.search_calls]
        assert not any(t in p for p in during_search for t in test_texts), f"{cls.name} saw a reported test question while tuning"
        saw_val = any(t in p for p in during_search for t in val_texts)
        assert saw_val == cls.tune_on_target, (cls.name, saw_val)
        if cls.tune_on_target:
            assert r.config["tuned_on"].startswith("new dataset") and r.config["n_tuning"] == len(val_texts)
        assert r.config["n_dev"] == len(dataset.source.dev)
    print("test_only_the_oracle_sees_the_new_dataset: OK")


def test_oracle_needs_a_tuning_sample():
    import dataclasses

    from prompt_opt_harness.optimizers import OracleOPRO
    from prompt_opt_harness.tasks import REASONING_TASK

    dataset, factory = _reasoning_setup()
    bare = dataclasses.replace(dataset, ood_val={})
    try:
        run_experiment("Oracle-OPRO", OracleOPRO, factory, bare, budget=40, seed=0, task=REASONING_TASK)
        raise AssertionError("oracle ran without a tuning sample")
    except ValueError as exc:
        assert "ood_val" in str(exc)
    print("test_oracle_needs_a_tuning_sample: OK")


# ----------------------------------------------------------------------------- analyze.py
def _payload(name, seed, src, ood, truncated=(), max_tokens=512, oracle=False, schema=2):
    """Hand-built log: src / ood are lists of 0/1 per question; `truncated` are indices of the pooled list that were cut off."""
    def recs(prefix, flags, offset):
        out = []
        for i, c in enumerate(flags):
            cut = (offset + i) in truncated
            out.append({"id": f"{prefix}{i}", "domain": "d", "question": "q", "gold": "1", "response": "r", "predicted": None if cut else "1",
                        "correct": bool(c) and not cut, "well_formed": not cut, "finish_reason": "length" if cut else "stop",
                        "truncated": cut, "output_tokens": max_tokens if cut else 40})
        return out
    s_recs, o_recs = recs("t", src, 0), recs("o", ood, len(src))
    sa = sum(r["correct"] for r in s_recs) / len(s_recs)
    oa = sum(r["correct"] for r in o_recs) / len(o_recs)
    return {"schema_version": schema, "timestamp": f"20261020T0000{seed:02d}Z", "name": name, "seed": seed,
            "config": {"model": "fake", "max_tokens": max_tokens, "tuned_on": "new dataset (oracle)" if oracle else "source tuning set"},
            "dev_score": sa + 0.1, "source_test_acc": sa, "dev_minus_test": 0.1, "ood_accs": {"x": oa}, "ood_gap": sa - oa,
            "worst_case_acc": oa, "variance": 0.0, "api_calls": 50, "input_tokens": 100, "output_tokens": 100,
            "eval_records": {"source_test": s_recs, "ood:x": o_recs} if schema >= 2 else {}}


def _write_logs(folder, payloads):
    import json, os
    os.makedirs(folder, exist_ok=True)
    for i, p in enumerate(payloads):
        with open(os.path.join(folder, f"{p['name'].lower()}_s{p['seed']}_{i}.json"), "w") as f:
            json.dump(p, f)


def _analyze(logs, out, compare=None):
    import argparse
    import analyze
    return analyze.build(argparse.Namespace(logs=logs, out=out, compare=compare, quiet=True))


def test_analyze_labels_transfer_and_overfitting():
    import csv, os, tempfile
    d = tempfile.mkdtemp()
    half = [1, 0] * 30          # 60 questions at 50%
    high = [1] * 54 + [0] * 6   # 90%
    payloads = []
    for seed in (1, 2):
        payloads += [
            _payload("Zero-shot", seed, half, half),
            _payload("Human-written", seed, half, half),
            _payload("OPRO", seed, high, high),            # gains everywhere
            _payload("GEPA-lite", seed, high, half),       # gains only on the source
            _payload("TextGrad-lite", seed, half, half),   # no gain
            _payload("Oracle-OPRO", seed, high, high, oracle=True),
        ]
    _write_logs(d + "/logs", payloads)
    text = _analyze(d + "/logs", d + "/out")
    rows = {l.split("|")[1].strip(): l for l in text.splitlines() if l.startswith("| ") and ("OVERFITS" in l or "transfers" in l or "no reliable" in l or "oracle" in l)}
    assert "gain transfers" in rows["OPRO"], rows["OPRO"]
    assert "OVERFITS" in rows["GEPA-lite"], rows["GEPA-lite"]
    assert "no reliable gain" in rows["TextGrad-lite"], rows["TextGrad-lite"]
    assert "oracle" in rows["Oracle-OPRO"]
    # numbers in the csv are the log values, not re-estimates
    with open(d + "/out/results.csv") as f:
        by = {r["method"]: r for r in csv.DictReader(f)}
    assert abs(float(by["OPRO"]["source_test"]) - 0.9) < 1e-9 and abs(float(by["GEPA-lite"]["new_data_acc"]) - 0.5) < 1e-9
    assert os.path.exists(d + "/out/fig_gap.svg") and "<svg" in open(d + "/out/fig_gap.svg").read()
    print("test_analyze_labels_transfer_and_overfitting: OK")


def test_analyze_truncation_verdict_and_cap_comparison():
    import tempfile
    d = tempfile.mkdtemp()
    half = [1, 0] * 30
    cut = set(range(0, 24))     # 20% of the 120 pooled questions cut off, all in the first (source) half
    low = [_payload(n, 1, [1] * 60, [1] * 60, truncated=cut, max_tokens=512) for n in ("Zero-shot", "OPRO")]
    high = [_payload(n, 1, [1] * 60, [1] * 60, max_tokens=1024) for n in ("Zero-shot", "OPRO")]
    _write_logs(d + "/low", low)
    _write_logs(d + "/high", high)
    t = _analyze(d + "/low", d + "/o1")
    assert "Verdict: CONFOUNDED" in t, "20% truncation must be flagged"
    t = _analyze(d + "/high", d + "/o2")
    assert "Verdict: OK" in t
    t = _analyze(d + "/low", d + "/o3", compare=d + "/high")
    assert "Token-cap sensitivity (512 vs 1024)" in t and "20.0% → 0.0%" in t
    assert "changes at least one gap" in t
    print("test_analyze_truncation_verdict_and_cap_comparison: OK")


def test_analyze_skips_old_logs_and_keeps_newest():
    import tempfile
    d = tempfile.mkdtemp()
    half = [1, 0] * 30
    old = _payload("Zero-shot", 1, half, half, schema=1)
    older = _payload("Zero-shot", 2, [0] * 60, [0] * 60); older["timestamp"] = "20250101T000000Z"
    newer = _payload("Zero-shot", 2, half, half); newer["timestamp"] = "20261020T000000Z"
    _write_logs(d + "/l", [old, older, newer])
    import analyze
    runs = analyze.load_runs(d + "/l")
    assert list(runs["Zero-shot"]) == [2], "schema-1 log is skipped"
    assert abs(runs["Zero-shot"][2]["source_test_acc"] - 0.5) < 1e-9, "newest log for (method, seed) wins"
    print("test_analyze_skips_old_logs_and_keeps_newest: OK")


def test_second_new_dataset_loaders():
    import random
    import types

    class FakeDS:
        def __init__(self, rows):
            self.rows = rows
        def shuffle(self, seed):
            rows = list(self.rows); random.Random(seed).shuffle(rows); return FakeDS(rows)
        def select(self, idx):
            return FakeDS([self.rows[i] for i in idx])
        def __iter__(self):
            return iter(self.rows)
        def __len__(self):
            return len(self.rows)

    def fake_load_dataset(name, config=None, split=None):
        if name == "openai/gsm8k":
            return FakeDS([{"question": f"{split} q{i}", "answer": f"work #### {i}"} for i in range(200)])
        if name == "MU-NLPC/Calc-svamp":
            return FakeDS([{"question": f"svamp q{i}", "result": i} for i in range(100)])
        if name == "qintongli/GSM-Plus":   # every 4th row is a 'critical thinking' row with no numeric answer
            return FakeDS([{"question": f"plus q{i}", "answer": "None" if i % 4 == 0 else f"{i},000"} for i in range(200)])
        if name == "EleutherAI/asdiv":
            return FakeDS([{"body": f"body{i}.", "question": f"how many {i}?", "answer": f"{i} (apples)"} for i in range(100)])
        if name == "ChilleD/MultiArith":
            return FakeDS([{"question": f"ma q{i}", "final_ans": str(i)} for i in range(100)])
        raise AssertionError(name)

    fake = types.ModuleType("datasets")
    fake.load_dataset = fake_load_dataset
    saved = sys.modules.get("datasets")
    sys.modules["datasets"] = fake
    try:
        from prompt_opt_harness.real_datasets_reasoning import load_real_reasoning_dataset

        d = load_real_reasoning_dataset(n_per_split=20, seed=3, ood_sets=("svamp", "gsmplus", "asdiv", "multiarith"))
        assert set(d.ood) == set(d.ood_val) == {"svamp", "gsmplus", "asdiv", "multiarith"}
        for name in d.ood:
            assert len(d.ood[name]) == len(d.ood_val[name]) == 20
            assert not {e.text for e in d.ood[name]} & {e.text for e in d.ood_val[name]}, name
            assert all(e.domain == name for e in d.ood[name] + d.ood_val[name])
        assert all("None" != e.label and "," not in e.label for e in d.ood["gsmplus"] + d.ood_val["gsmplus"])
        assert all(e.label.isdigit() for e in d.ood["asdiv"]), "'9 (apples)' becomes '9'"
        assert all(e.text.startswith("body") for e in d.ood["asdiv"]), "ASDiv question = body + question"
        # the original single-dataset call is unchanged
        one = load_real_reasoning_dataset(n_per_split=20, seed=3)
        assert set(one.ood) == {"svamp"} and [e.text for e in one.ood["svamp"]] == [e.text for e in d.ood["svamp"]]
        # the leak check still passes with several new datasets
        from prompt_opt_harness.harness import check_splits
        check_splits(d)
        try:
            load_real_reasoning_dataset(n_per_split=20, seed=3, ood_sets=("nope",))
            raise AssertionError("unknown dataset accepted")
        except ValueError as exc:
            assert "Unknown new dataset" in str(exc)
        try:
            load_real_reasoning_dataset(n_per_split=60, seed=3, ood_sets=("asdiv",))
            raise AssertionError("too-small dataset accepted")
        except RuntimeError as exc:
            assert "usable rows" in str(exc)
    finally:
        if saved is None:
            del sys.modules["datasets"]
        else:
            sys.modules["datasets"] = saved
    print("test_second_new_dataset_loaders: OK")


# ----------------------------------------------------------------------------- ported optimizers, guard, DSPy
def test_registry_groups_and_missing_packages():
    from dataclasses import replace
    from prompt_opt_harness.registry import GROUPS, OPTIMIZERS, available, resolve

    assert [s.key for s in resolve("core")] == ["opro", "evoprompt", "gepa", "textgrad", "random", "oracle"]
    assert {s.key for s in resolve("all")} == set(OPTIMIZERS)
    assert [s.key for s in resolve("opro, opro ,gepa")] == ["opro", "gepa"], "duplicates collapse, order kept"
    assert len(resolve("core,all")) == len(OPTIMIZERS)
    try:
        resolve("opro,nope")
        raise AssertionError("unknown optimizer accepted")
    except ValueError as exc:
        assert "nope" in str(exc)
    ok, why = available(replace(OPTIMIZERS["opro"], needs=("no_such_package_xyz",), install="pip3 install no-such"))
    assert not ok and "no_such_package_xyz" in why and "pip3 install" in why
    assert OPTIMIZERS["sapo-opro"].budget_mult == 3 and OPTIMIZERS["entropy-opro"].budget_mult == 5
    assert len({s.name for s in OPTIMIZERS.values()}) == len(OPTIMIZERS), "display names are unique"
    assert set(GROUPS) == {"core", "extended", "dspy", "all"}
    print("test_registry_groups_and_missing_packages: OK")


def test_ported_optimizers_stay_within_budget_and_keep_records_aligned():
    from prompt_opt_harness.tasks import REASONING_TASK

    dataset, factory = _reasoning_setup()
    specs = _registry_optimizers()
    assert {"pbt", "ensemble", "reg-opro", "dro-opro", "sapo-opro", "entropy-opro"} <= {s.key for s in specs}
    for spec in specs:
        for base_budget in (20, 61, 120):
            budget = base_budget * spec.budget_mult
            r = run_experiment(spec.name, spec.cls, factory, dataset, budget=budget, seed=4, task=REASONING_TASK)
            assert r.search_calls <= budget, (spec.key, budget, r.search_calls)
            assert len(r.candidate_pool) == len(r.candidate_stats) >= 1, spec.key
    # one logged pass per candidate even for variants that score several times
    for key in ("sapo-opro", "entropy-opro", "dro-opro"):
        from prompt_opt_harness.registry import OPTIMIZERS
        from prompt_opt_harness.optimizers.base import Optimizer
        spec = OPTIMIZERS[key]
        backend = factory()
        opt = spec.cls(backend, budget=200 * spec.budget_mult, seed=1, task=REASONING_TASK)
        opt.optimize(dataset.base_instruction, dataset.source.dev)
        assert len(opt.history) == len(opt.dev_records) == len(opt.dev_stats), key
        assert all(len(rs) == len(dataset.source.dev) for rs in opt.dev_records), key
    print("test_ported_optimizers_stay_within_budget_and_keep_records_aligned: OK")


def test_variants_change_the_score_as_described():
    from prompt_opt_harness.optimizers import OPRO
    from prompt_opt_harness.optimizers.wrappers import create_dro_optimizer, create_regularized_optimizer, create_sapo_optimizer
    from prompt_opt_harness.tasks import REASONING_TASK

    dataset, factory = _reasoning_setup()
    dev = dataset.source.dev
    plain = OPRO(factory(), budget=100, seed=1, task=REASONING_TASK)
    reg = create_regularized_optimizer(OPRO, penalty_per_word=0.01)(factory(), budget=100, seed=1, task=REASONING_TASK)
    text = "Solve it carefully step by step and show all of your work."
    a, b = plain._dev_score(text, dev), reg._dev_score(text, dev)
    assert abs((a - b) - 0.01 * len(text.split())) < 1e-9 or b == 0.0
    dro = create_dro_optimizer(OPRO, num_splits=3, split_ratio=0.5)(factory(), budget=100, seed=1, task=REASONING_TASK)
    assert dro._dev_score(text, dev) <= plain._dev_score(text, dev) + 1e-9, "worst subset can't beat the average by more than rounding"
    sapo_cls = create_sapo_optimizer(OPRO, num_perturbations=2)
    sapo = sapo_cls(factory(), budget=100, seed=1, task=REASONING_TASK)
    assert sapo._score_cost(dev) == 3 * len(dev) + 2
    assert not sapo._can_afford(0, 1, dev) or sapo.backend.stats.calls + 3 * len(dev) + 2 <= 100
    print("test_variants_change_the_score_as_described: OK")


class _Memorizer:
    """Backend whose answers depend on the instruction: 'variant' prompts only know the tuning questions."""

    def __init__(self, dataset, base_knows):
        from prompt_opt_harness.llm_backends import LLMCallStats
        self.stats, self.last_meta = LLMCallStats(), {}
        self.gold = {ex.text: ex.label for ex in dataset.source.dev + dataset.source.test + dataset.source.confirm}
        self.dev = {ex.text for ex in dataset.source.dev}
        self.base_knows = base_knows  # texts the starting prompt answers correctly
        self.n = 0

    def info(self):
        return {"backend": "memorizer", "model": "m", "temperature": 0.0, "max_tokens": 512}

    def generate(self, prompt, temperature=None, seed=None):
        import re
        m = re.search(r"Question: (.*)\nEnd your response", prompt, re.S)
        if not m:
            self.n += 1
            text = f"Solve the problem, variant {self.n}."
        else:
            q = m.group(1)
            knows = (q in self.dev) if "variant" in prompt else (q in self.base_knows)
            text = f"Answer: {self.gold[q] if knows else 0}"
        self.last_meta = {"finish_reason": "stop", "output_tokens": 5}
        self.stats.record(prompt, text, 0.0, output_tokens=5)
        return text


def test_guard_rejects_a_winner_that_only_memorized_the_tuning_questions():
    from prompt_opt_harness.optimizers import OPRO
    from prompt_opt_harness.optimizers.wrappers import create_guarded_optimizer
    from prompt_opt_harness.tasks import REASONING_TASK

    dataset, _ = _reasoning_setup()
    conf_and_test = {ex.text for ex in dataset.source.confirm + dataset.source.test}
    Guarded = create_guarded_optimizer(OPRO)
    assert Guarded.name == "Guard-OPRO" and Guarded.needs_confirm and not OPRO.needs_confirm

    # base prompt is right on confirm+test, wrong on the tuning questions; the 'variant' is the reverse
    r = run_experiment("Guard-OPRO", Guarded, lambda: _Memorizer(dataset, conf_and_test), dataset, budget=60, seed=1, task=REASONING_TASK)
    g = r.extra["guard"]
    assert g["checked"] and g["kept"] == "starting prompt", g
    assert g["winner_confirm_acc"] < g["start_confirm_acc"]
    assert r.best_instruction == dataset.base_instruction
    assert r.source_test_acc == 1.0, "the reported test score is for the prompt the guard kept"

    # when the winner is also right on the confirm questions, the guard keeps it
    r2 = run_experiment("Guard-OPRO", Guarded, lambda: _Memorizer(dataset, set()), dataset, budget=60, seed=1, task=REASONING_TASK)
    assert r2.extra["guard"]["kept"] in ("search winner", "starting prompt")
    plain = run_experiment("OPRO", OPRO, lambda: _Memorizer(dataset, conf_and_test), dataset, budget=60, seed=1, task=REASONING_TASK)
    assert plain.best_instruction != dataset.base_instruction and "guard" not in plain.extra, "unguarded OPRO keeps the memorizer"
    assert r.search_calls == plain.search_calls, "the guard adds no search calls"
    assert r.api_calls > plain.api_calls, "its confirmation calls are counted in the total"
    print("test_guard_rejects_a_winner_that_only_memorized_the_tuning_questions: OK")


def test_guard_needs_confirm_sample_and_leak_check_covers_it():
    import dataclasses
    from prompt_opt_harness.harness import check_splits
    from prompt_opt_harness.optimizers import OPRO
    from prompt_opt_harness.optimizers.wrappers import create_guarded_optimizer
    from prompt_opt_harness.tasks import REASONING_TASK

    dataset, factory = _reasoning_setup()
    bare = dataclasses.replace(dataset, source=dataclasses.replace(dataset.source, confirm=[]))
    try:
        run_experiment("g", create_guarded_optimizer(OPRO), factory, bare, budget=40, seed=1, task=REASONING_TASK)
        raise AssertionError("guard ran without a confirmation sample")
    except ValueError as exc:
        assert "confirmation" in str(exc)
    leaky = dataclasses.replace(dataset, source=dataclasses.replace(dataset.source, confirm=list(dataset.source.confirm) + [dataset.source.test[0]]))
    try:
        check_splits(leaky)
        raise AssertionError("confirmation/test overlap not caught")
    except ValueError as exc:
        assert "Data leak" in str(exc) and "confirmation" in str(exc)
    print("test_guard_needs_confirm_sample_and_leak_check_covers_it: OK")


def test_capped_backend_refuses_after_the_cap():
    from prompt_opt_harness.optimizers.dspy_adapters import CallBudgetExceeded, CappedBackend

    dataset, factory = _reasoning_setup()
    backend = factory()
    backend.generate("warm up")
    capped = CappedBackend(backend, 3)
    for _ in range(3):
        capped.generate(f"Question: {dataset.source.dev[0].text}\nEnd your response")
    try:
        capped.generate("one too many")
        raise AssertionError("cap not enforced")
    except CallBudgetExceeded:
        pass
    assert capped.used == 3 and capped.rejected == 1 and backend.stats.calls == 4
    print("test_capped_backend_refuses_after_the_cap: OK")


class _DspyScripted:
    """Answers task prompts (instruction-dependent) and DSPy's structured proposal prompts."""

    def __init__(self, dataset):
        from prompt_opt_harness.llm_backends import LLMCallStats
        self.stats, self.last_meta = LLMCallStats(), {}
        self.gold = {ex.text: ex.label for rows in ([dataset.source.dev, dataset.source.test, dataset.source.confirm]
                                                    + list(dataset.ood.values()) + list(dataset.ood_val.values())) for ex in rows}
        self.n = 0

    def info(self):
        return {"backend": "dspy-scripted", "model": "m", "temperature": 0.0, "max_tokens": 512}

    def generate(self, prompt, temperature=None, seed=None):
        import re
        m = re.search(r"Question: (.*)\nEnd your response", prompt, re.S)
        if m:
            text = f"Answer: {self.gold[m.group(1)] if 'carefully' in prompt else 0}"
        elif "Your output fields are" in prompt:
            names = re.findall(r"`(\w+)`", prompt.split("Your output fields are")[1].split("All interactions")[0])
            self.n += 1
            text = "".join(f"[[ ## {n} ## ]]\n{'Solve the problem carefully, variant %d.' % self.n if 'instruction' in n else 'ok'}\n\n" for n in names) + "[[ ## completed ## ]]"
        else:  # GEPA's reflection prompt is plain text and expects the new instruction inside a code block
            self.n += 1
            text = "```\nSolve the problem carefully, variant %d.\n```" % self.n
        self.last_meta = {"finish_reason": "stop", "output_tokens": 5}
        self.stats.record(prompt, text, 0.0, output_tokens=5)
        return text


def test_dspy_optimizers_if_installed():
    from prompt_opt_harness.registry import OPTIMIZERS, available
    from prompt_opt_harness.tasks import REASONING_TASK

    specs = [OPTIMIZERS["dspy-gepa"], OPTIMIZERS["dspy-mipro"]]
    if not all(available(s)[0] for s in specs):
        print("test_dspy_optimizers_if_installed: SKIPPED (" + "; ".join(available(s)[1] for s in specs if not available(s)[0]) + ")")
        return
    dataset, _ = _reasoning_setup()
    for spec in specs:
        for budget in (10, 60, 250):
            r = run_experiment(spec.name, spec.cls, lambda: _DspyScripted(dataset), dataset, budget=budget, seed=3, task=REASONING_TASK)
            assert r.search_calls <= budget, (spec.key, budget, r.search_calls)
            assert len(r.candidate_pool) == len(r.candidate_stats) >= 1
        assert "carefully" in r.best_instruction, "DSPy's better instruction (scored on all tuning questions) is returned at a big budget"
        assert r.dev_score == 1.0 and r.source_test_acc == 1.0
    print("test_dspy_optimizers_if_installed: OK")


if __name__ == "__main__":
    test_metrics_math()
    test_determinism()
    test_budget_is_respected()
    test_accuracy_in_range()
    test_reasoning_extraction()
    test_reasoning_dataset_shape()
    test_detailed_records_and_truncation()
    test_log_roundtrip_and_reports()
    test_proposal_temperature_reaches_backend()
    test_same_split_loader()
    test_duplicate_candidates_keep_their_own_stats()
    test_every_optimizer_stays_within_budget()
    test_new_optimizers_log_candidates_and_use_failures()
    test_random_search_is_reproducible()
    test_check_splits_rejects_overlap()
    test_only_the_oracle_sees_the_new_dataset()
    test_oracle_needs_a_tuning_sample()
    test_second_new_dataset_loaders()
    test_registry_groups_and_missing_packages()
    test_ported_optimizers_stay_within_budget_and_keep_records_aligned()
    test_variants_change_the_score_as_described()
    test_guard_rejects_a_winner_that_only_memorized_the_tuning_questions()
    test_guard_needs_confirm_sample_and_leak_check_covers_it()
    test_capped_backend_refuses_after_the_cap()
    test_dspy_optimizers_if_installed()
    test_analyze_labels_transfer_and_overfitting()
    test_analyze_truncation_verdict_and_cap_comparison()
    test_analyze_skips_old_logs_and_keeps_newest()
    print("\nAll smoke tests passed.")
