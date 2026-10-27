from __future__ import annotations

import re
from typing import Callable, Dict, Optional, Sequence, Tuple

from .datasets import Dataset, DomainSplit, Example
from .prompts import REASONING_BASE_INSTRUCTION


def _require_datasets_lib():
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise ImportError("pip install datasets") from exc


def _gsm8k_gold(answer_field: str) -> str:
    # GSM8K's answer field ends with "#### <number>"
    return answer_field.split("####")[-1].strip().replace(",", "")


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _first_number(value) -> Optional[str]:
    """First number in `value`, commas removed ('9 (apples)' -> '9'); None if there is none."""
    match = _NUM.search(str(value))
    return match.group(0).replace(",", "") if match else None


def _pick(row, *keys):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    raise KeyError(f"none of {keys} in dataset row with columns {sorted(row)}")


# New-dataset registry. 'svamp' is the original (kept exactly as it was). The others exist so the harness
# can estimate cross-domain variance and worst-case accuracy on reasoning, which one new dataset cannot.
# Column names for the non-SVAMP sets come from the Hugging Face dataset cards, not from a run here;
# if a download fails or a column is missing the loader stops with a clear message before any model call.
OOD_SETS: Dict[str, dict] = {
    "svamp": dict(hf=("MU-NLPC/Calc-svamp", "default", "test"),
                  question=lambda r: r["question"].strip(), gold=lambda r: str(r["result"]).replace(",", ""), numeric_only=False),
    "gsmplus": dict(hf=("qintongli/GSM-Plus", None, "testmini"),
                    question=lambda r: str(_pick(r, "question")).strip(), gold=lambda r: _first_number(_pick(r, "answer")), numeric_only=True),
    "asdiv": dict(hf=("EleutherAI/asdiv", "asdiv", "validation"),
                  question=lambda r: (str(r.get("body", "")).strip() + " " + str(_pick(r, "question")).strip()).strip(),
                  gold=lambda r: _first_number(_pick(r, "answer")), numeric_only=True),
    "multiarith": dict(hf=("ChilleD/MultiArith", None, "test"),
                       question=lambda r: str(_pick(r, "question")).strip(), gold=lambda r: _first_number(_pick(r, "final_ans", "answer")), numeric_only=True),
}


def _load_ood_set(name: str, n: int, seed: int):
    """(reported test examples, held-out tuning examples) for one new dataset: 2n rows, first n are the test set."""
    if name not in OOD_SETS:
        raise ValueError(f"Unknown new dataset '{name}'. Choose from: {', '.join(OOD_SETS)}")
    from datasets import load_dataset

    spec = OOD_SETS[name]
    hf_id, config, split = spec["hf"]
    try:
        rows = load_dataset(hf_id, config, split=split).shuffle(seed=seed)
    except Exception as exc:  # network, name or split problems
        raise RuntimeError(f"Could not load new dataset '{name}' ({hf_id}): {exc}. Run with OOD=svamp to use the original one only.") from exc
    examples = []
    for row in rows:
        try:
            question, gold = spec["question"](row), spec["gold"](row)
        except KeyError as exc:
            raise RuntimeError(f"New dataset '{name}' ({hf_id}) does not have the columns we expected: {exc}. "
                               "Run with OOD=svamp to use the original one only.") from exc
        if not question or gold is None:
            if spec["numeric_only"]:
                continue          # e.g. GSM-Plus 'critical thinking' rows whose answer is 'None'
            raise RuntimeError(f"New dataset '{name}' has a row without a usable question or answer.")
        examples.append((question, gold))
        if len(examples) == 2 * n:
            break
    if len(examples) < 2 * n:
        raise RuntimeError(f"New dataset '{name}' has only {len(examples)} usable rows; need {2 * n}. Lower N_PER_SPLIT or drop it from OOD.")
    test = [Example(id=f"{name}{i}", text=q, label=g, domain=name) for i, (q, g) in enumerate(examples[:n])]
    val = [Example(id=f"{name}val{i}", text=q, label=g, domain=name) for i, (q, g) in enumerate(examples[n:])]
    return test, val


def load_real_reasoning_dataset(n_per_split: int = 30, seed: int = 0, split_mode: str = "same",
                                ood_sets: Sequence[str] = ("svamp",)) -> Dataset:
    # source: GSM8K (openai/gsm8k, "main"). OOD: SVAMP -- same task, structural
    # variation + distractor numbers, meant to test robustness rather than difficulty.
    # split_mode "same": dev and test are disjoint slices of GSM8K's test split.
    # split_mode "train_test": dev from GSM8K train, test from GSM8K test (the old setup).
    if split_mode not in ("same", "train_test"):
        raise ValueError("split_mode must be 'same' or 'train_test'")
    _require_datasets_lib()
    from datasets import load_dataset

    source_domain = "gsm8k"

    test_split = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed)
    # A third slice of source questions, never used to tune, for the confirmation guard. The dev and test
    # slices are the same as before, so earlier results stay comparable.
    if split_mode == "same":
        dev_rows = test_split.select(range(n_per_split))
        test_rows = test_split.select(range(n_per_split, 2 * n_per_split))
        confirm_rows = test_split.select(range(2 * n_per_split, 3 * n_per_split))
    else:
        train_split = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
        dev_rows = train_split.select(range(n_per_split))
        test_rows = test_split.select(range(n_per_split))
        confirm_rows = train_split.select(range(n_per_split, 2 * n_per_split))

    source_dev = [
        Example(id=f"rdev{i}", text=row["question"].strip(), label=_gsm8k_gold(row["answer"]), domain=source_domain)
        for i, row in enumerate(dev_rows)
    ]
    source_test = [
        Example(id=f"rtest{i}", text=row["question"].strip(), label=_gsm8k_gold(row["answer"]), domain=source_domain)
        for i, row in enumerate(test_rows)
    ]
    source_confirm = [
        Example(id=f"rconf{i}", text=row["question"].strip(), label=_gsm8k_gold(row["answer"]), domain=source_domain)
        for i, row in enumerate(confirm_rows)
    ]
    assert not {e.text for e in source_dev} & {e.text for e in source_test}, "dev/test overlap"
    assert not {e.text for e in source_confirm} & {e.text for e in source_dev + source_test}, "confirmation overlap"

    # For each new dataset the first n_per_split rows are the reported test questions (for SVAMP, exactly as
    # before); the next n_per_split are a separate sample that only oracle optimizers may tune on.
    ood, ood_val = {}, {}
    for name in ood_sets:
        ood[name], ood_val[name] = _load_ood_set(name, n_per_split, seed)

    return Dataset(
        source_domain=source_domain,
        source=DomainSplit(dev=source_dev, test=source_test, confirm=source_confirm),
        ood=ood,
        base_instruction=REASONING_BASE_INSTRUCTION,
        ood_val=ood_val,
    )
