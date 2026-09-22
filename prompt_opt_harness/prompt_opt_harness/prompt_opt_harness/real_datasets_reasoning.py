from __future__ import annotations

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


def load_real_reasoning_dataset(n_per_split: int = 30, seed: int = 0) -> Dataset:
    # source: GSM8K (openai/gsm8k, "main"). OOD: SVAMP -- same task, structural
    # variation + distractor numbers, meant to test robustness rather than difficulty.
    _require_datasets_lib()
    from datasets import load_dataset

    source_domain = "gsm8k"

    train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed).select(range(n_per_split))
    test = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed).select(range(n_per_split))

    source_dev = [
        Example(id=f"rdev{i}", text=row["question"].strip(), label=_gsm8k_gold(row["answer"]), domain=source_domain)
        for i, row in enumerate(train)
    ]
    source_test = [
        Example(id=f"rtest{i}", text=row["question"].strip(), label=_gsm8k_gold(row["answer"]), domain=source_domain)
        for i, row in enumerate(test)
    ]

    svamp = load_dataset("MU-NLPC/Calc-svamp", "default", split="test").shuffle(seed=seed).select(range(n_per_split))
    svamp_examples = [
        Example(id=f"svamp{i}", text=row["question"].strip(), label=str(row["result"]).replace(",", ""), domain="svamp")
        for i, row in enumerate(svamp)
    ]

    return Dataset(
        source_domain=source_domain,
        source=DomainSplit(dev=source_dev, test=source_test),
        ood={"svamp": svamp_examples},
        base_instruction=REASONING_BASE_INSTRUCTION,
    )
