from __future__ import annotations

import random

from .datasets import Dataset, DomainSplit, Example


def _require_datasets_lib():
    try:
        import importlib
        import sys
        import os
        # Temporarily remove this package's directory from sys.path so the
        # external `datasets` package (huggingface) isn't shadowed by the
        # repo-local `datasets.py` module.
        orig_path = sys.path.copy()
        try:
            pkg_dir = os.path.abspath(os.path.dirname(__file__))
            sys.path = [p for p in sys.path if not (p and os.path.abspath(p).startswith(pkg_dir))]
            importlib.import_module("datasets")
        finally:
            sys.path = orig_path
    except ImportError as exc:
        raise ImportError("pip install datasets") from exc


def load_real_dataset(n_per_split: int = 30, seed: int = 0) -> Dataset:
    # source: SST-2 (nyu-mll/glue). OOD: amazon_polarity, tweet_eval/sentiment (neutral dropped)
    _require_datasets_lib()
    from datasets import load_dataset

    rng = random.Random(seed)
    source_domain = "sst2"

    sst2 = load_dataset("nyu-mll/glue", "sst2", split="validation")
    indices = list(range(len(sst2)))
    rng.shuffle(indices)
    dev_idx, test_idx = indices[:n_per_split], indices[n_per_split: 2 * n_per_split]

    def _sst2_example(i: int, prefix: str) -> Example:
        row = sst2[i]
        label = "Positive" if row["label"] == 1 else "Negative"
        return Example(id=f"{prefix}{i}", text=row["sentence"].strip(), label=label, domain=source_domain)

    source_dev = [_sst2_example(i, "sdev") for i in dev_idx]
    source_test = [_sst2_example(i, "stest") for i in test_idx]

    amazon = load_dataset("fancyzhx/amazon_polarity", split="test").shuffle(seed=seed).select(range(n_per_split))
    amazon_examples = [
        Example(
            id=f"amz{i}",
            text=row["content"].strip()[:500],
            label="Positive" if row["label"] == 1 else "Negative",
            domain="amazon",
        )
        for i, row in enumerate(amazon)
    ]

    tweets = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="test")
    tweets = tweets.filter(lambda row: row["label"] != 1)  # 1 == neutral
    tweets = tweets.shuffle(seed=seed).select(range(min(n_per_split, len(tweets))))
    tweet_examples = [
        Example(
            id=f"tw{i}",
            text=row["text"].strip(),
            label="Positive" if row["label"] == 2 else "Negative",
            domain="tweets",
        )
        for i, row in enumerate(tweets)
    ]

    return Dataset(
        source_domain=source_domain,
        source=DomainSplit(dev=source_dev, test=source_test),
        ood={"amazon": amazon_examples, "tweets": tweet_examples},
    )
