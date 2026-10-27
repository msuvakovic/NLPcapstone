from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from typing import Optional

from .harness import RunResult

SCHEMA_VERSION = 2


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def save_run(result: RunResult, log_dir: str = "logs") -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(log_dir, f"{_safe_name(result.name)}_s{result.seed}_{timestamp}.json")

    pool = []
    for i, (instruction, score) in enumerate(result.candidate_pool):
        entry = {"instruction": instruction, "dev_score": score}
        if i < len(result.candidate_stats):
            entry.update(result.candidate_stats[i])
        pool.append(entry)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp,
        "name": result.name,
        "seed": result.seed,
        "config": result.config,
        "best_instruction": result.best_instruction,
        "dev_score": result.dev_score,
        "source_test_acc": result.source_test_acc,
        "dev_minus_test": result.dev_minus_test,
        "ood_accs": result.ood_accs,
        "ood_gap": result.ood_gap,
        "worst_case_acc": result.worst_case_acc,
        "variance": result.variance,
        "api_calls": result.api_calls,
        "search_calls": result.search_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "wall_time_sec": result.wall_time,
        "eval_summary": result.eval_summary,
        "candidate_pool": pool,
        "dev_records": result.dev_records,
        "eval_records": result.eval_records,
        "extra": result.extra,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    return path


def find_run(name: str, seed: int, log_dir: str = "logs") -> Optional[str]:
    matches = sorted(glob.glob(os.path.join(log_dir, f"{_safe_name(name)}_s{seed}_*.json")))
    return matches[-1] if matches else None


def load_run(path: str) -> RunResult:
    with open(path) as f:
        d = json.load(f)
    return RunResult(
        name=d["name"],
        best_instruction=d["best_instruction"],
        dev_score=d["dev_score"],
        source_test_acc=d["source_test_acc"],
        ood_accs=d["ood_accs"],
        ood_gap=d["ood_gap"],
        worst_case_acc=d["worst_case_acc"],
        variance=d["variance"],
        api_calls=d["api_calls"],
        wall_time=d["wall_time_sec"],
        candidate_pool=[(c["instruction"], c["dev_score"]) for c in d["candidate_pool"]],
        seed=d.get("seed", 0),
        config=d.get("config", {}),
        search_calls=d.get("search_calls", 0),
        input_tokens=d.get("input_tokens", 0),
        output_tokens=d.get("output_tokens", 0),
        extra=d.get("extra", {}),
    )
