from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .harness import RunResult


def save_run(result: RunResult, log_dir: str = "logs") -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = result.name.lower().replace(" ", "_")
    path = os.path.join(log_dir, f"{safe_name}_{timestamp}.json")
    payload = result.to_dict() if hasattr(result, "to_dict") else {
        "name": result.name,
        "best_instruction": result.best_instruction,
        "dev_score": result.dev_score,
        "source_test_acc": result.source_test_acc,
        "ood_accs": result.ood_accs,
        "ood_gap": result.ood_gap,
        "worst_case_acc": result.worst_case_acc,
        "variance": result.variance,
        "api_calls": result.api_calls,
        "wall_time_sec": result.wall_time,
        "prompt_chars": result.prompt_chars,
        "prompt_words": result.prompt_words,
        "candidate_pool": [{"instruction": i, "dev_score": s} for i, s in result.candidate_pool],
    }
    # include tokens if present
    if hasattr(result, "input_tokens"):
        payload["input_tokens"] = getattr(result, "input_tokens")
        payload["output_tokens"] = getattr(result, "output_tokens")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
