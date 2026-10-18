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

    payload = {
        "name": result.name,
        "seed": result.seed,
        "best_instruction": result.best_instruction,
        "dev_score": result.dev_score,
        "source_test_acc": result.source_test_acc,
        "ood_accs": result.ood_accs,
        "ood_gap": result.ood_gap,
        "worst_case_acc": result.worst_case_acc,
        "variance": result.variance,
        "api_calls": result.api_calls,
        "optimization_calls": result.optimization_calls,
        "optimization_budget": result.optimization_budget,
        "optimizer_budget_exhausted": result.optimizer_budget_exhausted,
        "valid": result.valid,
        "stop_reason": result.stop_reason,
        "evaluation_protocol": result.evaluation_protocol,
        "evaluation_calls": result.api_calls - result.optimization_calls,
        "wall_time_sec": result.wall_time,
        "input_tokens": None,
        "output_tokens": None,
        "usage_source": "unavailable",
        "approx_input_words": result.approx_input_words,
        "approx_output_words": result.approx_output_words,
        "approx_count_method": result.usage_count_method,
        "candidate_pool": [{"instruction": i, "dev_score": s} for i, s in result.candidate_pool],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
