import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.build_capstone_report import (  # noqa: E402
    RESULT_DIR,
    SOURCE_ARTIFACTS,
    build_markdown,
    build_result,
    consolidated_protocol,
    source_manifest,
)


def test_capstone_report_sources_and_metrics():
    result = build_result()
    opro = result["controlled_sentiment"]["opro"]

    assert len(source_manifest()) == len(SOURCE_ARTIFACTS)
    assert len(opro["searches"]) == 3
    assert opro["same_prompt_all_seeds"] is True
    assert math.isclose(
        opro["mean_ood_accuracy"],
        (opro["amazon_accuracy"] + opro["tweets_accuracy"]) / 2,
    )
    assert math.isclose(
        opro["ood_gap"],
        opro["source_test_accuracy"] - opro["mean_ood_accuracy"],
    )
    assert len(result["reasoning_pilot"]["methods"]) == 6
    assert consolidated_protocol()["consolidation_environment"]["new_model_calls"] == 0


def test_committed_derived_files_match_generator():
    result = build_result()
    assert json.loads((RESULT_DIR / "summary.json").read_text()) == result
    assert (RESULT_DIR / "report.md").read_text() == build_markdown(result)