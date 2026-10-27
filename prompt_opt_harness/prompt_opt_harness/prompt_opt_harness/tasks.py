from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from .prompts import (
    BASE_INSTRUCTION, build_classification_prompt,
    REASONING_BASE_INSTRUCTION, build_reasoning_prompt,
)


@dataclass
class Task:
    name: str
    base_instruction: str
    build_prompt: Callable[[str, str], str]
    is_correct: Callable[[str, str], bool]
    extract: Callable[[str], Optional[str]] = lambda response: None
    well_formed: Callable[[str], bool] = lambda response: True


def _classification_correct(response: str, label: str) -> bool:
    return response.strip().lower().startswith(label.lower()[:3])


def _classification_extract(response: str) -> Optional[str]:
    words = response.strip().split()
    return words[0].strip(".,!:;'\"") if words else None


def _classification_well_formed(response: str) -> bool:
    first = (_classification_extract(response) or "").lower()
    return first in ("positive", "negative")


_ANSWER_LINE_RE = re.compile(r"Answer:\s*(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)
_ANY_NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def extract_number(text: str) -> Optional[float]:
    matches = _ANSWER_LINE_RE.findall(text)
    raw = matches[-1] if matches else (_ANY_NUMBER_RE.findall(text) or [None])[-1]
    if raw is None:
        return None
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _reasoning_extract(response: str) -> Optional[str]:
    value = extract_number(response)
    if value is None:
        return None
    return str(int(value)) if value == int(value) else str(value)


def _reasoning_well_formed(response: str) -> bool:
    return bool(_ANSWER_LINE_RE.search(response))


def _reasoning_correct(response: str, label: str) -> bool:
    predicted = extract_number(response)
    if predicted is None:
        return False
    try:
        gold = float(str(label).replace(",", ""))
    except ValueError:
        return False
    return abs(predicted - gold) < 1e-6


SENTIMENT_TASK = Task(
    name="sentiment classification",
    base_instruction=BASE_INSTRUCTION,
    build_prompt=build_classification_prompt,
    is_correct=_classification_correct,
    extract=_classification_extract,
    well_formed=_classification_well_formed,
)

REASONING_TASK = Task(
    name="math word problem",
    base_instruction=REASONING_BASE_INSTRUCTION,
    build_prompt=build_reasoning_prompt,
    is_correct=_reasoning_correct,
    extract=_reasoning_extract,
    well_formed=_reasoning_well_formed,
)
