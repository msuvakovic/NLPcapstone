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


def _classification_correct(response: str, label: str) -> bool:
    return response.strip().lower().startswith(label.lower()[:3])


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
)

REASONING_TASK = Task(
    name="math word problem",
    base_instruction=REASONING_BASE_INSTRUCTION,
    build_prompt=build_reasoning_prompt,
    is_correct=_reasoning_correct,
)
