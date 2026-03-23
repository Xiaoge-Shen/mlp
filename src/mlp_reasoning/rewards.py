from __future__ import annotations

import re
from typing import Any, Iterable, Optional

try:
    from math_verify import parse, verify
except ImportError:  # pragma: no cover - optional dependency fallback
    parse = None
    verify = None


ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion)


def extract_answer_text(text: str) -> Optional[str]:
    tagged = ANSWER_TAG_RE.search(text)
    if tagged:
        return tagged.group(1).strip()
    numbers = NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].strip()
    return None


def _normalize_numeric_string(text: Optional[str]) -> str:
    if text is None:
        return ""
    return text.replace(",", "").strip()


def answers_match(prediction: Optional[str], ground_truth: str) -> bool:
    prediction = _normalize_numeric_string(prediction)
    ground_truth = _normalize_numeric_string(ground_truth)
    if not prediction:
        return False
    if parse is not None and verify is not None:
        try:
            return bool(verify(parse(ground_truth), parse(prediction)))
        except Exception:
            pass
    return prediction == ground_truth


def format_reward(completions: Iterable[Any], **_: Any) -> list[float]:
    rewards = []
    for completion in completions:
        text = _completion_to_text(completion)
        rewards.append(1.0 if ANSWER_TAG_RE.search(text) else 0.0)
    return rewards


def parseable_answer_reward(completions: Iterable[Any], **_: Any) -> list[float]:
    rewards = []
    for completion in completions:
        text = _completion_to_text(completion)
        rewards.append(1.0 if extract_answer_text(text) is not None else 0.0)
    return rewards


def correctness_reward(
    completions: Iterable[Any],
    ground_truth: list[str],
    **_: Any,
) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, ground_truth):
        text = _completion_to_text(completion)
        prediction = extract_answer_text(text)
        rewards.append(1.0 if answers_match(prediction, gold) else 0.0)
    return rewards
