from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
GENERIC_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)
PLACEHOLDER_VALUES = {
    "",
    "none",
    "null",
    "unknown",
    "unspecified",
    "tbd",
    "n/a",
    "na",
    "?",
}
PLACEHOLDER_SUBSTRINGS = (
    "default",
    "usual",
    "somewhere",
    "around",
    "one of them",
)


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]
    raw_json: str


def _available_tool_metadata(available_tools: list[Any] | None) -> list[tuple[str, set[str]]]:
    metadata: list[tuple[str, set[str]]] = []
    for tool in available_tools or []:
        name = getattr(tool, "name", None)
        parameters = getattr(tool, "parameters", {}) or {}
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        field_names = set(properties.keys()) if isinstance(properties, dict) else set()
        if isinstance(name, str):
            metadata.append((name, field_names))
    return metadata


def _candidate_json_snippets(text: str) -> list[str]:
    candidates: list[str] = []
    for match in TOOL_CALL_RE.finditer(text):
        candidates.append(match.group(1))
    for match in FENCED_JSON_RE.finditer(text):
        candidates.append(match.group(1))
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    generic_match = GENERIC_JSON_RE.search(text)
    if generic_match:
        candidates.append(generic_match.group(1))

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.strip()
        if key and key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def parse_tool_calls(text: str, available_tools: list[Any] | None = None) -> list[ParsedToolCall]:
    calls: list[ParsedToolCall] = []
    tool_metadata = _available_tool_metadata(available_tools)
    for raw_json in _candidate_json_snippets(text):
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if isinstance(name, str) and isinstance(arguments, dict):
            calls.append(ParsedToolCall(name=name, arguments=arguments, raw_json=raw_json))
            continue

        if "error" in payload:
            continue

        if len(tool_metadata) == 1:
            inferred_name, field_names = tool_metadata[0]
            payload_keys = set(payload.keys())
            if payload_keys and (not field_names or payload_keys.issubset(field_names) or payload_keys & field_names):
                calls.append(ParsedToolCall(name=inferred_name, arguments=payload, raw_json=raw_json))
    return calls


def is_missing_argument_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in PLACEHOLDER_VALUES:
            return True
        return any(marker in lowered for marker in PLACEHOLDER_SUBSTRINGS)
    if isinstance(value, list):
        return len(value) == 0 or any(is_missing_argument_value(item) for item in value)
    if isinstance(value, dict):
        return len(value) == 0 or any(is_missing_argument_value(item) for item in value.values())
    return False
