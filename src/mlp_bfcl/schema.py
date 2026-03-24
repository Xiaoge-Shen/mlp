from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DialogueMessage:
    role: str
    content: str


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedExample:
    example_id: str
    user_request: str
    tools: list[ToolSpec]
    history: list[DialogueMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    gold: dict[str, Any] = field(default_factory=dict)
