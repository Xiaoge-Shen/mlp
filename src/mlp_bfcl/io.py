from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import DialogueMessage, NormalizedExample, ToolSpec


def load_normalized_examples(path: str | Path, max_examples: int | None = None) -> list[NormalizedExample]:
    path = Path(path)
    examples: list[NormalizedExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            examples.append(
                NormalizedExample(
                    example_id=raw["id"],
                    user_request=raw["user_request"],
                    tools=[ToolSpec(**tool) for tool in raw.get("tools", [])],
                    history=[DialogueMessage(**msg) for msg in raw.get("history", [])],
                    metadata=raw.get("metadata", {}),
                    gold=raw.get("gold", {}),
                )
            )
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
