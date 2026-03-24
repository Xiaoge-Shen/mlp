from __future__ import annotations

import json

from .schema import NormalizedExample


def render_tools_block(example: NormalizedExample) -> str:
    return json.dumps(
        [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in example.tools
        ],
        ensure_ascii=False,
        indent=2,
    )


def direct_system_prompt() -> str:
    return (
        "You are a careful tool-calling assistant. "
        "Choose the best tool and arguments from the provided schema. "
        "If the request is impossible with the available tools, say so briefly."
    )


def clarify_system_prompt() -> str:
    return (
        "You are a careful tool-calling assistant. "
        "Ask exactly one short clarification question only if required information is missing. "
        "Do not call a tool yet."
    )


def repair_system_prompt() -> str:
    return (
        "You are repairing a failed tool call. "
        "Use the error feedback to produce one corrected tool call. "
        "Do not repeat the same mistake."
    )


def render_user_block(example: NormalizedExample) -> str:
    history = "\n".join(f"{msg.role}: {msg.content}" for msg in example.history)
    if not history:
        history = "(none)"
    return (
        f"User request:\n{example.user_request}\n\n"
        f"Dialogue history:\n{history}\n\n"
        f"Available tools:\n{render_tools_block(example)}"
    )


def render_repair_block(example: NormalizedExample, previous_attempt: str, error_message: str) -> str:
    return (
        f"{render_user_block(example)}\n\n"
        f"Previous failed attempt:\n{previous_attempt}\n\n"
        f"Failure signal:\n{error_message}"
    )
