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
        "Return exactly one of the following two formats and nothing else. "
        "If the request is executable, output exactly one <tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call> block. "
        "If important required information is missing or the request is not executable yet, output exactly one JSON object of the form {\"error\": \"...\"}. "
        "Do not add explanations, markdown, or extra prose."
    )


def verification_system_prompt() -> str:
    return (
        "You are a strict tool-call verifier. "
        "You will be given the user request, dialogue history, tool schema, and a draft action. "
        "Decide whether the draft should be executed immediately or replaced by a clarification question. "
        "Return exactly one JSON object with keys decision, reason, and missing_fields. "
        "The decision must be either execute or clarify. "
        "Choose clarify if the draft relies on guessed arguments, unresolved references, missing required information, or unsupported assumptions about state. "
        "Choose execute only if the draft is fully justified by the request and available context."
    )


def clarify_system_prompt(missing_fields: list[str] | None = None) -> str:
    field_hint = ""
    if missing_fields:
        field_hint = f" Focus on the missing fields: {', '.join(missing_fields)}."
    return (
        "You are a careful tool-calling assistant. "
        "Ask exactly one short clarification question in plain text only. Do not call any tool and do not output JSON."
        + field_hint
    )


def repair_system_prompt() -> str:
    return (
        "You are repairing a failed tool call. "
        "Return exactly one <tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call> block and nothing else. "
        "Use the failure signal to correct the call and do not repeat the same mistake."
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


def render_verification_block(example: NormalizedExample, draft_text: str) -> str:
    return (
        f"{render_user_block(example)}\n\n"
        f"Draft action:\n{draft_text}"
    )


def render_repair_block(example: NormalizedExample, previous_attempt: str, error_message: str) -> str:
    return (
        f"{render_user_block(example)}\n\n"
        f"Previous failed attempt:\n{previous_attempt}\n\n"
        f"Failure signal:\n{error_message}"
    )
