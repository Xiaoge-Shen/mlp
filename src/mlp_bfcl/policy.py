from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from .config import PolicyConfig
from .schema import NormalizedExample
from .toolcall import parse_tool_calls, is_missing_argument_value


class PolicyAction(str, Enum):
    DIRECT = "direct"
    CLARIFY = "clarify"
    REPAIR = "repair"
    ABSTAIN = "abstain"


@dataclass
class PolicyState:
    clarifications_used: int = 0
    repairs_used: int = 0
    turns_used: int = 0
    decision_log: list[str] = field(default_factory=list)


@dataclass
class FailureSignal:
    parse_failed: bool = False
    execution_failed: bool = False
    unsupported_tool: bool = False
    schema_mismatch: bool = False
    raw_error: str = ""


@dataclass
class DraftInspection:
    tool_call_count: int = 0
    parsed_tool_names: list[str] = field(default_factory=list)
    missing_required_fields: list[str] = field(default_factory=list)
    unknown_tool_names: list[str] = field(default_factory=list)
    should_clarify: bool = False
    reasons: list[str] = field(default_factory=list)
    verifier_decision: str | None = None
    verifier_reason: str = ""
    verifier_missing_fields: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "tool_call_count": self.tool_call_count,
            "parsed_tool_names": self.parsed_tool_names,
            "missing_required_fields": self.missing_required_fields,
            "unknown_tool_names": self.unknown_tool_names,
            "should_clarify": self.should_clarify,
            "reasons": self.reasons,
            "verifier_decision": self.verifier_decision,
            "verifier_reason": self.verifier_reason,
            "verifier_missing_fields": self.verifier_missing_fields,
        }


class EscalationPolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def choose_initial_action(self, example: NormalizedExample, state: PolicyState) -> PolicyAction:
        if state.turns_used >= self.config.max_total_turns:
            state.decision_log.append("budget_exhausted_before_start")
            return PolicyAction.ABSTAIN

        lower_request = example.user_request.lower()
        if self._looks_under_specified(lower_request):
            state.decision_log.append("initial_action=clarify")
            return PolicyAction.CLARIFY

        state.decision_log.append("initial_action=direct")
        return PolicyAction.DIRECT

    def inspect_direct_draft(self, example: NormalizedExample, draft_text: str) -> DraftInspection:
        inspection = DraftInspection()
        parsed_calls = parse_tool_calls(draft_text, example.tools)
        inspection.tool_call_count = len(parsed_calls)
        inspection.parsed_tool_names = [call.name for call in parsed_calls]

        if self._looks_under_specified(example.user_request.lower()):
            inspection.should_clarify = True
            inspection.reasons.append("request_marker")

        if not parsed_calls:
            if inspection.should_clarify:
                inspection.reasons.append("no_tool_call_in_draft")
            return inspection

        tool_map = {tool.name: tool for tool in example.tools}
        primary_call = parsed_calls[0]
        if primary_call.name not in tool_map:
            inspection.unknown_tool_names.append(primary_call.name)
            inspection.reasons.append("unknown_tool")
            return inspection

        required_fields = self._extract_required_fields(example, primary_call.name)
        missing_fields = [
            field_name
            for field_name in required_fields
            if field_name not in primary_call.arguments or is_missing_argument_value(primary_call.arguments.get(field_name))
        ]
        if missing_fields:
            inspection.missing_required_fields.extend(missing_fields)
            inspection.should_clarify = True
            inspection.reasons.append("missing_required_arguments")
        return inspection

    def apply_verifier_payload(self, inspection: DraftInspection, verifier_text: str) -> DraftInspection:
        try:
            payload = json.loads(verifier_text)
        except json.JSONDecodeError:
            return inspection
        if not isinstance(payload, dict):
            return inspection

        decision = payload.get("decision")
        reason = payload.get("reason", "")
        missing_fields = payload.get("missing_fields", [])

        if isinstance(decision, str):
            inspection.verifier_decision = decision.lower()
        if isinstance(reason, str):
            inspection.verifier_reason = reason
        if isinstance(missing_fields, list):
            inspection.verifier_missing_fields = [item for item in missing_fields if isinstance(item, str)]

        if inspection.verifier_decision == "clarify":
            inspection.should_clarify = True
            inspection.reasons.append("verifier_clarify")
            if inspection.verifier_missing_fields:
                for field_name in inspection.verifier_missing_fields:
                    if field_name not in inspection.missing_required_fields:
                        inspection.missing_required_fields.append(field_name)
        elif inspection.verifier_decision == "execute":
            inspection.reasons.append("verifier_execute")
        return inspection

    def choose_after_failure(self, failure: FailureSignal, state: PolicyState) -> PolicyAction:
        if state.turns_used >= self.config.max_total_turns:
            state.decision_log.append("budget_exhausted_after_failure")
            return PolicyAction.ABSTAIN

        if state.repairs_used >= self.config.max_repairs:
            state.decision_log.append("repair_budget_exhausted")
            return PolicyAction.ABSTAIN

        if failure.parse_failed or failure.execution_failed or failure.schema_mismatch:
            state.decision_log.append("failure_action=repair")
            return PolicyAction.REPAIR

        state.decision_log.append("failure_action=abstain")
        return PolicyAction.ABSTAIN

    def _looks_under_specified(self, lower_request: str) -> bool:
        if any(marker.lower() in lower_request for marker in self.config.ambiguity_markers):
            return True
        if any(marker.lower() in lower_request for marker in self.config.missing_argument_markers):
            return True
        return False

    @staticmethod
    def _extract_required_fields(example: NormalizedExample, tool_name: str | None = None) -> list[str]:
        required_fields: list[str] = []
        for tool in example.tools:
            if tool_name is not None and tool.name != tool_name:
                continue
            parameters = tool.parameters or {}
            required = parameters.get("required", [])
            for item in required:
                if item not in required_fields:
                    required_fields.append(item)
        return required_fields
