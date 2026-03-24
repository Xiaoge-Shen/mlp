from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import PolicyConfig
from .schema import NormalizedExample


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


class EscalationPolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def choose_initial_action(self, example: NormalizedExample, state: PolicyState) -> PolicyAction:
        if state.turns_used >= self.config.max_total_turns:
            state.decision_log.append("budget_exhausted_before_start")
            return PolicyAction.ABSTAIN

        lower_request = example.user_request.lower()
        if self._looks_under_specified(lower_request, example):
            state.decision_log.append("initial_action=clarify")
            return PolicyAction.CLARIFY

        state.decision_log.append("initial_action=direct")
        return PolicyAction.DIRECT

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

    def _looks_under_specified(self, lower_request: str, example: NormalizedExample) -> bool:
        if any(marker.lower() in lower_request for marker in self.config.ambiguity_markers):
            return True
        if any(marker.lower() in lower_request for marker in self.config.missing_argument_markers):
            return True
        required_names = self._extract_required_fields(example)
        if not required_names:
            return False
        grounded = sum(1 for field_name in required_names if field_name.lower() in lower_request)
        return grounded < len(required_names)

    @staticmethod
    def _extract_required_fields(example: NormalizedExample) -> list[str]:
        required_fields: list[str] = []
        for tool in example.tools:
            parameters = tool.parameters or {}
            required = parameters.get("required", [])
            for item in required:
                if item not in required_fields:
                    required_fields.append(item)
        return required_fields
