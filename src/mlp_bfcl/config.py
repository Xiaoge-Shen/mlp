from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EndpointConfig:
    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: int = 180


@dataclass
class PolicyConfig:
    max_clarifications: int = 1
    max_repairs: int = 1
    max_total_turns: int = 3
    temperature: float = 0.0
    max_output_tokens: int = 512
    ambiguity_markers: list[str] = field(default_factory=list)
    missing_argument_markers: list[str] = field(default_factory=list)


@dataclass
class StudyConfig:
    study_name: str
    bfcl_project_root: str
    normalized_input: str
    output_root: str
    categories: list[str]
    official_model_name: str
    backend: str = "vllm"
    skip_server_setup: bool = True
    num_gpus: int = 1
    gpu_memory_utilization: float = 0.9
    endpoint: EndpointConfig | None = None
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    variants: list[str] = field(default_factory=lambda: ["direct", "clarify", "repair", "escalation"])
    notes: str = ""

    @property
    def output_root_path(self) -> Path:
        return Path(self.output_root)


def _require(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise KeyError(f"Missing required config key: {key}")
    return data[key]


def load_study_config(path: str | Path) -> StudyConfig:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    endpoint = None
    if "endpoint" in data and data["endpoint"] is not None:
        endpoint = EndpointConfig(**data["endpoint"])
    policy = PolicyConfig(**data.get("policy", {}))
    return StudyConfig(
        study_name=_require(data, "study_name"),
        bfcl_project_root=_require(data, "bfcl_project_root"),
        normalized_input=_require(data, "normalized_input"),
        output_root=data.get("output_root", "outputs/bfcl"),
        categories=data.get("categories", []),
        official_model_name=_require(data, "official_model_name"),
        backend=data.get("backend", "vllm"),
        skip_server_setup=data.get("skip_server_setup", True),
        num_gpus=data.get("num_gpus", 1),
        gpu_memory_utilization=data.get("gpu_memory_utilization", 0.9),
        endpoint=endpoint,
        policy=policy,
        variants=data.get("variants", ["direct", "clarify", "repair", "escalation"]),
        notes=data.get("notes", ""),
    )

