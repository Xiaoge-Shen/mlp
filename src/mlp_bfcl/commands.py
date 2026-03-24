from __future__ import annotations

from pathlib import Path

from .config import StudyConfig


def build_bfcl_generate_command(
    config: StudyConfig,
    result_dir: str,
    test_categories: list[str] | None = None,
) -> str:
    categories = ",".join(test_categories or config.categories)
    parts = [
        "bfcl generate",
        f"--model {config.official_model_name}",
        f"--test-category {categories}",
        f"--backend {config.backend}",
        f"--num-gpus {config.num_gpus}",
        f"--gpu-memory-utilization {config.gpu_memory_utilization}",
        f"--result-dir {result_dir}",
    ]
    if config.skip_server_setup:
        parts.append("--skip-server-setup")
    return " ".join(parts)


def build_bfcl_evaluate_command(
    config: StudyConfig,
    result_dir: str,
    score_dir: str,
    test_categories: list[str] | None = None,
    partial_eval: bool = False,
) -> str:
    categories = ",".join(test_categories or config.categories)
    parts = [
        "bfcl evaluate",
        f"--model {config.official_model_name}",
        f"--test-category {categories}",
        f"--result-dir {result_dir}",
        f"--score-dir {score_dir}",
    ]
    if partial_eval:
        parts.append("--partial-eval")
    return " ".join(parts)


def study_root(config: StudyConfig) -> Path:
    return config.output_root_path / config.study_name
