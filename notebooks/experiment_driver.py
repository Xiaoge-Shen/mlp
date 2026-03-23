from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=ROOT)


def train_experiment(
    config_path: str,
    output_dir: str,
    extra_args: list[str] | None = None,
) -> None:
    command = [
        sys.executable,
        "scripts/train_rl.py",
        "--config",
        config_path,
        "--output-dir",
        output_dir,
    ]
    if extra_args:
        command.extend(extra_args)
    run_command(command)


def evaluate_experiment(
    base_model: str,
    output_dir: str,
    adapter_path: str | None = None,
    extra_args: list[str] | None = None,
) -> None:
    command = [
        sys.executable,
        "scripts/evaluate_gsm8k.py",
        "--base-model",
        base_model,
        "--output-dir",
        output_dir,
    ]
    if adapter_path:
        command.extend(["--adapter-path", adapter_path])
    if extra_args:
        command.extend(extra_args)
    run_command(command)


if __name__ == "__main__":
    train_experiment(
        config_path="configs/grpo_qwen25_0.5b_gsm8k.yaml",
        output_dir="outputs/grpo_notebook_example",
        extra_args=["--max-steps", "20"],
    )
