from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlp_bfcl.commands import build_bfcl_evaluate_command, build_bfcl_generate_command, study_root
from mlp_bfcl.config import load_study_config


def _quote(value: str) -> str:
    if any(ch in value for ch in [" ", "(", ")"]):
        return f'"{value}"'
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local BFCL runtime files from the study config.")
    parser.add_argument("--config", required=True, help="Path to the JSON study config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    if config.service is None:
        raise RuntimeError("The study config must define a service section.")

    root = study_root(config)
    root.mkdir(parents=True, exist_ok=True)

    bfcl_project_root = Path(config.bfcl_project_root)
    bfcl_project_root.mkdir(parents=True, exist_ok=True)
    dotenv_path = bfcl_project_root / ".env"

    dotenv_lines = [
        f"LOCAL_SERVER_ENDPOINT={config.service.host}",
        f"LOCAL_SERVER_PORT={config.service.port}",
    ]
    if config.service.require_api_key:
        dotenv_lines.extend(
            [
                f"REMOTE_OPENAI_BASE_URL=http://{config.service.host}:{config.service.port}/v1",
                f"REMOTE_OPENAI_API_KEY=${{{config.service.api_key_env}}}",
            ]
        )
    dotenv_path.write_text("\n".join(dotenv_lines) + "\n", encoding="utf-8")

    serve_parts = [
        "vllm serve",
        _quote(config.service.model),
        f"--host {config.service.host}",
        f"--port {config.service.port}",
        f"--dtype {config.service.dtype}",
        f"--generation-config {config.service.generation_config}",
        f"--max-model-len {config.service.max_model_len}",
        f"--gpu-memory-utilization {config.gpu_memory_utilization}",
    ]
    if config.service.served_model_name:
        serve_parts.append(f"--served-model-name {_quote(config.service.served_model_name)}")
    if config.service.require_api_key:
        serve_parts.append(f"--api-key ${config.service.api_key_env}")
    serve_command = " ".join(serve_parts)

    direct_result_dir = str((root / "official" / "direct" / "result").as_posix())
    direct_score_dir = str((root / "official" / "direct" / "score").as_posix())
    generate_command = build_bfcl_generate_command(config, direct_result_dir, ["multi_turn_base"])
    evaluate_command = build_bfcl_evaluate_command(config, direct_result_dir, direct_score_dir, ["multi_turn_base"])

    runtime_md = "\n".join(
        [
            f"# Runtime Setup: {config.study_name}",
            "",
            f"Single source of truth: `{Path(args.config).as_posix()}`",
            "",
            "## 1. Sync BFCL .env",
            "",
            f"Generated: `{dotenv_path.as_posix()}`",
            "",
            "## 2. Start vLLM",
            "",
            "```bash",
            serve_command,
            "```",
            "",
            "## 3. Smoke-test official BFCL baseline",
            "",
            "```bash",
            generate_command,
            evaluate_command,
            "```",
        ]
    )
    runtime_path = root / "runtime_setup.md"
    runtime_path.write_text(runtime_md, encoding="utf-8")

    print(runtime_md)


if __name__ == "__main__":
    main()
