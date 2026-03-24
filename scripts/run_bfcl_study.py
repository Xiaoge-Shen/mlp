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
from mlp_bfcl.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a run plan for the BFCL coursework study.")
    parser.add_argument("--config", required=True, help="Path to the JSON study config.")
    parser.add_argument("--print-only", action="store_true", help="Only print the run plan.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    root = study_root(config)
    root.mkdir(parents=True, exist_ok=True)

    direct_result_dir = str((root / "official" / "direct" / "result").as_posix())
    direct_score_dir = str((root / "official" / "direct" / "score").as_posix())

    plan = {
        "study_name": config.study_name,
        "categories": config.categories,
        "variants": config.variants,
        "official_direct_generate": build_bfcl_generate_command(config, direct_result_dir),
        "official_direct_evaluate": build_bfcl_evaluate_command(config, direct_result_dir, direct_score_dir),
        "policy_variant_commands": {
            variant: (
                f"python scripts/run_bfcl_policy.py "
                f"--config {Path(args.config).as_posix()} "
                f"--variant {variant}"
            )
            for variant in config.variants
        },
    }

    run_plan_md = "\n".join(
        [
            f"# BFCL Run Plan: {config.study_name}",
            "",
            "## Official BFCL Baseline",
            "",
            "```bash",
            plan["official_direct_generate"],
            plan["official_direct_evaluate"],
            "```",
            "",
            "## Policy Variants",
            "",
            "These commands assume you have prepared a normalized JSONL file and a reachable OpenAI-compatible endpoint.",
            "",
            "```bash",
            *plan["policy_variant_commands"].values(),
            "```",
        ]
    )

    if not args.print_only:
        write_json(root / "run_manifest.json", plan)
        (root / "run_plan.md").write_text(run_plan_md, encoding="utf-8")

    print(run_plan_md)


if __name__ == "__main__":
    main()
