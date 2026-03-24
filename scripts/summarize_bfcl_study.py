from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlp_bfcl.commands import study_root
from mlp_bfcl.config import load_study_config
from mlp_bfcl.io import write_json
from mlp_bfcl.results import load_variant_metrics, read_csv_rows, write_summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate BFCL study outputs into report-friendly summaries.")
    parser.add_argument("--config", required=True, help="Path to the JSON study config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    root = study_root(config)

    rows: list[dict[str, object]] = []
    for variant in config.variants:
        metrics = load_variant_metrics(root / "policy_runs" / variant / "metrics.json")
        if metrics:
            rows.append(
                {
                    "variant": variant,
                    "num_examples": metrics.get("num_examples"),
                    "clarify_rate": metrics.get("clarify_rate"),
                    "repair_rate": metrics.get("repair_rate"),
                    "mean_latency_seconds": metrics.get("mean_latency_seconds"),
                    "mean_completion_tokens": metrics.get("mean_completion_tokens"),
                }
            )

    official_rows = read_csv_rows(root / "official" / "direct" / "score" / "data_multi_turn.csv")
    summary = {
        "study_name": config.study_name,
        "policy_metrics": rows,
        "official_multi_turn_rows": official_rows,
    }

    write_json(root / "summary.json", summary)
    write_summary_csv(root / "summary.csv", rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
