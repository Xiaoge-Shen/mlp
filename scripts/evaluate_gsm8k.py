from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlp_reasoning.data import load_gsm8k_dataset
from mlp_reasoning.eval import evaluate_dataset, load_model_and_tokenizer, save_eval_outputs


def load_yaml_defaults(config_path: str | None) -> dict:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping.")
    return data


def build_parser(defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a base or LoRA-adapted model on GSM8K.")
    parser.set_defaults(**defaults)

    parser.add_argument("--config", default=None, help="Optional YAML config file.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--prompt-style", default="xml_cot")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-api-key", default=None)

    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, remaining = pre_parser.parse_known_args()
    defaults = load_yaml_defaults(pre_args.config)
    defaults["config"] = pre_args.config
    parser = build_parser(defaults)
    return parser.parse_args(remaining)


def maybe_log_to_wandb(args: argparse.Namespace, metrics: dict, records: list[dict]) -> None:
    if not args.wandb_project:
        return

    import wandb

    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        job_type="eval",
        config=vars(args),
    )
    run.log(metrics)
    sample_rows = records[:20]
    if sample_rows:
        table = wandb.Table(columns=list(sample_rows[0].keys()))
        for row in sample_rows:
            table.add_data(*[row[column] for column in table.columns])
        run.log({"sample_predictions": table})
    run.finish()


def main() -> None:
    args = parse_args()

    dataset = load_gsm8k_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.eval_split,
        max_samples=args.eval_samples,
        prompt_style=args.prompt_style,
    )
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
    )
    metrics, records = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.batch_size,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
    )
    save_eval_outputs(args.output_dir, metrics, records)
    maybe_log_to_wandb(args, metrics, records)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


