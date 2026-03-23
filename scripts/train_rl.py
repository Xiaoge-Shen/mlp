from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlp_reasoning.data import load_gsm8k_dataset
from mlp_reasoning.rewards import correctness_reward, format_reward, parseable_answer_reward


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def load_yaml_defaults(config_path: str | None) -> dict:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping.")
    return data


def build_parser(defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Qwen2.5 with TRL GRPO or SAPO on GSM8K.")
    parser.set_defaults(**defaults)

    parser.add_argument("--config", default=None, help="Optional YAML config file.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--prompt-style", default="xml_cot")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output-dir", default="outputs/grpo_run")
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--wandb-project", default="mlp-gsm8k-rl")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-api-key", default=None)

    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--gradient-checkpointing", type=str2bool, default=True)

    parser.add_argument("--use-lora", type=str2bool, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    parser.add_argument("--loss-type", choices=["grpo", "dapo", "dr_grpo", "sapo"], default="grpo")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--sapo-temperature-pos", type=float, default=1.0)
    parser.add_argument("--sapo-temperature-neg", type=float, default=1.05)
    parser.add_argument("--reward-weights", type=float, nargs="+", default=[0.25, 1.0])
    parser.add_argument("--include-parse-reward", type=str2bool, default=False)

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-completion-length", type=int, default=160)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--log-completions", type=str2bool, default=True)

    parser.add_argument("--use-vllm", type=str2bool, default=False)
    parser.add_argument("--vllm-mode", choices=["server", "colocate"], default="server")
    parser.add_argument("--vllm-enable-sleep-mode", type=str2bool, default=False)

    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, remaining = pre_parser.parse_known_args()
    defaults = load_yaml_defaults(pre_args.config)
    defaults["config"] = pre_args.config
    parser = build_parser(defaults)
    return parser.parse_args(remaining)


def build_reward_stack(args: argparse.Namespace):
    reward_funcs = [format_reward, correctness_reward]
    reward_weights = list(args.reward_weights)
    if args.include_parse_reward:
        reward_funcs.insert(1, parseable_answer_reward)
        if len(reward_weights) == 2:
            reward_weights = [reward_weights[0], 0.1, reward_weights[1]]
    if len(reward_funcs) != len(reward_weights):
        raise ValueError(
            f"reward_weights must match reward funcs. Got {len(reward_weights)} weights for {len(reward_funcs)} rewards."
        )
    return reward_funcs, reward_weights


def maybe_login_wandb(api_key: str | None) -> None:
    if not api_key:
        return
    import wandb

    wandb.login(key=api_key)


def dump_run_config(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def build_grpo_config(args: argparse.Namespace, reward_weights: list[float], run_name: str) -> GRPOConfig:
    signature = inspect.signature(GRPOConfig.__init__).parameters

    config_kwargs = {
        "output_dir": args.output_dir,
        "run_name": run_name,
        "report_to": args.report_to,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "log_completions": args.log_completions,
        "bf16": args.precision == "bf16",
        "fp16": args.precision == "fp16",
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "beta": args.beta,
        "epsilon": args.epsilon,
        "reward_weights": reward_weights,
        "use_vllm": args.use_vllm,
        "vllm_mode": args.vllm_mode,
        "vllm_enable_sleep_mode": args.vllm_enable_sleep_mode,
    }

    filtered_kwargs = {key: value for key, value in config_kwargs.items() if key in signature}

    if "loss_type" in signature:
        filtered_kwargs["loss_type"] = args.loss_type
    elif args.loss_type != "grpo":
        raise RuntimeError(
            "Your installed TRL does not support configurable loss_type. Upgrade TRL before using SAPO."
        )

    supports_sapo_temps = "sapo_temperature_pos" in signature and "sapo_temperature_neg" in signature
    if supports_sapo_temps:
        filtered_kwargs["sapo_temperature_pos"] = args.sapo_temperature_pos
        filtered_kwargs["sapo_temperature_neg"] = args.sapo_temperature_neg
    elif args.loss_type == "sapo":
        raise RuntimeError(
            "Your installed TRL version does not expose SAPO config parameters. "
            "Upgrade TRL to a version whose GRPOConfig supports sapo_temperature_pos/neg."
        )

    return GRPOConfig(**filtered_kwargs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    maybe_login_wandb(args.wandb_api_key)
    dump_run_config(args)

    os.environ["WANDB_PROJECT"] = args.wandb_project

    run_name = args.wandb_run_name or Path(args.output_dir).name

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "dtype": resolve_dtype(args.precision),
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_dataset = load_gsm8k_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.train_split,
        max_samples=args.train_samples,
        seed=args.seed,
        prompt_style=args.prompt_style,
    )

    reward_funcs, reward_weights = build_reward_stack(args)
    training_args = build_grpo_config(args, reward_weights, run_name)

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=args.lora_target_modules,
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    final_checkpoint = Path(args.output_dir) / "checkpoint-final"
    trainer.save_model(str(final_checkpoint))


if __name__ == "__main__":
    main()
