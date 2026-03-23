from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlp_reasoning.rewards import ANSWER_TAG_RE, answers_match, extract_answer_text


def resolve_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def load_model_and_tokenizer(
    base_model: str,
    adapter_path: Optional[str] = None,
    precision: str = "bf16",
    attn_implementation: Optional[str] = "sdpa",
):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_dtype(precision),
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    max_prompt_length: int = 256,
    max_new_tokens: int = 160,
) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    completion_ids = outputs[:, prompt_length:]
    return tokenizer.batch_decode(completion_ids, skip_special_tokens=True)


def evaluate_dataset(
    model,
    tokenizer,
    dataset,
    batch_size: int = 4,
    max_prompt_length: int = 256,
    max_new_tokens: int = 160,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []

    total = 0
    correct = 0
    tagged = 0
    parseable = 0
    completion_token_lengths = []

    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
        completions = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=batch["prompt"],
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
        )

        for question, gold, prompt, completion in zip(
            batch["question"],
            batch["ground_truth"],
            batch["prompt"],
            completions,
        ):
            prediction = extract_answer_text(completion)
            is_tagged = bool(ANSWER_TAG_RE.search(completion))
            is_parseable = prediction is not None
            is_correct = answers_match(prediction, gold)
            completion_token_lengths.append(len(tokenizer.encode(completion, add_special_tokens=False)))

            total += 1
            correct += int(is_correct)
            tagged += int(is_tagged)
            parseable += int(is_parseable)

            records.append(
                {
                    "question": question,
                    "prompt": prompt,
                    "completion": completion,
                    "predicted_answer": prediction,
                    "ground_truth": gold,
                    "is_tagged": is_tagged,
                    "is_parseable": is_parseable,
                    "is_correct": is_correct,
                }
            )

    metrics = {
        "num_examples": float(total),
        "accuracy": correct / total if total else 0.0,
        "tag_success_rate": tagged / total if total else 0.0,
        "parse_success_rate": parseable / total if total else 0.0,
        "mean_completion_tokens": (
            sum(completion_token_lengths) / len(completion_token_lengths) if completion_token_lengths else 0.0
        ),
    }
    return metrics, records


def save_eval_outputs(output_dir: str, metrics: dict[str, float], records: list[dict[str, Any]]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_path = output_path / "metrics.json"
    predictions_path = output_path / "predictions.jsonl"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
