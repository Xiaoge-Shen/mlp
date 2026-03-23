from __future__ import annotations

import os
import re
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


DEFAULT_PROMPT_TEMPLATE = """You are solving a grade-school math problem.
Think step by step, then end with a single XML answer tag:
<answer>YOUR_FINAL_NUMERIC_ANSWER</answer>

Question:
{question}
"""


def extract_gsm8k_gold(answer_text: str) -> str:
    """Extract the final GSM8K answer after the #### marker."""
    if "####" in answer_text:
        return answer_text.split("####", maxsplit=1)[1].strip()
    return answer_text.strip()


def build_prompt(question: str, prompt_style: str = "xml_cot") -> str:
    if prompt_style != "xml_cot":
        raise ValueError(f"Unsupported prompt_style: {prompt_style}")
    return DEFAULT_PROMPT_TEMPLATE.format(question=question.strip())


def _load_raw_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
) -> Dataset:
    if os.path.exists(dataset_name):
        dataset_obj = load_from_disk(dataset_name)
        if isinstance(dataset_obj, DatasetDict):
            return dataset_obj[split]
        return dataset_obj
    return load_dataset(dataset_name, dataset_config, split=split)


def load_gsm8k_dataset(
    dataset_name: str = "openai/gsm8k",
    dataset_config: Optional[str] = "main",
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
    prompt_style: str = "xml_cot",
) -> Dataset:
    """Load GSM8K and convert it to the prompt/ground_truth format TRL expects."""
    dataset = _load_raw_dataset(dataset_name, dataset_config, split)
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    def _map_example(example: dict, idx: int) -> dict:
        question = example["question"].strip()
        ground_truth = extract_gsm8k_gold(example["answer"])
        return {
            "id": idx,
            "question": question,
            "ground_truth": ground_truth,
            "prompt": build_prompt(question, prompt_style=prompt_style),
        }

    return dataset.map(
        _map_example,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc=f"Formatting {split} split for reasoning RL",
    )
