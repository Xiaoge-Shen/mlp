from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlp_bfcl.io import write_jsonl


DEFAULT_DATASET = ROOT / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data" / "BFCL_v4_multi_turn_miss_param.json"
DEFAULT_GOLD = ROOT / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data" / "possible_answer" / "BFCL_v4_multi_turn_miss_param.json"
DEFAULT_FUNC_DOC_DIR = ROOT / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data" / "multi_turn_func_doc"
DEFAULT_OUTPUT = ROOT / "data" / "bfcl_normalized" / "bfcl_v4_multi_turn_miss_param_turns.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BFCL v4 multi_turn_miss_param into normalized turn-level JSONL.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--func-doc-dir", default=str(DEFAULT_FUNC_DOC_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_tool_registry(func_doc_dir: Path) -> dict[str, dict]:
    registry: dict[str, dict] = {}
    for path in sorted(func_doc_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                name = raw.get("name")
                if isinstance(name, str):
                    registry[name] = {
                        "name": name,
                        "description": raw.get("description", ""),
                        "parameters": raw.get("parameters", {}),
                    }
    return registry


def extract_tool_names(entry: dict) -> list[str]:
    tool_names: list[str] = []
    excluded = set(entry.get("excluded_function", []))
    for item in entry.get("path", []):
        if not isinstance(item, str) or "." not in item:
            continue
        _, func_name = item.split(".", 1)
        if func_name in excluded:
            continue
        if func_name not in tool_names:
            tool_names.append(func_name)
    return tool_names


def flatten_history(question_turns: list[list[dict]], turn_index: int) -> list[dict]:
    history: list[dict] = []
    for prior_turn in question_turns[:turn_index]:
        for message in prior_turn:
            role = message.get("role")
            content = message.get("content")
            if isinstance(role, str) and isinstance(content, str):
                history.append({"role": role, "content": content})
    return history


def main() -> None:
    args = parse_args()
    dataset_rows = load_jsonl(args.dataset)
    gold_rows = load_jsonl(args.gold)
    tool_registry = build_tool_registry(Path(args.func_doc_dir))
    gold_by_id = {row["id"]: row["ground_truth"] for row in gold_rows}

    exported_rows: list[dict] = []
    example_counter = 0

    for dataset_entry in dataset_rows:
        example_id = dataset_entry["id"]
        question_turns = dataset_entry.get("question", [])
        gold_turns = gold_by_id.get(example_id, [])
        tool_names = extract_tool_names(dataset_entry)
        tools = [tool_registry[name] for name in tool_names if name in tool_registry]

        for turn_index, turn_messages in enumerate(question_turns):
            if args.max_examples is not None and example_counter >= args.max_examples:
                break

            user_messages = [msg.get("content", "") for msg in turn_messages if msg.get("role") == "user"]
            if not user_messages:
                continue

            gold_turn = gold_turns[turn_index] if turn_index < len(gold_turns) else []
            empty_ground_truth = len(gold_turn) == 0
            expected_behavior = "clarify" if empty_ground_truth else "direct"
            failure_signal = {
                "parse_failed": False,
                "execution_failed": empty_ground_truth,
                "schema_mismatch": False,
                "raw_error": "Ground truth contains no tool call for this turn." if empty_ground_truth else "",
            }

            exported_rows.append(
                {
                    "id": f"{example_id}_turn_{turn_index}",
                    "user_request": user_messages[-1],
                    "history": flatten_history(question_turns, turn_index),
                    "tools": tools,
                    "metadata": {
                        "source_category": "multi_turn_miss_param",
                        "original_example_id": example_id,
                        "turn_index": turn_index,
                        "empty_ground_truth": empty_ground_truth,
                        "failure_signal": failure_signal,
                    },
                    "gold": {
                        "expected_behavior": expected_behavior,
                        "ground_truth": gold_turn,
                    },
                }
            )
            example_counter += 1

        if args.max_examples is not None and example_counter >= args.max_examples:
            break

    write_jsonl(args.output, exported_rows)
    print(json.dumps({"num_exported": len(exported_rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
