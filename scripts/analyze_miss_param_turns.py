from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlp_bfcl.io import write_json, write_jsonl
from mlp_bfcl.toolcall import parse_tool_calls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse BFCL multi_turn_miss_param results turn by turn.")
    parser.add_argument(
        "--ground-truth",
        default=str(ROOT / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data" / "possible_answer" / "BFCL_v4_multi_turn_miss_param.json"),
    )
    parser.add_argument(
        "--result-path",
        default=str(ROOT / "outputs" / "bfcl" / "bfcl_qwen3_budget_policy" / "official" / "direct" / "result" / "Qwen_Qwen3-4B-Instruct-2507-FC" / "multi_turn" / "BFCL_v4_multi_turn_miss_param_result.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "bfcl" / "bfcl_qwen3_budget_policy" / "analysis" / "miss_param"),
    )
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


def clean_preview(text: str, limit: int = 220) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_rows = load_jsonl(args.ground_truth)
    result_rows = load_jsonl(args.result_path)

    gt_by_id = {row["id"]: row["ground_truth"] for row in ground_truth_rows}

    turn_rows: list[dict] = []
    empty_turn_rows: list[dict] = []
    example_rows: list[dict] = []

    empty_turn_count = 0
    empty_turn_with_call = 0
    required_turn_count = 0
    required_turn_without_call = 0
    tool_calls_per_turn: list[int] = []

    for row in result_rows:
        example_id = row["id"]
        gt_turns = gt_by_id.get(example_id, [])
        result_turns = row.get("result", [])

        example_empty_turns = 0
        example_harmful_calls = 0
        example_required_turns = 0
        example_missed_required = 0

        for turn_index, gt_turn in enumerate(gt_turns):
            model_turn = result_turns[turn_index] if turn_index < len(result_turns) else []
            model_texts = [item for item in model_turn if isinstance(item, str)]

            parsed_calls = []
            for text in model_texts:
                parsed_calls.extend(parse_tool_calls(text))

            model_tool_names = [call.name for call in parsed_calls]
            model_tool_call_count = len(parsed_calls)
            tool_calls_per_turn.append(model_tool_call_count)
            is_empty_ground_truth = len(gt_turn) == 0
            emitted_any_tool_call = model_tool_call_count > 0

            turn_row = {
                "id": example_id,
                "turn_index": turn_index,
                "empty_ground_truth": is_empty_ground_truth,
                "ground_truth_call_count": len(gt_turn),
                "model_tool_call_count": model_tool_call_count,
                "model_tool_names": model_tool_names,
                "emitted_any_tool_call": emitted_any_tool_call,
                "first_model_text": clean_preview(model_texts[0]) if model_texts else "",
            }
            turn_rows.append(turn_row)

            if is_empty_ground_truth:
                empty_turn_count += 1
                example_empty_turns += 1
                empty_turn_rows.append(turn_row)
                if emitted_any_tool_call:
                    empty_turn_with_call += 1
                    example_harmful_calls += 1
            else:
                required_turn_count += 1
                example_required_turns += 1
                if not emitted_any_tool_call:
                    required_turn_without_call += 1
                    example_missed_required += 1

        example_rows.append(
            {
                "id": example_id,
                "empty_turn_count": example_empty_turns,
                "harmful_call_count_on_empty_turns": example_harmful_calls,
                "required_turn_count": example_required_turns,
                "no_call_count_on_required_turns": example_missed_required,
            }
        )

    summary = {
        "num_examples": len(result_rows),
        "num_turns": len(turn_rows),
        "num_empty_ground_truth_turns": empty_turn_count,
        "num_required_turns": required_turn_count,
        "harmful_call_count_on_empty_turns": empty_turn_with_call,
        "harmful_call_rate_on_empty_turns": round(empty_turn_with_call / empty_turn_count, 4) if empty_turn_count else 0.0,
        "no_tool_call_rate_on_empty_turns": round((empty_turn_count - empty_turn_with_call) / empty_turn_count, 4) if empty_turn_count else 0.0,
        "no_tool_call_count_on_required_turns": required_turn_without_call,
        "no_tool_call_rate_on_required_turns": round(required_turn_without_call / required_turn_count, 4) if required_turn_count else 0.0,
        "mean_model_tool_calls_per_turn": round(statistics.mean(tool_calls_per_turn), 4) if tool_calls_per_turn else 0.0,
    }

    write_jsonl(output_dir / "turn_level.jsonl", turn_rows)
    write_jsonl(output_dir / "empty_ground_truth_turns.jsonl", empty_turn_rows)
    write_jsonl(output_dir / "example_level.jsonl", example_rows)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
