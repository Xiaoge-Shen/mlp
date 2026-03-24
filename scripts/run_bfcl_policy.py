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

from mlp_bfcl.config import load_study_config
from mlp_bfcl.io import load_normalized_examples, write_json, write_jsonl
from mlp_bfcl.openai_client import OpenAICompatibleClient
from mlp_bfcl.policy import EscalationPolicy, FailureSignal, PolicyAction, PolicyState
from mlp_bfcl.prompts import (
    clarify_system_prompt,
    direct_system_prompt,
    render_repair_block,
    render_user_block,
    repair_system_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a normalized BFCL-style policy experiment.")
    parser.add_argument("--config", required=True, help="Path to the JSON study config.")
    parser.add_argument("--variant", required=True, choices=["direct", "clarify", "repair", "escalation"])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _call_model(client: OpenAICompatibleClient, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> tuple[str, dict]:
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return response.text, {
        "latency_seconds": response.latency_seconds,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "raw": response.raw,
    }


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    if config.endpoint is None:
        raise RuntimeError("The study config must define an endpoint section for policy runs.")

    examples = load_normalized_examples(config.normalized_input, max_examples=args.max_examples)
    output_dir = Path(args.output_dir) if args.output_dir else config.output_root_path / config.study_name / "policy_runs" / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAICompatibleClient(config.endpoint)
    policy = EscalationPolicy(config.policy)

    predictions: list[dict] = []
    traces: list[dict] = []
    latencies: list[float] = []
    completion_tokens: list[int] = []
    clarify_count = 0
    repair_count = 0
    abstain_count = 0

    for example in examples:
        state = PolicyState()
        selected = PolicyAction(args.variant)
        if selected == PolicyAction.ESCALATION:
            selected = policy.choose_initial_action(example, state)

        state.turns_used += 1
        initial_prompt = render_user_block(example)

        if selected == PolicyAction.CLARIFY:
            clarify_count += 1
            state.clarifications_used += 1
            text, meta = _call_model(
                client,
                clarify_system_prompt(),
                initial_prompt,
                config.policy.temperature,
                config.policy.max_output_tokens,
            )
            prediction = {"final_text": text, "final_action": "clarify"}
            trace = {"initial_action": "clarify", "repair_used": False}
        else:
            text, meta = _call_model(
                client,
                direct_system_prompt(),
                initial_prompt,
                config.policy.temperature,
                config.policy.max_output_tokens,
            )
            prediction = {"final_text": text, "final_action": "direct"}
            trace = {"initial_action": "direct", "repair_used": False}

        latencies.append(meta["latency_seconds"])
        if meta["completion_tokens"] is not None:
            completion_tokens.append(meta["completion_tokens"])

        failure_metadata = example.metadata.get("failure_signal", {})
        failure = FailureSignal(
            parse_failed=failure_metadata.get("parse_failed", False),
            execution_failed=failure_metadata.get("execution_failed", False),
            unsupported_tool=failure_metadata.get("unsupported_tool", False),
            schema_mismatch=failure_metadata.get("schema_mismatch", False),
            raw_error=failure_metadata.get("raw_error", ""),
        )
        should_try_repair = args.variant == "repair" or (
            args.variant == "escalation"
            and selected == PolicyAction.DIRECT
            and policy.choose_after_failure(failure, state) == PolicyAction.REPAIR
        )

        if should_try_repair:
            repair_count += 1
            state.repairs_used += 1
            state.turns_used += 1
            repair_text, repair_meta = _call_model(
                client,
                repair_system_prompt(),
                render_repair_block(example, text, failure.raw_error or "Synthetic failure signal not provided."),
                config.policy.temperature,
                config.policy.max_output_tokens,
            )
            prediction["repair_text"] = repair_text
            prediction["final_text"] = repair_text
            prediction["final_action"] = "repair"
            trace["repair_used"] = True
            latencies.append(repair_meta["latency_seconds"])
            if repair_meta["completion_tokens"] is not None:
                completion_tokens.append(repair_meta["completion_tokens"])

        if prediction["final_action"] == "abstain":
            abstain_count += 1

        predictions.append(
            {
                "id": example.example_id,
                "variant": args.variant,
                "prediction": prediction,
                "gold": example.gold,
                "metadata": example.metadata,
            }
        )
        traces.append(
            {
                "id": example.example_id,
                "variant": args.variant,
                "decision_log": state.decision_log,
                "trace": trace,
            }
        )

    metrics = {
        "study_name": config.study_name,
        "variant": args.variant,
        "num_examples": len(examples),
        "clarify_rate": round(clarify_count / len(examples), 4) if examples else 0.0,
        "repair_rate": round(repair_count / len(examples), 4) if examples else 0.0,
        "abstain_rate": round(abstain_count / len(examples), 4) if examples else 0.0,
        "mean_latency_seconds": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "mean_completion_tokens": round(statistics.mean(completion_tokens), 2) if completion_tokens else 0.0,
    }

    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_jsonl(output_dir / "traces.jsonl", traces)
    write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
