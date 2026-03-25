# Handoff

## Current Context

- Workspace root: `D:\mlp`
- The BFCL topic has been narrowed to a coursework-safe claim: `a lightweight verifier-based clarification guard and escalation policy for under-specified BFCL multi-turn tool calls`
- Current date context: 25 March 2026; deadline remains 27 March 2026 at 12:00
- The user has already run the official direct BFCL baseline locally through vLLM and the BFCL CLI

## Current Engineering State

- Official direct BFCL results exist under `outputs/bfcl/bfcl_qwen3_budget_policy/official/direct/`
- Current useful direct baseline numbers are:
  - `multi_turn_base = 27.50%`
  - `multi_turn_miss_param = 18.00%`
  - `memory_kv = 12.26%`
- A new diagnostics script exists at `scripts/analyze_miss_param_turns.py`
- That script has already produced `outputs/bfcl/bfcl_qwen3_budget_policy/analysis/miss_param/summary.json`
- The current miss-param diagnostic summary is:
  - `harmful_call_rate_on_empty_turns = 0.6158`
  - `no_tool_call_rate_on_empty_turns = 0.3842`
- `src/mlp_bfcl/toolcall.py` now provides shared tool-call parsing
- `src/mlp_bfcl/policy.py` and `scripts/run_bfcl_policy.py` now implement verifier-based Clarify-Guard and Escalation
- A 200-turn exported `multi_turn_miss_param` pilot has been run for `direct`, `clarify`, and `escalation`
- `report/main.tex` has been updated to match the verifier-based project claim and the current mixed-result pilot finding
- `report/main.pdf` now builds successfully via `pdflatex` + `bibtex`

## Immediate Priorities

- Use the 200-turn pilot as the main method result, with `Clarify-Guard` as the strongest variant and `Escalation` as a secondary comparison
- Tighten the report wording around the main trade-off: fewer harmful calls, more false clarifications, roughly doubled latency
- Decide whether to add one more analysis table or stop here and focus on polishing the report
- Keep `memory_kv` as a negative-control reference, not as the main battlefront

## Resume Checklist

- Read `notes/project_brief.md`
- Read this handoff
- Read `notes/progress_tracker.md`
- Inspect `README.md`
- Inspect `report/main.tex`
- Inspect `outputs/bfcl/bfcl_qwen3_budget_policy/analysis/miss_param/summary.json`
- Inspect `outputs/bfcl/bfcl_qwen3_budget_policy/policy_runs/direct_200/metrics.json`
- Inspect `outputs/bfcl/bfcl_qwen3_budget_policy/policy_runs/clarify_200/metrics.json`
- Inspect `outputs/bfcl/bfcl_qwen3_budget_policy/policy_runs/escalation_200/metrics.json`
- Inspect `outputs/bfcl/bfcl_qwen3_budget_policy/official/direct/score/data_multi_turn.csv`

- Key comparison file: `outputs/bfcl/bfcl_qwen3_budget_policy/analysis/policy_200_comparison.json`
