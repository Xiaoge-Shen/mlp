# Handoff

## Current Context

- Workspace root: `D:\mlp`
- The final topic is now the BFCL project: `a fixed-budget escalation policy over direct tool calling, clarification, and one-step repair`.
- There are 5 days remaining until the coursework deadline.
- The user has a strong local laptop for development and a school cluster environment for Slurm GPU runs.

## Current Engineering State

- A coursework-ready LaTeX draft exists in `report/main.tex` with bibliography `report/references.bib`.
- A BFCL experiment scaffold exists in `src/mlp_bfcl/`, `scripts/run_bfcl_study.py`, `scripts/run_bfcl_policy.py`, `scripts/summarize_bfcl_study.py`, and `configs/bfcl_qwen3_budget_policy.json`.
- A normalized JSONL schema example exists in `data/bfcl_normalized/sample.jsonl`.
- Old GRPO/GSM8K code and artifacts have been removed from the repository.

## Immediate Priorities

- Run the official direct BFCL baseline from the generated run plan.
- Replace the sample normalized JSONL with a real exported BFCL subset.
- Run the four BFCL variants: `Direct`, `Clarify`, `Repair`, and `Escalation`.
- Fill the report tables after the first benchmark results land.

## Resume Checklist

- Read `notes/project_brief.md`
- Read this handoff
- Read `notes/progress_tracker.md`
- Inspect `README.md`
- Inspect `report/main.tex`
- Inspect `outputs/bfcl/<study>/run_plan.md`
