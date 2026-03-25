# MLP Coursework Workspace

This repository now focuses on the BFCL coursework project:

`Can a lightweight schema-aware clarification guard reduce harmful tool calls and improve robustness on under-specified BFCL multi-turn cases for a small open model under a fixed inference budget?`

## Layout

```text
D:\mlp
|-- configs/
|   `-- bfcl_qwen3_budget_policy.json
|-- data/
|   `-- bfcl_normalized/
|-- outputs/
|   `-- bfcl/
|-- report/
|   |-- main.tex
|   `-- references.bib
|-- scripts/
|   |-- analyze_miss_param_turns.py
|   |-- prepare_bfcl_runtime.py
|   |-- run_bfcl_policy.py
|   |-- run_bfcl_study.py
|   `-- summarize_bfcl_study.py
|-- src/
|   `-- mlp_bfcl/
|-- notes/
`-- requirements.txt
```

## Single Source Of Truth

Edit only `configs/bfcl_qwen3_budget_policy.json` for:

- BFCL categories
- local vLLM host/port
- max context length
- policy parameters

Then regenerate the local runtime files:

```powershell
python scripts/prepare_bfcl_runtime.py --config configs/bfcl_qwen3_budget_policy.json
python scripts/run_bfcl_study.py --config configs/bfcl_qwen3_budget_policy.json
```

This writes:

- `outputs/bfcl_project_root/.env` for the official BFCL CLI
- `outputs/bfcl/<study>/runtime_setup.md` with the exact `vllm serve` command
- `outputs/bfcl/<study>/run_plan.md` with the exact `bfcl generate/evaluate` commands

## Current Project Story

The direct official BFCL baseline already exists for the main slices:

- `multi_turn_base = 27.50%`
- `multi_turn_miss_param = 18.00%`
- `memory_kv = 12.26%`

The project is now centred on the gap between `multi_turn_base` and `multi_turn_miss_param`.

The main method is no longer framed as a full interactive agent loop. Instead, it is:

- `Direct`: official BFCL baseline
- `Clarify-Guard`: generate a draft tool call, inspect it against required schema fields, and switch to a clarification question only when the request appears under-specified
- `Escalation`: preserve direct behaviour on normal cases and trigger Clarify-Guard only when necessary

`Repair` remains optional and secondary.

## Typical Workflow

1. Generate runtime setup and run plan:

```powershell
python scripts/prepare_bfcl_runtime.py --config configs/bfcl_qwen3_budget_policy.json
python scripts/run_bfcl_study.py --config configs/bfcl_qwen3_budget_policy.json
```

2. In WSL, start vLLM using the command from `outputs/bfcl/<study>/runtime_setup.md`.

3. In WSL, run the official direct baseline from `outputs/bfcl/<study>/run_plan.md`.

4. Analyse the official `multi_turn_miss_param` result:

```powershell
python scripts/analyze_miss_param_turns.py
```

5. Run local policy smoke tests:

```powershell
python scripts/run_bfcl_policy.py `
  --config configs/bfcl_qwen3_budget_policy.json `
  --variant clarify `
  --max-examples 100

python scripts/run_bfcl_policy.py `
  --config configs/bfcl_qwen3_budget_policy.json `
  --variant escalation `
  --max-examples 100
```

6. Aggregate results:

```powershell
python scripts/summarize_bfcl_study.py --config configs/bfcl_qwen3_budget_policy.json
```

## Notes

- `data/bfcl_normalized/sample.jsonl` is still only a schema example for local iteration.
- The official BFCL baseline still runs through `bfcl generate` and `bfcl evaluate`.
- `scripts/analyze_miss_param_turns.py` adds the turn-level diagnostics missing from the official BFCL aggregate score.
- The repository-local policy runner now uses schema-aware draft inspection rather than only keyword heuristics.
- Local best practice is to run the vLLM endpoint without an API key during BFCL debugging.
