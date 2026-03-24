# MLP Coursework Workspace

This repository now focuses on the BFCL coursework project:

`Under a fixed inference budget, can a simple escalation policy that chooses between direct tool calling, clarification, and one-step repair improve BFCL performance for small open models?`

## Layout

```text
D:\mlp
├── configs/
│   └── bfcl_qwen3_budget_policy.json
├── data/
│   └── bfcl_normalized/
├── outputs/
│   └── bfcl/
├── report/
│   ├── main.tex
│   └── references.bib
├── scripts/
│   ├── run_bfcl_policy.py
│   ├── run_bfcl_study.py
│   └── summarize_bfcl_study.py
├── src/
│   └── mlp_bfcl/
├── notes/
└── requirements.txt
```

## Core Files

- `configs/bfcl_qwen3_budget_policy.json`: main experiment config
- `scripts/run_bfcl_study.py`: generate the official BFCL baseline run plan
- `scripts/run_bfcl_policy.py`: run one local policy variant on normalized JSONL input
- `scripts/summarize_bfcl_study.py`: aggregate policy and official BFCL outputs
- `report/main.tex`: coursework report draft

## Typical Workflow

1. Generate the run plan:

```powershell
python scripts/run_bfcl_study.py --config configs/bfcl_qwen3_budget_policy.json
```

2. Run the official direct baseline from the generated plan in `outputs/bfcl/<study>/run_plan.md`.

3. Run local policy smoke tests:

```powershell
python scripts/run_bfcl_policy.py `
  --config configs/bfcl_qwen3_budget_policy.json `
  --variant escalation `
  --max-examples 100
```

4. Aggregate results:

```powershell
python scripts/summarize_bfcl_study.py --config configs/bfcl_qwen3_budget_policy.json
```

## Notes

- `data/bfcl_normalized/sample.jsonl` is only a schema example for local iteration.
- The official BFCL baseline still runs through `bfcl generate` and `bfcl evaluate`.
- The repository-local BFCL scripts use only the Python standard library.
