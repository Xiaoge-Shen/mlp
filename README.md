# MLP GSM8K RL Workspace

This workspace contains a minimal but reusable TRL-based pipeline for comparing `GRPO` and `SAPO` on `Qwen2.5-0.5B` for GSM8K-style reasoning experiments.

It is intentionally lighter than Open-R1:

- one training script for both `GRPO` and `SAPO`
- one evaluation script
- W&B logging built in
- YAML configs for reproducible runs
- a small notebook-friendly experiment driver

## Layout

```text
D:\mlp
├── configs/
│   ├── grpo_qwen25_0.5b_gsm8k.yaml
│   └── sapo_qwen25_0.5b_gsm8k.yaml
├── notebooks/
│   └── experiment_driver.py
├── scripts/
│   ├── train_rl.py
│   └── evaluate_gsm8k.py
├── src/
│   └── mlp_reasoning/
│       ├── __init__.py
│       ├── data.py
│       ├── eval.py
│       └── rewards.py
├── notes/
└── requirements.txt
```

## Install

Create or activate an environment first, then install:

```powershell
pip install -r requirements.txt
```

Recommended extras for cluster work:

- log into W&B first: `wandb login`
- use the latest `transformers`
- use `math-verify>=0.5.2`

## Train

GRPO:

```powershell
python scripts/train_rl.py --config configs/grpo_qwen25_0.5b_gsm8k.yaml
```

SAPO:

```powershell
python scripts/train_rl.py --config configs/sapo_qwen25_0.5b_gsm8k.yaml
```

Example override from a notebook or shell:

```powershell
python scripts/train_rl.py `
  --config configs/sapo_qwen25_0.5b_gsm8k.yaml `
  --max-steps 200 `
  --train-samples 1024 `
  --output-dir outputs/sapo_budget_200
```

## Evaluate

Evaluate a base model:

```powershell
python scripts/evaluate_gsm8k.py `
  --base-model Qwen/Qwen2.5-0.5B `
  --output-dir outputs/eval_base_qwen25_0.5b
```

Evaluate a trained adapter:

```powershell
python scripts/evaluate_gsm8k.py `
  --base-model Qwen/Qwen2.5-0.5B `
  --adapter-path outputs/grpo_qwen25_0.5b_gsm8k/checkpoint-final `
  --output-dir outputs/eval_grpo_qwen25_0.5b
```

## W&B

Training logs are sent through TRL via `report_to="wandb"`.

You can also pass an API key directly:

```powershell
python scripts/train_rl.py `
  --config configs/grpo_qwen25_0.5b_gsm8k.yaml `
  --wandb-api-key YOUR_KEY
```

## Notebook Workflow

`notebooks/experiment_driver.py` contains a small Python helper for running multiple configurations from a notebook using `subprocess`.

The intended loop is:

1. change config or CLI overrides
2. launch `train_rl.py`
3. run `evaluate_gsm8k.py`
4. compare W&B runs

## Scope

This is a strong engineering baseline for the MLP coursework, but it is not the full project by itself. To make it coursework-ready, you still need:

- a clear research question
- a controlled experiment matrix
- proper reporting of accuracy, parse rate, reward behaviour, and training stability
- discussion of why `SAPO` is better or worse than `GRPO` under the same budget
