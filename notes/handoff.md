# Handoff

## Current Context

- Workspace root: `D:\mlp`
- This directory contains the CW4 brief, the report template, and a new reasoning-RL experiment scaffold.
- There are 5 days remaining until the coursework deadline.
- The user has a strong local laptop for development and a school cluster environment for Slurm GPU runs.
- The user originally had no project yet, but a candidate topic has emerged: `Qwen2.5-0.5B on GSM8K with GRPO vs SAPO`.

## Known Environment

- Local machine is suitable for coding, preprocessing, debugging, and smaller experiments.
- Remote cluster access is configured through SSH config.
- Use `dice` and `mlp1` as the relevant remote hosts when needed.
- Do not record secrets in this file.

## Current Engineering State

- A new TRL-based workspace scaffold exists under `scripts/`, `src/`, `configs/`, and `notebooks/`.
- `scripts/train_rl.py` supports both `GRPO` and `SAPO` through the same pipeline.
- `scripts/evaluate_gsm8k.py` supports base-model and adapter evaluation.
- W&B integration is wired into training and optional evaluation logging.
- Runtime ML dependencies are not installed in the current Python environment yet.

## Immediate Priorities

- Decide whether to commit to the GSM8K reasoning-RL topic.
- If yes, install dependencies from `requirements.txt`.
- Run a tiny smoke test for training.
- Run a baseline evaluation for the chosen base or instruct model.
- Freeze the final four-experiment matrix before large runs.

## Progress Tracking

- Primary tracker: `notes/progress_tracker.md`
- Update the tracker whenever a milestone advances, a decision changes, or a new blocker appears.
- Use the tracker rather than chat history to understand project state.

## Resume Checklist

- Read `notes/project_brief.md`
- Read this handoff
- Read `notes/progress_tracker.md`
- Inspect `README.md` for the new training and evaluation commands
- Confirm the final topic before spending compute
- Decide what must run locally versus on cluster
