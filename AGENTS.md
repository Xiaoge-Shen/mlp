# MLP Coursework 4 Workspace

This workspace is for the University of Edinburgh Machine Learning Practical 2025/26 Coursework 4 final project report and supporting code.

## Scope

- Treat this directory as the canonical workspace for CW4.
- The original PDF briefs in the root are source documents and must not be modified.
- Prioritise work that improves the final submission quality within the remaining time budget.

## Deliverables

- Final report PDF named `finalReport-<groupID>.pdf`
- Source code folder named `<groupID>_source_code/`
- Submission directory named `coursework4-<groupID>/`
- Final archive named `<groupID>.zip`

## Working Rules

- Do not store passwords, tokens, private keys, or other secrets in repository files.
- Keep durable project context in `notes/` so a new Codex session can resume without relying on chat history.
- Use `notes/progress_tracker.md` as the primary progress source of truth and update it after substantial work.
- Prefer reproducible experiments over ad hoc runs.
- Any experiment used in the report should have enough detail to reproduce: data split, preprocessing, model variant, hyperparameters, seed, hardware, and command.
- Prefer local development for iteration and debugging; use the school cluster for longer GPU jobs and final runs.
- Before substantial work, read `notes/project_brief.md`, `notes/handoff.md`, and `notes/progress_tracker.md`.
- When project code is added later, keep generated outputs out of the submission source folder unless explicitly required.

## Environment Context

- Local machine: Windows laptop with RTX 5090 Laptop GPU, 64 GB RAM, Intel U9 275H CPU.
- Remote environment: school cluster access is configured via SSH config; GPU jobs are expected to run through Slurm on `mlp1`.
- Sensitive credentials may exist outside the workspace and must never be copied into workspace files.

## Expected Output Style

- Prefer concise, technical writing suitable for a coursework report.
- Keep notes actionable and up to date.
- When unsure, optimise for report quality, reproducibility, and time-to-deadline.
