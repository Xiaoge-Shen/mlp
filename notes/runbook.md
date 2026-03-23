# Runbook

## Purpose

This file is the durable operations note for running experiments locally and on the school cluster.

## Local Workflow

- Use the local machine for fast iteration, debugging, preprocessing, and smoke tests.
- Record exact commands used for report-worthy experiments.
- Keep seeds and hyperparameters explicit.

## Cluster Workflow

- Use SSH config to connect to the school environment.
- Prefer Slurm jobs for GPU experiments instead of manual long-running shells.
- For each formal run, record:
  - code revision or snapshot
  - command
  - dataset split
  - output path
  - wall time
  - GPU type if relevant

## Packaging Rule

- Final submission source folder must exclude logs, datasets, and model weights unless explicitly required by the brief.

## Pending Details

- Project code layout
- Actual training and evaluation commands
- Slurm job script template
- Result collection path
