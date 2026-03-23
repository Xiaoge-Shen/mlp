# Progress Tracker

## How To Use

- Treat this file as the durable project status board.
- Update `status`, `evidence`, and `next action` after meaningful progress.
- Keep entries short and factual.
- If a plan changes, update this file first, then continue work.

## Overall Status

- Current phase: topic selection with a reasoning-RL code scaffold ready
- Deadline: 12:00 Friday 27 March 2026
- Time remaining at creation: 5 days
- Overall confidence: medium-low because the engineering scaffold exists, but the final project topic is not frozen yet

## Milestones

### 1. Research Question Freeze

- Status: in progress
- Goal: compress the project into one clear research question with one dataset, one metric, one baseline, and one main method.
- Evidence: the strongest current candidate is `Qwen2.5-0.5B on GSM8K: GRPO vs SAPO under fixed compute budget`
- Next action: decide whether to commit to the GSM8K reasoning-RL topic or switch to a lower-risk alternative today

### 2. Experiment Matrix Design

- Status: in progress
- Goal: define at most four experiments with a clear purpose for each.
- Evidence: a preliminary four-experiment structure exists for `base/instruct baseline`, `GRPO`, `SAPO`, and `budget analysis`
- Next action: freeze the final experiment matrix once the topic is confirmed

### 3. Data And Evaluation Setup

- Status: in progress
- Goal: fix dataset access, preprocessing, split, and metrics.
- Evidence: the new workspace includes GSM8K loaders, answer extraction, math verification fallback, and evaluation outputs for accuracy, tag success rate, and parse success rate
- Next action: install the runtime dependencies and verify that the evaluation script runs end-to-end on a small GSM8K subset

### 4. Minimal Reproducible Pipeline

- Status: in progress
- Goal: run a small end-to-end train/eval workflow locally.
- Evidence: a TRL-based scaffold now exists in this workspace with one training script for both `GRPO` and `SAPO`, YAML configs, W&B integration, and a notebook experiment driver
- Next action: install dependencies and run a 10-20 step smoke test locally

### 5. Baseline Result

- Status: blocked
- Goal: obtain the first credible result for the report.
- Evidence: the evaluation code exists, but no run has been executed in this workspace yet
- Next action: evaluate the chosen base model and, if needed, the instruct model on a small GSM8K slice

### 6. Main Method Result

- Status: blocked
- Goal: compare the core method against the baseline fairly.
- Evidence: the trainer supports both `loss_type=grpo` and `loss_type=sapo`, but no training result exists yet
- Next action: run the GRPO baseline first under a very small fixed budget

### 7. Ablation And Analysis

- Status: blocked
- Goal: explain why the method works or fails.
- Evidence: the experiment hooks exist, but no model outputs or curves exist yet
- Next action: choose a single additional factor such as training budget or completion length after the first GRPO and SAPO runs

### 8. Formal Cluster Runs

- Status: blocked
- Goal: run the final report-worthy experiments on `mlp1` via Slurm.
- Evidence: no cluster-specific script exists yet; the current scaffold is local-first
- Next action: only prepare Slurm once the local smoke test succeeds and the topic is frozen

### 9. Report Draft

- Status: blocked
- Goal: turn experiment progress into report sections before the final day.
- Evidence: there is enough structure to write methodology once the topic is frozen, but no results exist yet
- Next action: start a report outline immediately after the first baseline and RL runs

### 10. Submission Packaging

- Status: blocked
- Goal: produce the final coursework directory and zip without missing files.
- Evidence: final report and source tree do not exist yet
- Next action: keep the naming rules visible and leave packaging until the last stage

## Blockers

- The final project topic is still not frozen.
- The runtime dependencies for this new workspace are not installed yet.
- No experiment has been executed in this workspace yet.
- The group ID is unknown.

## Next Critical Input Needed

- Confirm whether to commit to the GSM8K `GRPO vs SAPO` topic
- Install the runtime stack from `requirements.txt`
- Run one local smoke test and one baseline evaluation
