# Progress Tracker

## How To Use

- Treat this file as the durable project status board.
- Update `status`, `evidence`, and `next action` after meaningful progress.
- Keep entries short and factual.

## Overall Status

- Current phase: BFCL experiment implementation with report and code scaffold ready
- Deadline: 12:00 Friday 27 March 2026
- Overall confidence: medium because the project framing is fixed, but no BFCL benchmark results exist yet

## Milestones

### 1. Research Question Freeze

- Status: completed
- Goal: fix one clear question, one benchmark family, and one main method.
- Evidence: the final topic is now `BFCL: budget-aware escalation policy over direct calling, clarification, and one-step repair for small open models`
- Next action: none

### 2. Experiment Matrix Design

- Status: in progress
- Goal: define the final four-system comparison and target BFCL subsets.
- Evidence: the planned systems are `Direct`, `Clarify`, `Repair`, and `Escalation`, with current target subsets `missing_parameters`, `multi_turn_base`, and `memory`
- Next action: freeze the exact subsets before large runs

### 3. Data And Evaluation Setup

- Status: in progress
- Goal: connect the local policy runner to a real exported BFCL subset and the official BFCL evaluator.
- Evidence: `scripts/run_bfcl_study.py` generates the official baseline plan, and `data/bfcl_normalized/sample.jsonl` documents the normalized local input schema
- Next action: export a real BFCL subset into `data/bfcl_normalized/`

### 4. Minimal Reproducible Pipeline

- Status: completed
- Goal: have a runnable local scaffold before full experiments.
- Evidence: BFCL scripts compile, run-plan generation works, and summary generation works on empty outputs
- Next action: run the first live smoke test against a reachable endpoint

### 5. Baseline Result

- Status: blocked
- Goal: obtain the first official BFCL direct baseline.
- Evidence: `outputs/bfcl/bfcl_qwen3_budget_policy/run_plan.md` exists, but no official score CSV has been produced yet
- Next action: run `bfcl generate` and `bfcl evaluate` for the direct baseline

### 6. Main Method Result

- Status: blocked
- Goal: compare `Escalation` against `Direct` fairly.
- Evidence: the local policy runner exists, but no BFCL evaluation result has been attached to it yet
- Next action: run one local policy smoke test, then wire the outputs into the BFCL evaluation workflow

### 7. Ablation And Analysis

- Status: blocked
- Goal: explain when clarification or repair helps and what the cost is.
- Evidence: trace logging exists in `run_bfcl_policy.py`, but no traces have been collected from real runs
- Next action: add one first-pass analysis after the initial smoke tests

### 8. Formal Cluster Runs

- Status: blocked
- Goal: move longer BFCL runs to the school cluster if local serving becomes the bottleneck.
- Evidence: no cluster-specific BFCL script exists yet
- Next action: only prepare cluster automation if the local setup becomes too slow

### 9. Report Draft

- Status: in progress
- Goal: keep the paper draft ahead of the experiments.
- Evidence: `report/main.tex` and `report/references.bib` are already in place
- Next action: update the Methods and Experiments sections after the first real run

### 10. Submission Packaging

- Status: blocked
- Goal: package the final report and code cleanly.
- Evidence: submission directory has not been prepared yet
- Next action: postpone until results stabilise

## Blockers

- No official BFCL run has been executed yet.
- The normalized sample file is only a schema example, not real benchmark data.
- The exact model/backend pairing still needs to be confirmed at runtime.
