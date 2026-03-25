# Progress Tracker

## How To Use

- Treat this file as the durable project status board.
- Update `status`, `evidence`, and `next action` after meaningful progress.
- Keep entries short and factual.

## Overall Status

- Current phase: BFCL direct baseline complete; 200-turn verifier-based guard pilot completed; consolidating mixed-result analysis into the report
- Deadline: 12:00 Friday 27 March 2026
- Overall confidence: medium because the benchmark path is working, the direct baseline is stable, and the 200-turn guard pilot now provides a concrete mixed-result finding

## Milestones

### 1. Research Question Freeze

- Status: completed
- Goal: fix one clear question, one benchmark family, and one main method.
- Evidence: the project is now framed as `a lightweight verifier-based clarification guard and escalation policy for under-specified BFCL multi-turn cases`
- Next action: none

### 2. Experiment Matrix Design

- Status: completed
- Goal: define the final systems and target BFCL subsets.
- Evidence: the main systems are now `Direct`, `Clarify-Guard`, and `Escalation`; the main subsets are `multi_turn_miss_param`, `multi_turn_base`, and `memory_kv` as a negative control
- Next action: keep `Repair` optional only

### 3. Data And Evaluation Setup

- Status: completed
- Goal: attach turn-level diagnostics to the official BFCL output and keep the local policy runner usable for pilots.
- Evidence: `scripts/analyze_miss_param_turns.py` now reads the official BFCL miss-param result and `scripts/export_bfcl_miss_param_turns.py` exports real turn-level pilots
- Next action: none

### 4. Minimal Reproducible Pipeline

- Status: completed
- Goal: have a runnable local scaffold before full experiments.
- Evidence: local BFCL scripts compile, official BFCL baseline runs, and the miss-param diagnostics script runs successfully
- Next action: none

### 5. Baseline Result

- Status: completed
- Goal: obtain the first official BFCL direct baseline.
- Evidence: official BFCL score CSVs exist in `outputs/bfcl/bfcl_qwen3_budget_policy/official/direct/score/`
- Next action: treat `multi_turn_base = 27.50%` and `multi_turn_miss_param = 18.00%` as the core baseline comparison

### 6. Main Method Result

- Status: completed
- Goal: compare `Escalation` against `Direct` fairly.
- Evidence: the 200-turn pilot metrics now exist for `direct_200`, `clarify_200`, and `escalation_200`, with the verifier-based guard reducing harmful-call rate on empty turns from `66.67%` to `14.58%` and `18.75%`
- Next action: present `Clarify-Guard` as the strongest pilot variant and treat `Escalation` as a secondary comparison rather than chasing more policy complexity

### 7. Ablation And Analysis

- Status: completed
- Goal: explain when clarification helps and what the runtime cost is.
- Evidence: on the 200-turn exported miss-param pilot, `Clarify-Guard` cuts harmful-call rate on empty turns to `14.58%`, but raises no-call rate on required turns to `59.21%`; latency also rises from `0.52s` to about `0.93s`
- Next action: present this as a mixed robustness-cost trade-off in the final report

### 8. Formal Cluster Runs

- Status: blocked
- Goal: move longer BFCL runs to the school cluster if local serving becomes the bottleneck.
- Evidence: local runs are currently viable
- Next action: only prepare cluster automation if full policy runs become too slow

### 9. Report Draft

- Status: completed
- Goal: keep the paper draft ahead of the experiments.
- Evidence: `report/main.tex` now reflects the verifier-based guard, the 200-turn pilot trade-off table, and builds successfully to `report/main.pdf`
- Next action: only fill final author/group metadata and do final stylistic polishing

### 10. Submission Packaging

- Status: blocked
- Goal: package the final report and code cleanly.
- Evidence: submission directory has not been prepared yet
- Next action: postpone until results stabilise

## Blockers

- The local policy runner still uses exported turn-level BFCL pilots rather than the official BFCL evaluator.
- The current guard is useful for reducing harmful calls but still over-clarifies many required turns.
- The final report now needs tightening and honest interpretation more than additional method complexity.

- Key comparison file: `outputs/bfcl/bfcl_qwen3_budget_policy/analysis/policy_200_comparison.json`
