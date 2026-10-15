# PIDG Codex Execution Pack

This folder is designed to be copied into the root of a PIDG code repository and given to Codex or another coding agent.

## Files

- `AGENTS.md` — repository-level rules and architecture constraints.
- `CODEX_TASK.md` — detailed milestone-by-milestone implementation plan.
- `experiments/pidg_experiment_matrix.yaml` — machine-readable experiment ladder and benchmark plan.
- `experiments/RESULTS_TEMPLATE.csv` — standard table for collecting experimental outputs.

## Recommended Codex instruction

Use the following prompt after placing these files in the PIDG repository:

> Read `AGENTS.md`, `CODEX_TASK.md`, and `experiments/pidg_experiment_matrix.yaml` first. Start with M0 only. Inspect the repository, create `PIDG_REPO_MAP.md`, reproduce the original PIDG baseline, record exact commands and metrics, and stop after M0 with a summary of files changed and commands run. Do not implement M1 until M0 passes.

Then proceed milestone by milestone.

## Critical workflow

Do not ask Codex to implement M0–M10 in one pass. Recommended sequence:

1. M0 baseline reproduction
2. M1 deformation gradient
3. M2 Neo-Hookean module
4. M3 constitutive loss
5. M4 fixed-vs-learnable material ablation
6. M5–M6 multi-material/interface
7. M7 force
8. M8 staged optimization
9. M9 identifiability
10. M10 final benchmark automation

This makes failures attributable and preserves a clean ablation story.
