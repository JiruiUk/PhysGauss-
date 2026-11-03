# R0 progress — 2026-11-03

The accepted partial static pool has been bound to the immutable common15k
state.32 related unit tests passed. Native gradient/Adam, actual dynamic-CMR
exclusion, flags-off120-update repeat comparison, enabled120-update smoke,
15060→15120 full-checkpoint recovery and native restored geometry invariance
passed. The off native RGB RMS0.00178145 and alpha RMS0.00364371 are within
the predeclared3x unchanged-repeat envelope (0.00117030 and0.00258489).
This is numerical parity, not a cross-process bitwise determinism claim.

Implementation and preflight were committed as `23f53c35`. Formal R0 pipeline
started at 2026-11-03 04:46:16 UTC, PID265148; controller PID270036 and H0 trainer
PID270724. Both branch receipts bind the same full15k checkpoint hash. H0 runs
first, followed by H1, each targeting50k with35000 actual updates. Launch metadata:
`background_launch.json`; live state: `runs/pidg_static_pool/pipeline.json`,
`controller.json` and `active.json`.

No formal R0 result is available yet. The latest completed experiment is
scratch-integrity v2. The pipeline will evaluate both final50k states and export
native fixed-view videos; actual image review will determine the next step.
Thread follow-up automation `pidg` checks for meaningful changes every10 minutes
and does not start extra candidates.
