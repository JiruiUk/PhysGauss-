# R0 progress — 2026-11-03

The accepted partial static pool has been bound to the immutable common15k
state.32 related unit tests passed. Native gradient/Adam, actual dynamic-CMR
exclusion, flags-off120-update repeat comparison, enabled120-update smoke,
15060→15120 full-checkpoint recovery and native restored geometry invariance
passed. The off native RGB RMS0.00178145 and alpha RMS0.00364371 are within
the predeclared3x unchanged-repeat envelope (0.00117030 and0.00258489).
This is numerical parity, not a cross-process bitwise determinism claim.

No formal R0 training result is available yet. The latest completed experiment
is scratch-integrity v2; it is preserved unchanged. Full R0 training begins only
after the implementation checkpoint commit. The full pipeline will then evaluate
both final50k states and update this plan's numeric status; native manual review
will determine the evidence-based next step. Thread follow-up automation `pidg`
checks for meaningful changes every10 minutes and does not start extra candidates.
