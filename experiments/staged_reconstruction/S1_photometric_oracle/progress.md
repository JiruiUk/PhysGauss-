# S1-D5 evaluator-only photometric audit

Status: evaluator implemented/tested/frozen; full measurements pending.
D4 complete/rejected; all192 candidates' saves and original matcher replay
audited, no Gaussian training. See S1_rgb_prior_proposals/v1/assessment.md and
final_review.json. Do not repeat D4/D3/D2, change thresholds or swap Small.

The isolated photometric_oracle package reuses the original patch matcher and
D4 acceptance function. 28 tests (14 D5 / 14 inherited D4) pass, including
default-off identity/CLI, analytic scene/occlusion, grid conversion, grouping,
unit invalid cost, and separate-process aggregate save/replay. No PIDG/Gaussian
training path changes, so no training smoke or Adam recovery claim is made.
Initial26 tests passed; adding aggregate recovery exposed the existing pytest
temporary-directory access issue, then a missing parent directory. All failed
logs are retained; task-specific basetemp and creating its parent resolve it.
No matcher, fixture threshold, production source, or protocol was changed.

Freeze contains730 source/parent/data files, including the full original v2
dataset and original D1 outputs. SHA256:
1cb17a8cf5353a008cba57cc0cf55367e9668b2cfe19a342930055526eb003ec.
Only aggregate statistics/per-view JSON and declared fixed color images leave
the evaluator. Distribution quantiles use fixed .01 cost/.02 difference bins;
means/counts are exact sums. Cross-groups partition candidates, while all6 is
explicitly a subset of fit3+bothheld visible. Missing candidates remain unknown.

Next execute this one fixed two-condition diagnostic. Actual process check:
only unrelated serve_integrity_monitor.py; no D5 lock/process/stop request.
No subagents. No source/weight/history rewrites or paused task restoration.
GT stays evaluator-only; no inferred depth/normal/mask/selected-point assets.

The analytic plane controls explicitly revisit the fractional interpolation
bias from D4's preserved initial unit-test log, rather than hiding it by the
integer-shift test control. Neither test fixture nor oracle geometry can be
used as a real training prior. Actual fixed images and numeric denominators
must be reviewed before drawing a cost/occlusion/search conclusion.
