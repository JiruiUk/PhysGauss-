# S1-D5 evaluator-only photometric audit

Status: complete and independently reviewed; no reconstruction/training output.
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

Two-condition192-view measurement and independent replay completed3347ad67.
All423747/423100 parent candidates counted;421292/420623 have reference-visible
static GT. Continuous truth passes original acceptance17268/17324 (4.10/4.12%),
nearest same192-grid truth0/0. Even same-face/all6-visible/gross-error points pass
only13/27723 and12/27604. Fractional analytic plane still minimizes at4.04947
instead of4; integer control3.99303. Quantization/visibility alone do not explain
the objective mismatch. D1 fit optimization creates selection bias; no claim
that these diagnostics prove one cause or that a new warp has already helped.
Actually viewed six fixed color figures and all analytic control curves.
No GT pixel arrays or training assets exported. See v1/assessment.md and
v1/independent_audit.json. D5 must not restart; no Gaussian/real continuation.
Next single mechanism is the separately preregistered RGB plane-warped support
candidate S1-D6, with data-derived reversible local slopes, all parent points
retained and unchanged strict geometry/Gaussian admission gates.

Original launch process check: only unrelated serve_integrity_monitor.py;
no D5 lock/process/stop request. Completed evaluator releases its CUDA process.
No subagents. No source/weight/history rewrites or paused task restoration.
GT stays evaluator-only; no inferred depth/normal/mask/selected-point assets.

The analytic plane controls explicitly revisit the fractional interpolation
bias from D4's preserved initial unit-test log, rather than hiding it by the
integer-shift test control. Neither test fixture nor oracle geometry can be
used as a real training prior. Actual fixed images and numeric denominators
must be reviewed before drawing a cost/occlusion/search conclusion.
