# 2026-11-03: R0 static-pool implementation and preflight

User authorized execution of the revised experiment plan and ongoing updates
based on completed results. Latest scratch-integrity v2 was independently
audited:26 training-evidence source/copy hashes, both full final checkpoints,
35000 matched camera samples,3500 finite geometry events, recomputed per-frame
metric and constraint means, and5231 output hashes agreed. Fixed-view native
images still fragmented despite strong local-loss improvements.

Created `experiments/adaptive_reconstruction_plan.md` and immutable
`experiments/static_pool/plan.json`. R0 is a separate reconstruction branch;
original PIDG and the M0–M10 physics order are preserved. No free force,
material learning, learned segmentation, extra Gaussian or soft transfer loss
is introduced.

Implementation: `variants/static_pool`. A minimal C1 fork changes only its
Tuning import, renderer entry policy and dynamic CMR sampling domain. Accepted
static rows use fixed reference deformation/covariance offsets; their canonical
Gaussian geometry, opacity and SH remain learnable. This is a logical pool
within existing storage, not a new model-size budget. Unknown/dynamic rows
retain original warps. Training, validation and inference share the same policy.

Bound14624 of88916 rows from the immutable scratch common15k state
`db9560a4e6a62324bb98933da2ec6d21949bdb8619111b939382f80af9c6bb21`.
Reference captures use only existing training-best-support frame choices.
Binding SHA:
`2aa9c7f68696763eedda0d0d4c8ffdccbeba7faa67a13a7a6d830347be6fb492`.
Known static rows cover60.68% of blue-core contribution on average,47.86% in
the middle third. They are a partial conditional static hypothesis, not GT.

32 related tests passed. Real1600×900 GPU checks proved finite trainable static
geometry/appearance gradients and Adam updates, zero static output-deformation
gradients, retained dynamic deformation gradients and exclusion of selected
rows in the actual CMR function. Source checkpoint remained unchanged.

Flags-off, original and original-repeat each completed120 actual updates from
the same15k state, including the15100 CMR event. Parameter RMS gates and exact
off RNG/camera-state checks passed. Native RGB/alpha differences were within
3x original-repeat RMS+1e-7. Inert screen-radius statistics are excluded after
densification has ended; all live buffers and parameters retain their gates.
Enabled smoke completed15000→15060→15120 with full optimizer/RNG restore.
Restored selected-static centers/covariances were exactly invariant across
five query times. Preflight receipt: `experiments/static_pool/preflight.json`.

Formal matched H0/H1 arms are planned to finish at50k (35000 new updates each),
only `hard_static_pool` differing. No LR reset or historical final-state
initialization. Final states are selected in advance, with native observed
val30/test50 and three fixed cameras over200 times. Training-image tracks held
from triangulation are explicitly not held-out model geometry GT.

Pipeline: train→metrics/static diagnostics→native sequences→full video decode→
numeric report. A native visual review and plan revision are required before
promotion. Thread heartbeat `pidg` checks every10min, stays quiet while unchanged,
and will review completion/failure and revise the plan without weight sweeps or
automatically adding new physics. Separate implementation and result commits.

Implementation/preflight committed as `23f53c35`. Hidden formal pipeline launched
2026-11-03 04:46:16 UTC (PID265148), with matched source receipts and only the main
flag differing; initial H0 trainer PID270724. The pipeline runs both arms serially
before evaluation. No formal training result or promotion is claimed at launch.
