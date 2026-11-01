# Validation review, revision, and retraining

This worker observes the active completeness queue every5 seconds and records
each new validation checkpoint in `observations.jsonl`, including a matched-
iteration C1 comparison. It shares the existing absolute eight-hour deadline,
2026-11-01 05:17:35 UTC; it never resets the budget or retrains the baseline.

After the four20k screens, a pure tested policy reads validation metrics and
generates `proposals.json` with evidence, source configuration, changed module
weights, reasons and source-selection hash. It waits until the existing queue
finishes GPU work before starting its own GPU preflight and training. The
running experiment and its frozen source/configuration are not edited.

If high-alpha color defects dominate alpha holes, the new opt-in component is
the mean worst20% RGB error among adequately covered foreground pixels
(rendered alpha>=0.8; detached gate), weight0.01, every4 steps with interval
compensation, starting8k with4k ramp. This emphasizes actual training RGB
appearance in opaque regions. Smoke tests use alpha>=0.05 to exercise the
gradient before coverage has converged. No semantic/instance labels or
new material/force parameters are learned. Coverage/strain weights are reduced
when validation image quality or background spill deteriorates. If validated
density is retained, its per-window clone cap is reduced to128.

All follow-ups start from the same full C1 random-init3k state, with the
original50k schedule and120/30/50 split. One20k feedback screen and at most
one measured conservative retry are allowed. Only a candidate satisfying the
four-image-metric/spill guards AND improving the source20k completeness score
can continue to50k. At most one additional50k run is promoted. If evidence
does not support a change, the worker stops without duplicating the baseline.

CPU loss/policy tests precede runtime integration. A real raster-color CUDA
gradient check,100-step training,70-step resume and validation-PNG parity must
pass before any feedback20k run; a separate milestone is committed afterward.
Only validation chooses revisions or checkpoints. The final test opens after
the promoted model's selections are locked; no new revision is selected from
test results. Reports retain the single-scene/seed, historical test exposure,
and2D-proxy limitations. No model is automatically substituted for the C1
baseline when results regress.

The final report explains what changed and compares validation metrics with
C1 and the source model. Exports include GT/render pairs and, when a source
50k model exists, GT | C1 | before | after panels. The unified local dashboard
is `experiments/training_monitor/index.html` on the existing loopback server.
