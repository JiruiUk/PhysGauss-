# H3 worst-frame repair experiments

This package preserves frozen PIDG/H3 weights and source. It introduces two
independently switchable changes, with no additional network parameters:

| Arm | `integrity_unit_rotation` | `integrity_hard_frame_probability` |
|---|---:|---:|
| R1 | true | 0 |
| R2 | false | 0.25 |
| R3 | true | 0.25 |

The covariance quaternion is `q_canonical + delta_q`. The released CUDA
`computeCov3D` does not normalize this sum. R1/R3 normalize it before rasterizing.
This changes the parameterization and must be tested with retraining; applying
normalization directly to an already optimized H3 does not preserve its covariance.
Nonunit inputs still produce positive semidefinite `M^T M`; their existence alone
does not prove the unique cause of an image artifact.

R2/R3 maintain per-training-frame foreground RGB L1 EMAs (decay 0.9). After8k,
25% of selection opportunities replace the original stack-sampled frame with a
random frame from the hardest25% training frames. The remaining75% retain the
original selection. No validation/test image, alpha or error enters these EMAs.
Sampler state is stored alongside the full optimizer/RNG checkpoint.

All arms branch from the existing identical, randomly initialized training3k
full state and train to50k total. They do not refine the completed H3 weights.
The existing H3 and C1 are reused. Fixed120/30/50 time-disjoint but interleaved
splits and shared `(frame_id-1)/199` clock remain unchanged.

`evaluate.py` first checked posthoc interpolation using training timestamps only.
Center interpolation worsened both mean and worst-five validation PSNR, including
frame84, so no interpolation prior/readout is enabled in the training arms.
The archived v0 full-state diagnostic also changed quaternion magnitudes; it is a
confounded exploratory check, not an interpolation-only result. The corrected
`state_lerp_raw` readout preserves those magnitudes.

`run.py` runs unit/CUDA checks, a100-step70/100 resume smoke test, commits the
milestone, then trains all three arms under an8-hour maximum wall-clock budget.
Only validation selects a candidate. No candidate is automatically called better
merely for reaching50k; H3 remains selected when the appearance/tail/coverage guards
fail. Final test exports start only after selections are locked. Historical test
exposure remains a limitation.

Outputs:

- `experiments/temporal_repair/diagnosis.md`: exposed-test vs validation diagnosis.
- `experiments/temporal_repair/monitor.html`:5-second progress and metric curves.
- `experiments/temporal_repair/selection.json`: validation decision and guard results.
- `experiments/temporal_repair/assessment.md`: completed final50k comparison.
- `runs/pidg_temporal_repair/render_exports/final50000/gt_h3_repairs/index.html`:
  GT/C1/H3/R1/R2/R3 concatenations, preserving original1600x900 source panels.

Baseline preservation is tested with new flags off at the real rasterizer; no
baseline file is edited. Pause via this queue's own `stop_queue.request`; state
is saved before the GPU is released. The8-hour deadline includes preflight.
