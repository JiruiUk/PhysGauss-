# Native train reliability and recurring anchors: PASS for graph feasibility only

The exact user-selected S3-U final6k parent was audited at all120 observed
training times using the unchanged native1600 renderer. Both independent
1536-probe estimates per frame were checked against192 population-spanning
one-hot sample rows. Every frame passed the registered measurement gates.
Independent verification recomputes exact-support errors from saved arrays;
the median across frame/seed medians is2.3257%,
and the worst frame/seed median is3.3670%.
Original native RGB/alpha are exact; all model tensors and global RNG remain
unchanged. Actual optimizer updates:0. Test RGB read:False.

## Registered selection results

Spatial candidates use voxels twice the original initialization spacing.
Qualified top2% ranks are computed separately for both seeds, then candidates
with at least5 significant frames are intersected. No absent observation is
filled to meet a count. Physical body count is not inferred.

- Seed anchor counts:[2214, 2212].
- Stable intersection:2195 /15,507 dynamic rows.
- Minimum currently trusted stable anchors in an observed frame:77.
- Anchors recurring at least5 times within early/middle/late thirds:[860, 739, 734].
- Independent per-frame selection Jaccard: minimum0.9558,
  median0.9849.
- Fraction of qualified model points within the registered radius of a stable
  anchor: minimum96.6346%,
  median100.0000%.
  This describes S3-U support and is not GT surface coverage.

|Registered gate|Passed|
|---|---|
|all_native_measurements_pass|True|
|independent_selection_jaccard|True|
|stable_anchor_count|True|
|trusted_anchor_count_each_frame|True|
|recurring_in_each_temporal_third|True|
|observed_model_support_neighbor_coverage|True|

## Interpretation and next action

This establishes which color-supported point rows can be tested as motion
anchors. It does not prove their depth or material identity. Graph feasibility
is distinct from training preflight and the complete-object goal. No graph loss
or new model has been trained in this milestone. The full6x9 then6x200 novel-view
acceptance remains unfulfilled; no model is promoted.

Any following motion intervention must retain uncertain depth and avoid claiming
that nearby rows belong to one object. It needs separate opt-in tests, native
flags-off parity, actual short training/recovery checks, matched continuation
controls and actual novel-view image review. I10's numerical repeat sensitivity
still prevents interpreting a single PSNR difference as a stable causal effect.

Evidence: `anchor_selection/assessment.json`, `observed_native120/result.json`,
`verification_native120.json`, `observed_native120_protocol.json`,
`output/staged_reconstruction/uncertainty_motion/anchor_selection/anchors.npz`.
Source audit, adaptations and failed low-sample/wrapper attempts are retained
in `research.md`, `measurement_assessment.md` and versioned protocol/log files.
