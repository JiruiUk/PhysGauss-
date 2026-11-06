# L7-P: Rubber Duck RGB pseudo-depth admission before a weak constraint

User authorizes scale stability, motion-compensated consistency and occlusion
boundary checks, then a revocable robust weak constraint only on reliable data.
Retain J6; do not use failed L4/L5/L6 descendant weights. Fixed official Depth
Anything V2 Small and518 preprocessing, RGB only, no new masks/count/GT.

First run one default-off read-only audit on all120 observed training images.
Relative inverse depth has no metric scale. Calibrate each image using existing
RGB stationary-compatible triangulations, spatial fit/check separation and
spatial deduplication. These are stationary hypotheses, not static labels or GT.
No calibration to the parent rendered depth, alpha segmentation, D3 aligned
depth, validation images, GT geometry or oracle assets. Missing/degenerate
calibration stays unknown; raw maps and all observations remain saved.

Inspect coefficient variation AND geometric stability under a deterministic
fit-anchor split. Raw coefficient variation alone is not physical scale drift.
Check held anchors and support extent. Only local interpolation is admissible;
do not extrapolate a calibration across the full image for training.

Use unchanged T4 own-visible reciprocal motion bindings as hypotheses. At each
observed UV backproject prior depth to world coordinates. Subtract the exact
J6 point's time-varying world position; fit a constant offset using even
observations, check odd observations. Also report camera-only compensation
without motion correction and apparent camera-Z variation. Parent compensation
is an assumption, not independent validation or evidence of correct geometry.
Do not reassociate a track using depth or change original binding gates.

Depth discontinuity in5x5 is a boundary proxy, not GT occlusion. Audit all
original4433 tracker proposals including invisible/rejected observations:
report forward/reverse visibility and closure failure stratified by prior
boundary, plus all off-image/nonfinite/missing cases. No rejection is called
background/dynamic. Prior jumps and missing calibration suspend depth use.

The protocol declares a LOCAL evidence gate, not old T4 global promotion:
30 reliable motion tracks,25 distinct rows,5 track starts per temporal third;
30 calibrated views,8 per third; all original T4 global failures remain.
This audit cannot establish true thickness/full surface coverage. If failed,
save the default-off weak loss and tests but do not run Gaussian training.
If passed, separately fix a matched full-J6 RGB vs RGB+weak-depth protocol,
native flags-off/gradients and complete twoAdam/RNG/camera/physics/controller
recovery before any2000-step arms. Keep original .3dB and actual86/188 cleanup
gates. No spring stacking, sweep, threshold relaxation or automatic extension.

Freeze all sources/inputs before inference; independently reload the model and
replay fixed0/24/72/119 raw maps and all CPU decisions. View these four full-RGB
panels and all three temporal-band summaries. Every milestone separate commit.
J6 historical color-conditioned lineage remains disclosed; this audit uses no
new color class, semantic labels, object counts, materials, forces or physics.
