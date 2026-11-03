# Evaluator-only mask replay (before predictions)

The saved thin_static_visible is the clean-scene ray mask. Intersect it with
observed_static_visible to evaluate currently visible thin rays. Do not count a
hidden thin ray as an observed thin surface. Original archived audits unchanged.

Saved v2 GT does not contain the full observed-depth or empty-background masks.
Replay the frozen analytic trace in the independent evaluator only, check its
RGB and static-visibility arrays against saved GT, and use observed depth==0 to
define true empty background. Static_depth==0 alone also includes moving
surfaces and is NOT an empty-background mask. No inferred RGB/color threshold
is used. Freeze the unchanged generator source hash before inference.

Evaluate static-visible/thin denominators over ALL96 images, including those
without acceptable calibration. Also report observed-surface MAE (including
moving surfaces) as a distinct diagnostic. The declared static-use gate tests
whether raw valid prior depth could safely be used directly as static evidence;
failure on dynamic occluders does not refute observed-depth prediction.
