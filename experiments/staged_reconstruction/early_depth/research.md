# I4 early geometry prior: fresh optimization, identical S3-U initial state

I3 improves fixed pseudo-depth ordering but retains severe novel-view sheets.
The next hypothesis changes when geometry is introduced, rather than sweeping
the prior weight. Use original S3-U warmup step-zero full state and16k+6k
schedule, with original width/camera schedule and optimizer. Add only the same
training120 ordinal prior used by I3; omit I1's parent-derived graph so the
original S3-U is the matched no-depth control. This is not a single-switch
timing ablation relative to I3 because I3 also retains the I1 graph prior.

MoDGS (<https://arxiv.org/html/2406.00434v4>) motivates considering geometry
from the beginning. This does not reproduce its3D-aware initialization or
invertible architecture: the original S3-U positions and all original PIDG
fields remain. The ordinal prior and its .03 weight,100-update ramp,
alpha10 in scene units, pairs/masks and800px auxiliary depth rendering stay
unchanged. MoDGS's published normalized-scene alpha100 differs; no equivalence
of these hyperparameters is claimed.

Budget is fixed before training at16,000 warmup and6,000 physics steps. The
original step-zero checksum, full Adam/RNG, train camera order, frozen static
and image/physics objectives are audited. The original S3-U final6k remains
the user baseline; a trained intermediate is not substituted. A zero-feature
smoke control verifies the inherited training path and avoids invoking any
depth loss. A200-step ordinal smoke checks independent40→80→200 recovery.

At16k the model saves and pauses for actual native-image review plus original
warmup numerical guardrails before physics. Final completeness acceptance is
the unchanged six-camera screen then6×200 sequence review. No test RGB enters
iterative model selection. Learned depth remains a model prior, not GT.
