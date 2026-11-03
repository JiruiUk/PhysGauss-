# Canonical support and adaptive density: read-only qualification

Primary source read: [3D Gaussian Splatting, Sec5.2](https://arxiv.org/html/2308.04079v1#S5.SS2).
It interleaves geometry optimization with adaptive population changes, uses
screen-position gradients to propose cloning/splitting, and cautions that
floaters can grow spuriously. A gradient threshold from its setup is not
automatically calibrated for PIDG's loss, dynamics or image resolution.

[4D-GS, Sec4.3 and AppendixA.1](https://arxiv.org/html/2310.08528v3#S4.SS3)
supports the importance of canonical initialization. Its appendix also
acknowledges camera/motion correlation minima in monocular reconstruction.
It does not establish that extra Gaussians solve that ambiguity or guarantee
complete surfaces. The paper text search did not specify a density rule to
copy; do not attribute such a rule to it.

## Verified repository distinction

`third_party/PIDG/train_pidg.py:245-254` collects view-space gradients and can
invoke `densify_and_prune`. Its Gaussian model implements clone/split/prune and
optimizer-state changes. In contrast, `union/prepare.py` explicitly sets
`densification: false`. S3-U and I8/I9 retain the15,507 dynamic centers initialized
from the earliest training image; there are no population changes. That was
the selected protocol, not a newly identified runtime defect. Original PIDG
particle index bookkeeping must not be interpreted as verified physical identity.

## Independent diagnostic, no new training arm

`scripts/audit_canonical_support.py` loads only pinned checkpoints and reads
all120 training images at width800. It records original reconstruction-objective
screen-position gradients, rasterizer radius admission, dynamic-only alpha on
eroded observed foreground, canonical nearest-point distances and proximity to
the existing coordinate clamp. It verifies every model tensor and snapshot
unchanged, saves per-point arrays/hashes, performs no density operations and
does not read test RGB. The threshold0.0002 is descriptive only, not a gate or
a selected training hyperparameter. Radius admission is not physical visibility.

Compare the selected S3-U final6k, failed I8 final16k and immutable I9 step1000
as diagnostic cases with different budgets/architectures. I9's early snapshot
cannot establish a matched-budget effect. Observed coverage deficits might
support evaluating population adaptation later, but high observed coverage
with broken new views instead points to unresolved geometry/motion ambiguity.
Neither outcome alone establishes a causal bottleneck. Do not change live I9,
mix initialization/population/strain changes, or replace the original appearance,
static-preservation and actual native completeness gates.
