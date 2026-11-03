# S1-D2: source-envelope uncertainty, one predeclared candidate

D1 reduced quantization noise but left about20.3% same-ray static-depth errors
outside0.05. Audit whether the original six source views expose those errors
before using any depth prior in Gaussian training. This is an opt-in RGB-derived
geometry-evidence module, not new physics, object attribution or a reproduction
of a paper's calibrated uncertainty model.

Fix all D1 candidate pixels and depths. For each, project into the original six
source views, nearest-sample their saved D1 depth, and backproject that pixel
to the reference camera. No new RGB match search or candidate growth/pruning.
Missing/out-of-view/invalid depths supply no evidence and never a free-space
vote. A closer sampled surface is possible occlusion: widen the envelope and
mark its availability, never infer the candidate is dynamic or absent.

Construct an empirical interval containing the D1 center and all valid
source-returned reference depths, padded by half of the local192-level inverse
depth grid spacing. It is an empirical envelope, NOT a95% posterior interval.
Repeated timestamps at identical camera poses are not independent perspectives;
count distinct rounded calibrated poses separately. Record source reprojected
pixel errors, temporal span and possible closer-surface counts without labels.
Confidence is only a bounded evidence score, not a probability of staticity.

A fixed tight-evidence subset requires>=3 distinct source camera poses,
>=20 inherited frame-index time span including the reference, no observed
closer-surface ambiguity, maximum returned reprojection error<=1.5px and
interval width<=0.04 scene units. These values are fixed before outputs; never
relax or sweep them based on GT. Unknown/wide candidates remain stored.

Independent evaluator GT is opened only after data-only inference. Diagnose
all-candidate interval coverage AND width, then tight-subset interval coverage,
same-ray center errors, thin/visible coverage and error distributions. Also
report correctly rejected uncertainty vs confidently wrong depth. Gate for
considering a separately predeclared paired Gaussian interval-loss experiment:
both cases must retain>=25% of original candidates, cover>=25% of original
supported visible static/ thin pixels, have>=95% GT-depth interval coverage,
<=2% tight candidates with no static surface or center error>0.05, and center
static-depth MAE<=0.02. Empty support is failure; broad intervals alone cannot
pass. See protocol.json for machine-readable unchanged values.

Tests cover geometry, missing/occluded samples, duplicate camera independence,
temporal insufficiency, width/reprojection bounds and exact flags-off depth
identity. Freeze code/protocol and parent outputs before one two-condition run.
Keep per-view receipts, state, stop/lock protection and independent evaluation.
No Gaussian training in this stage. A successful evidence gate only permits
designing one paired robust interval-depth synthetic experiment, with a new
fixed protocol, native gradients/flags-off/full-state resume and independent
static-GT/image/geometry/thin gates before any real continuation.

Do not restart S1-C/N/D1, add blanket depth/normal/alpha-one losses, modify
frozen weights/sources, tune thresholds, use GT to select training pixels,
or resume paused work. Preserve S3-U unified dynamic pool and unknown identity.
If this evidence cannot separate wrong basins with useful thin coverage, record
the measured limitation before deciding whether an independent geometric prior
or additional observations need a user choice. No claim of data impossibility
or 3DGS artifact reduction follows from this diagnostic alone.
