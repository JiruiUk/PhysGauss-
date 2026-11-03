# S1-D3: fixed pretrained RGB depth, before Gaussian integration

User selected Depth Anything V2 Small. This stage measures whether its learned
observed-surface geometry adds accurate coverage beyond the sparse D2 support.
No static category, mask, number of objects or GT enters inference. A moving
occluder is an observed surface too: a depth prediction does not establish a
static identity. Black/infinite background must be audited, not color-masked.

Use the pinned official Small implementation, its unchanged BGR infer_image
preprocessing at518 and verified official weights. No dependency replacement,
finetuning, model-size comparison, augmentation or threshold search. Data-only
opt-in module, with exact flags-off object identity; original PIDG unchanged.

The relative inverse-depth output is affine aligned per view to D2 tight
RGB-inferred anchors. Deterministic spatial fit/check blocks separate fitting
from acceptance. Fit Theil-Sen using at most256 fit points, with positive slope
and nondegenerate anchor geometry. Held anchors check extrapolation locally;
they are NOT independent GT and cannot certify the rest of the image. Missing
or rejected calibration remains unknown. No alternate alignment on failure.

Save raw relative output even on calibration failure. Valid aligned depth is
positive and inside inherited camera-derived near/far; do not clamp predictions
into this range. Report a fixed reference-grid diagnostic for every image.
Use only evaluator GT to measure static-visible accuracy/coverage, thin rays,
dynamic-front and empty-background false static geometry. Also report temporal
variation at repeated camera poses on evaluator-visible static pixels. No
per-image GT alignment and no GT-driven point selection or thresholds.

Both conditions must pass the protocol evidence gate before proposing a
separate matched Gaussian experiment. This stage does not run Gaussian
optimization, recover hidden surfaces, measure real-scene artifacts or generate
new86/188 videos. Any subsequent Gaussian protocol requires native gradients,
flags-off and independent full Adam/RNG/camera recovery and independent static
completeness checks. Prior numerical improvement cannot bypass these gates.
