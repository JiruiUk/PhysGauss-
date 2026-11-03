# S1-D6: one RGB local-plane-warped support candidate

D5 found that original normalized-patch cost rarely admits even privileged
true geometry, including same-reference-face/all6-center-visible large-error
rays. Fractional analytic plane shifts bias its minimum; quantization and
center visibility alone do not explain the measured gap. These are motivation,
not proof that plane warp will help. D1 already optimized original fit cost.

Change one mechanism: how source5x5 RGB support is aligned and normalized.
Infer a reversible local inverse-depth slope from original D1 neighboring
depths, then project all25 reference rays under that local plane and bilinearly
sample raw source RGB before patch normalization. Do not interpolate already
normalized source feature vectors. No GT plane, normal, visibility, mask,
object count, Small alignment or semantic/static label enters inference.
Fixed Small code/weights/outputs are retained; this is a matching candidate,
not a replacement pretrained model or a direct prior loss.

Slope inference is fallible: split original5x5 D1 neighbors into deterministic
fit/check parity groups, excluding the central candidate. Least-squares fit
inverse depth, require nondegenerate support and fixed held depth residual
checks. Missing/contradictory support means no proposal and unchanged parent,
not a static/dynamic/background decision. Even admitted slopes are proposals,
never GT normals. Keep exactly every original candidate, with no point growth.

For admitted slopes search the original192 inverse-depth grid. Evaluate parent
and proposal under the SAME new warped cost, fit best3 of original4 sources,
and require original two held costs nonworse plus one strictly improved with
the unchanged1e-6 numeric tolerance. Do not compare incomparable old/new costs
to accept points. Default-off returns the parent unchanged. There is no extra
normal optimization, window/model/threshold sweep or continuous refinement.

Prerequisites: unit geometry/gradient/degenerate/identity tests, flags-off, source
and training-only input freeze, separate-process complete inference recovery.
Then one serialized96-view run per diffuse/glossy condition; independent GT
evaluation follows frozen outputs. Reuse the three fixed D5 analytic controls
and fixed0/24/72 images, with cost/acceptance and geometry diagnostics. No
production candidate is chosen by control or GT results. Retain failed logs.

Numerical improvement and Gaussian evidence gates are copied exactly from D4,
not loosened to suit the new method. A limited improvement remains inference
only; neither it nor privileged cost results allow Gaussian/real training.
Any later Gaussian stage needs its own matched protocol, native gradients,
flags-off and independent full Adam/RNG/camera recovery, and two-condition
static GT/structure gates before real continuation. Keep S3-U and unknowns.

Literature: PatchMatch Stereo BMVC2011 sections1/2.1/2.2 motivate plane-aligned
support; this is a restricted adaptation with D1-derived slopes, not a paper
replication. ACMM CVPR2019 abstract only was verified; PDF403. No full-reading
or reproduction claim. See ../S1_photometric_oracle/literature.md.

Implementation and results are pending. This protocol is declared before D6
candidate code, test outputs, control curves or inferred depths are produced.
