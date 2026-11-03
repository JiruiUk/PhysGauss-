# S1-D7: one cross-view local-plane transport candidate

D6's target-local D1 slope gate excludes about98.8% of original gross errors.
It improves small errors but fails both numerical/geometry gates. Keep the
result and all thresholds; do not repeat D6 or relax its0.02 residual check.

The new mechanism supplies a plane hypothesis from OTHER fitting views. Given
the current D6 reference point, project it into exactly the original first4
fitting sources, query each nearest source pixel, and use only an original D1
local plane already admitted under the unchanged D6 fit/held/center and plane
validity checks. Anchors use saved D6 coefficients plus original D1 source
depth, never updated source depth with an unchecked intercept. Intersect the
source plane with the target ray and express its slope in the reference camera.
Require the hit's source projection inside the original source5x5 footprint.
Neither reference nor source has a trusted static label or physical identity.

The target can receive a proposal even if its OWN D1 neighborhood was not
admitted. This changes the source of hypotheses, not the source anchor check.
At most4 geometric hypotheses come from4 fitting source images, unrelated to
object counts. No random normal bank, normal optimization,192-level re-sweep,
new source selection, loss, weight/model switch or point addition.

Current parent is the fixed D6 updated depth at every original candidate.
For incumbent warped cost use the target's original D1 slope if finite and
plane-valid at that depth, otherwise a declared frontoparallel zero slope.
This defines a comparison cost, not a reliable-normal assertion. For each
transported hypothesis use its own plane slope/depth and the same raw RGB
support matcher. Select by best3 of4 fit, then apply unchanged two-held
nonworse/one-strict-improvement equation and1e-6 tolerance. Ambiguous/invalid
or absent source hypotheses leave the parent depth unchanged.

Wrong source planes, footprint overlap, repeated texture and moving occluders
remain failure modes. No source plane is called GT; no unknown is treated as
dynamic/background. Independent static/visible-thin evaluation and unchanged
strict gates determine whether the candidate helps. All parent candidates and
unchanged depths remain saved; source-plane metadata has no material meaning.

Before full inference: meaningful geometry/parallel/behind-camera/footprint/
gradient/flags-off/acceptance tests, train-only/source freeze, independent-process
complete4 versus2+2 recovery for both conditions. Then one serialized96-view
run per diffuse/glossy case with original six RGB sources; full membership and
acceptance replay plus fixed0/24/72 raw RGB replay, independent original GT
evaluation and actual image review. No tuning/sweeps or GT-selected training
points. Keep the original D4 gate objects exactly, with D6 as numeric parent.

Literature: read PatchMatch Stereo BMVC2011 sections1/2.1/2.2 motivate local
plane representations and aligned support. This specific cross-view transport
proposal is our adaptation, not a claim of reading/replicating the full paper's
propagation method. ACMM CVPR2019 abstract verified, PDF403; no replication
claim. See ../S1_photometric_oracle/literature.md.

No Gaussian training is authorized by this stage. Only after strict evidence
gates pass may a separate matched Gaussian protocol be declared with native
gradients/flags-off/full Adam-RNG-camera recovery and two-condition structural
GT checks. Preserve PIDG, frozen sources/weights, fixed Small and S3-U; do not
restore paused branches or old real18k→24k. Code/results are not yet produced.
