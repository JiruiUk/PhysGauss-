# Rubber Duck reconstruction plan — revised 2026-11-03

This is a reconstruction/observability branch. The PIDG baseline and the existing
M0–M10 physics milestone order remain unchanged. Every new branch defaults off.

The original S1–S6 training stages remain the execution plan; their explicit
status is in `staged_reconstruction_plan.md`. R0–R4 below are diagnostic work
items, not replacement stages. Completing R0 did not complete S1 or the overall
task. Negative results require corrective work, without cancelling S2–S6.

## Evidence and decisions

1. Transfer-integrity v1 (matched 2k continuation) improved local objectives but
   did not repair fixed-view fragmentation.
2. Scratch-integrity v2 (fresh common 15k, matched 35k continuations to 50k)
   reduced short-track EPE 1.8039→0.2800 and static anchor distance
   0.47937→0.16306. Test PSNR was 40.8475→40.8205 dB. Fixed-view fragmentation
   and negligible late early-bank support remained. More optimization of the
   same local constraints is not the next experiment.

## Execution ladder

| Step | Experiment | Admission / decision |
|---|---|---|
| R0 | Verified static subset with a time-independent rendering map | Completed; image guardrails passed, structure did not improve; H1 not promoted |
| R0a | Static selection/reference geometry audit | Completed: reference inconsistency already existed at15k; motivated the conditional S1 reconstruction, without certifying contact geometry |
| R1 | Persistent known object support / object-centric canonical representation | First audit coverage and independent identity evidence; unknown stays unknown |
| R2 | Reliable tracks, separated from static geometry | Retain confidence gates; do not infer identity through ambiguous occlusion |
| R3 | Kinematic compatibility, then CMR | Rendered trajectory, velocity and coordinate derivatives must be defined consistently |
| R4 | Known-surface nonpenetration, then fixed low-dimensional contact | Requires accepted object identities and surface geometry; no per-Gaussian free forces |

Rubber Duck remains the main development dataset. Small manufactured tests
supplement F/material/contact verification; they do not replace its benchmark.

## R0 scope

Two matched arms start from the completed scratch experiment's immutable common
15k checkpoint, retain the full optimizer/LR/RNG/camera state, and train to 50k.
No old final50k trained state is used for initialization. Only the R0 main flag
differs. No v1/v2 soft transfer/track/shape losses are enabled.

The accepted 14,624 static rows are a *partial* image-supported hypothesis,
covering about 60.68% of blue-core contributions on average, 47.86% in the middle
third. Other rows remain on the original path; no labels propagate by proximity.
Each accepted row receives a fixed deformation transform and covariance offsets
captured at its pre-existing training-only best-support reference frame. Its
underlying Gaussian parameters and SH appearance remain trainable. This makes
the selected geometry independent of query time without adding particles or
learnable parameters. It does not prove that the captured geometry is correct.
Known-static rows are excluded from the dynamic CMR sampling domain. This
exclusion is part of the declared treatment, not a separate claimed benefit.

## Evaluation and adaptation rules

- Fixed final50k states; validation30 and test50 remain outside gradients.
- Compare R0 arms directly; historical C1/S0/S1 are contextual references only.
- Predeclare native1600×900 rendering, fixed cameras2/86/188 × all200 times,
  moving-camera control, per-region image metrics, and static-subset invariance.
- Check independent held-observation static-track reprojection, where available.
  Structural invariance and low training loss are not independent geometry GT.
- View-completeness measures are descriptive proxies, never true completeness.
- Image guardrail: validation PSNR loss ≤0.2dB and standard VGG LPIPS relative
  worsening ≤5% against the matched control. A visual/geometry benefit must
  additionally be present; meeting guardrails alone is not success.
- Do not promote a model based only on all-scene alpha: inspect static and
  dynamic regions and fixed-view native crops separately.
- After evaluation, append evidence and revise the next step. If static-only
  structure improves but dynamic fragmentation persists, move to R1. If the
  static subset remains poorly supported or image quality degrades materially,
  audit selection/reference geometry before another long run. Do not increase
  weights or launch parameter sweeps automatically.
- Each implementation/result milestone is committed separately. Historical
  source, datasets, checkpoints and reports remain protected.

## Current status

R0: implementation and preflight passed. All 32 related tests, native gradient
and Adam checks, flags-off numerical parity, and full-state recovery passed.
Implementation milestone: `23f53c35`; launch record: `4738e2b9`. Both matched
arms finished50k with35000 actual updates each. Final metrics, native image
review,16 fully decoded videos and independent evidence audit are complete.
H1 is not promoted; the next step is R0a, not a full R1 training run.
Immutable settings remain in `static_pool/plan.json`.

## R0 completed measurements

Final50k measurements and reviewed images: `static_pool/assessment.md`.
Both image guardrails passed: validation PSNR40.455640→40.437735dB and standard
VGG LPIPS0.01360374→0.01418889 (+4.30%). Test PSNR40.846448→40.797551dB.
Fixed86/188 early/middle/late native crops retained the fragmentation pattern,
while the observed camera trajectory remained visually good.

Selected-static validation blue contribution fell51.60%→12.12%; selected median
opacity fell0.999996→0.003908 (68.18% below0.01 in H1). Training-image track
reprojection worsened1.84→7.56px at400px width. This supports opacity suppression
of the fixed subset, without identifying its root cause. Static invariance is
structural; it does not establish geometry recovery. H1 is not promoted.

## R0a admission and audit scope

1. Use training-only observations to map suppressed rows to their reference
   frames, initial and final coordinates/scales/opacity, and visible support.
2. Audit whether independent per-row reference warps form consistent static
   surfaces. Compare initial/final reprojection and cross-time support, including
   occlusion; retain unknown labels where evidence is ambiguous.
3. Check feasibility of one consistent reference geometry or calibrated static
   fit before proposing a short controlled experiment. Do not fix the result by
   threshold changes chosen on validation/test images.
4. Only after this audit, write a separate short-run protocol and admission gate.
   No new full candidate, automatic weight sweep, R1 model, constitutive/contact
   or unknown-force module has been launched from the R0 result.

Rubber Duck remains the main benchmark. Missing simultaneous multi-view/physical
GT limits conclusions; the manufactured F/material tests remain supplementary
and the original M0–M10 order remains unchanged.

## R0a initial reference audit completed

`static_reference_audit/results.json` independently recomputes the same fixed
mapping in the immutable initial15k and final H1 states. The37 conditional
tracks/567 observations already had mean7.79751px reprojection error before R0
training, versus7.56417px afterward (400px width). Initial selected opacity
below0.01 was0%; final68.18%. The14624 rows use120 separate reference frames.
The initial static reference was not established as consistent geometry; this
is not solely a late optimization drift result. The audit is read-only and
does not count as S1 training. Continue S1 reference/mask preparation and then
independent static optimization; S2–S6 remain pending in the restored stage plan.

## S1--S6 actual continuation (supersedes the earlier pending status)

S1 completed12000 updates and conditional static admission. Three sequential
S2 candidates completed16000 each; the first two failed and remain preserved.
The third passes unchanged image gates (dynamic24.25343dB/alpha0.982654,
static35.53011dB) and permits conditional sequential reconstruction corrections.
Its fixed-view fragments and imperfect frame42 ownership remain unresolved; no
physical identity or complete geometry success has been established. See
staged_reconstruction/S2_geometry_recovery/assessment.md.

S3 real-parent recovery exposed a rank-deficient canonical-neighbor case. A new
independent state_recovery entry uses actual current-world same-bank neighbors,
retains the original validity thresholds, and saves derived state. Corrected
40→80 recovery passes136 Adam states and exact camera/Python-RNG replay. Its6000
formal updates, native evaluation and actual fixed-view review are complete;
validation dynamic25.392299dB/static35.941669dB. All57 related tests pass.

S4 binds91 train-only fragments/455observations to79 fixed rows and completes
4000 updates with full evaluation/review. Exact forward one-hot contributions
pass the original visibility-partition gate after preserving an earlier failed
backward-approximation attempt. Validation dynamic25.340218dB/static36.199356dB,
but track projection worsens0.536841→0.785017px@400: retain this negative result.

S5 completes4000 updates with validation dynamic25.603743dB/static35.983147dB
and actual fixed-view review. Track error0.664671px@400 remains worse than S3.
All800 conditional-interior calls have zero active violations; there was no
actual collision correction. Its limited conditional continuation admission
is not admission of the original-plan complete contact scope. S6 real-parent
40→80 recovery passes138 Adam states; its predeclared4000 updates, independent
native val/test, full sampler/state audits and actual image review are complete.
Validation dynamic28.327031dB/static36.169914dB, test dynamic28.563049dB;
455 qualified track observations average0.427107px@400 (20.44% below S3).
All800 conditional-interior calls are inactive. Static geometry is exactly S1;
only its appearance was unfrozen. Three final videos decode200frames each.
Observed-view contours improve, but novel-view sheets/holes and long late bank
streaks remain. Physical IDs after ambiguity remain unknown.

S5's manufactured300-step optimizer validation is separate from real-scene
training. Rubber Duck has no certified contact surface or acquisition seconds;
contact force/friction/restitution remain off. S6 jointly refines the admitted
image/kinematic/track/conditional-interior objectives and static appearance at
0.25 learning rates; static geometry remains S1. Full operational continuation
is in staged_reconstruction/runbook.md. These are conditional reconstruction
experiments, not material/contact identifiability claims.

## Measured completion and unresolved original scope

The revised conditional S1--S6 training lineage completed46000 formal updates;
two failed S2 candidates add32000 preserved updates. Standard native1600x900
quality is measured for every admitted S2--S6 final state. Final S6 validation
full PSNR30.58520dB, SSIM0.975011, VGG-LPIPS0.041803; test LPIPS0.041247.
See staged_reconstruction/final_summary.md and output/staged_reconstruction/index.html.
These are sequential image/qualified-track results, not causal ablations or
matched legacy-camera PIDG comparisons.

No further training is automatically queued. The original complete geometry,
persistent physical identity and contact goals remain unresolved; S5 implemented
only a conditional static-interior prior that never activated in the real scene.
Object-object nonpenetration, contact forces, friction and restitution remain
unvalidated/off, rather than silently removed from the original objective.

Next evidence priority is geometry/identity observability, supported by a small
calibrated RGB synthetic supplement with metric geometry, acquisition seconds,
material-point trajectories and persistent IDs. Keep asynchronous single-view
RGB as training input and synchronized novel views/3D truth as evaluation only.
First isolate known motion without collision; then admit genuine collision
tests only with certified surfaces/IDs/time and independently exported contact
truth. Manufactured sphere/plane optimization is not a replacement for this
RGB reconstruction benchmark. Freeze split/metrics/criteria before running;
do not increase weights, sweep candidates or add constitutive/unknown forces.
The concrete retained dependencies are in staged_reconstruction_plan.md.


## S3-U: user-authorized unified dynamic population (2026-11-03)

The user selected one unified dynamic Gaussian population without estimating
body number. This is a separate branch, not unknown-count instance discovery.
No instance labels, original four-group initializations, identity ledgers or
trained S2/S3 dynamic weights are inputs. Frozen S1 static support is shared.
One shared canonical population, deformation network and velocity/stress field
replace separate instance fields. The global15,507 point budget is unchanged;
capacity and initialization differences are disclosed rather than treated as
a matched single-switch ablation.

Frozen protocol: staged_reconstruction/S3_union/protocol.json. Execute16k
image-only warmup, actual native review and unchanged numeric gates, then6k
S3 kinematic/CMR continuation. Existing weights, time units, camera convention
and static freeze remain. Validate each phase's40-to80 full-state recovery
before formal training. Dynamic-only spatial stencils have no physical-boundary
exclusion. No contact/constitutive/unknown-force/instance-count inference is on.
Compare native val/test and fixed86/fixed188/original-camera videos against
original S3, preserve negative results and late identity uncertainty.

Implementation:49 related tests, native renderer parity(max3.5763e-7) and
warmup40-to80 Adam/RNG/sampler/static-freeze verification passed. Formal warmup
is ready; no historical paused trainer or automation is resumed. Test setup
failures involving the shared temp directory and a missing new parent directory
are retained in the test receipt; the isolated fresh test directory passes.
