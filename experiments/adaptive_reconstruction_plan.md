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
| R0a | Static selection/reference geometry audit | Required after R0: inspect opacity suppression, reference consistency and multi-time support before another long run |
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
