# Rubber Duck reconstruction plan — revised 2026-11-03

This is a reconstruction/observability branch. The PIDG baseline and the existing
M0–M10 physics milestone order remain unchanged. Every new branch defaults off.

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
| R0 | Verified static subset with a time-independent rendering map | Current step; unit tests, native GPU gradient/parity/resume checks precede full training |
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
No formal R0 training result exists yet. The matched arms begin after the
implementation milestone commit. Immutable settings are in `static_pool/plan.json`;
subsequent observations and plan changes belong in `static_pool/progress.md`
and assessment. The active thread follow-up checks completion every 10 minutes.
