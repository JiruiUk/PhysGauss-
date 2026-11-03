# One S2 recovery candidate

The frozen RGB-only16k attempt fails the predeclared S2 validation admission:
native dynamic PSNR9.351dB, dynamic-only alpha0.4945, and static PSNR33.894dB
(S1 validation35.644). Its code, optimizer state, checkpoints and metrics are
preserved. Do not advance to S3 on this model or lower its admission thresholds.

This independent candidate changes two explicitly recorded assumptions:

- Per-object depth initialization uses a front-offset local static depth prior,
  with continuous slab jitter. The same15,507 dynamic points and architecture
  remain. The inferred depths are not measured depth or physical surfaces.
- Original RGB loss gets visible-yellow RGB weighting and one-sided image
  coverage. Qualified early identities constrain their own observed component;
  unknown groups and unassigned components form a permutation-invariant union.
  Nearest image distances are geometric supervision, not physical-ID assignment.
  Hidden splats are not forced into the observed silhouette. RGB still penalizes
  excess visible splats. CMR, sparse tracks and contact remain off.

`train.py` explicitly binds the new scene/objective/run and report paths into the
unchanged S2 driver in this entry point only. It never rewrites that driver or
loads an earlier attempt's optimizer into a different objective. Original S2
numeric evaluation is reused; separate review exports recovery images so prior
exports cannot be overwritten. Recovery sources/protocol/initialization and
inherited S1/S2 hashes are verified before formal training.

Preparation, native finite gradients, exact static freeze,31 related tests and
separate-process40→80 resume passed. Formal16k final-state training is required;
short smoke is not successful reconstruction. This is a combined repair, not a
single-factor causal ablation. No candidate sweep or loss-weight search is run.

```powershell
. .\reproduction\scripts\enable_cuda.ps1
envs/pidg-cu128/python.exe -m variants.staged_reconstruction.recovery.train --enable-recovery --enable-object-stage
envs/pidg-cu128/python.exe experiments/staged_reconstruction/review_objects.py --checkpoint-dir S2_recovery/S2_objects --recovery
```

Recovery state/checkpoints: `runs/pidg_staged_reconstruction/S2_recovery/`.
Protocol and reports: `experiments/staged_reconstruction/S2_recovery/`.
The full S1–S6 task remains active during this repair.
