# S2 bounded-scale geometry recovery

The preceding16k recovery restores dynamic-image coverage (native validation
PSNR25.086dB, dynamic alpha0.98499) but still fails static preservation:
31.335dB versus S1's35.644dB. Its code, weights and complete result remain fixed.
The5k read-only audit finds all scales of groups0/1 clamped to1e-4 at42/86,
while other groups reach the0.15 upper limit. Large colored splats can explain
observed yellow pixels while object groups disappear or pollute the static view.

This sequential correction uses the same initial points/depth priors/networks
and16k budget. A separate flag interprets the scale head with a positive bounded
sigmoid scale policy in[0.15,2.5] initial spacings, preserving the initial0.65
spacing exactly. It changes neither the original PIDG class nor earlier entry
points. The static RGB term gets the same0.5 normalization as foreground RGB;
conditional blue appearance is supervision, not a GT surface label. Physical
identities remain unknown after their observed connection is lost.

Three scale tests,40 related tests, native gradients/static freeze and independent
40→80 recovery pass. These are engineering checks, not a formal image/geometry
admission. Native image, contribution and fixed-view review retain the same
admission thresholds. This is one diagnosis-driven correction, not a parameter
sweep or a single-factor causal ablation.

```powershell
. .\reproduction\scripts\enable_cuda.ps1
envs/pidg-cu128/python.exe -m variants.staged_reconstruction.geometry_recovery.train --enable-geometry-recovery --enable-object-stage
envs/pidg-cu128/python.exe experiments/staged_reconstruction/review_objects.py --geometry-recovery --checkpoint-dir S2_geometry_recovery/S2_objects
```

State/protocol/reports are isolated under `S2_geometry_recovery/`. The full
S1–S6 task stays active; S3 requires actual reviewed S2 reconstruction admission.
