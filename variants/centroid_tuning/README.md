# Training-image foreground centroid ablation

This is an independent fork of the audited PIDG runtime. The only copied trainer
change is its optional `Tuning` import; historical forks and weights stay frozen.

The user requested an overall center constraint on the original model. Physical
masses and per-object labels are unavailable. This experiment therefore uses the
**2D whole-scene foreground alpha centroid**, not a physical 3D center of mass.
It does not add a learnable center head, segmentation, or freely learned masses.

For image width W and height H, coordinates are x/(W-1), y/(H-1). Predicted and GT
centroids are the alpha-weighted means of these coordinates. The denominator is
the mean alpha clamped to 1e-6. Loss is the squared Euclidean distance between
the predicted centroid and a detached GT centroid, weight 1. Only the original
120 training RGBA alphas define targets. No silhouette L1 or area/opacity penalty
is added. Uniform alpha scaling leaves the objective unchanged away from the floor.

Default base: original PIDG RGB objective and legacy physics-time convention on
the established clean 120 train /30 validation /50 test protocol. Rendering uses
the existing dataset metadata clock. No LPIPS training term, isotropic prior,
velocity consistency, local affine term, or extra coverage densification is used.
`--base v3` is a separately declared alternative; it cannot replace an existing
experiment plan. Original conventional densification remains through 15k.

Both arms start from a newly captured full random state. Run a common 3k prefix,
then copy its full tensors, optimizer, RNG, camera stack, and selection history.
Change only centroid_weight from 0 to 1 for G1. Both arms end at total50k, not
an additional50k; G1 ramps from zero to full weight over 3k-5k. This avoids
pre-activation divergence in the earlier ablations. CUDA divergence after the
branch remains a limitation. Historical final-only controls cannot provide this
paired full3k state, so a new matched control is needed.

Preflight: CPU analytic/gradient/membership tests; actual CUDA alpha gradient
finite differences; flags-off20; shared20 -> two100 smoke branches; validation
PNG metric parity; full-state branch/resume. Long runs start only after passing.
Checkpoints are selected using validation only, both choices are locked before
final test exports. Existing test-set exposure is disclosed.

Completion automatically exports all five snapshot tags, official four metrics,
alpha coverage diagnostics, centroid distances and GT/render pairs. Reports and
five-second charts: `experiments/centroid_tuning`. Runs: `runs/pidg_centroid_tuning`.

A global centroid can hide compensating errors across ducks/obstacles and cannot
guarantee correct depth, watertightness, local coverage, or physical mass recovery.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File variants/centroid_tuning/launch.ps1 -Smoke
powershell -NoProfile -ExecutionPolicy Bypass -File variants/centroid_tuning/launch.ps1
```
