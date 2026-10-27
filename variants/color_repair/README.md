# C1/C2/C3 color repair and scratch retraining

The three repaired arms are independent of the frozen historical coverage runs.
They retain V3 W256_D0, LPIPS0.02, shared frame/physics time, the120/30/50 split,
original optimizer and ordinary densification through15k. No centroid or iso term
is introduced. Historical weights are used only for read-only diagnosis/comparison.

The training-frame diagnostic reads nine predetermined train images per original
model and exports float RGB, raw SH statistics and visible-contributor candidates.
It verifies source checkpoint hashes before/after loading. Contributor negativity
confirms a clamping path, not unique pixel-level causation.

Repair package, independently gated by config:

- `positive_color`: Softplus(beta5) of SH+0.5, evaluated in the exact **deformed**
  center-to-camera directions used by the CUDA renderer. Pass colors_precomp to
  the existing rasterizer; no CUDA modification. Flag-off uses the original path.
- Inverse-initialize random step0 DC coefficients to preserve initial positive
  colors; higher orders remain zero. Original exact-zero colors use a1e-4 floor.
  A resume restores saved coefficients and never applies initialization twice.
- Mask weight0.03, activated after3k with a2k linear ramp, versus old0.1 fromstep1.
- Foreground-balanced RGB L1 weight0.1, same schedule, normalized by3*sum(train alpha).
  Targets are only original120 training RGB/alpha; no held-out targets enter loss.
- C2/C3 local affine weight5e-5, radius0.05,128 anchors/up to8 neighbors. The fit,
  interval10 and affine invariances remain. There are no real object IDs; weaker
  proximity constraints still do not guarantee object isolation.
- C3 samples color moments at training deficit pixels along with coverage votes.
  Require at least3 training times and max channel standard deviation<=0.15.
  Seed child DC from inverse-activated weighted train RGB, reset higher-order SH,
  inherit parent trajectory index/scale/rotation. At most256 children per2k window,
  opacity clamped[0.005,0.05], canonical jitter std0.25*min(parent scale,0.02).
  Keep parents; skip contributor-free pixels; no late pruning/reset. Coverage
  scores, time votes and color moments are checksum-bound to full checkpoints.

All three arms run to total50k from a newly captured random state. Train a common
3k prefix before scheduled objectives; C1 branches there. A common affine-enabled
prefix reaches15k, before coverage sampling. C2 and C3 copy that **same complete
15k checkpoint**, including optimizer/RNG/selection history. This fixes the previous
pre-density divergence confound. CUDA nondeterminism after branching remains.

These are combined repair packages versus historical C1/C2/C3, not isolated
causal estimates for each change. Within the new study, C2/C3 share exactly the
same pre-density state. One scene/seed and historical test exposure are disclosed.

Preflight: unit tests, real CUDA flags-off parity, deformed-view SH parity,
negative-channel finite differences, color-state checksum/resume, flags-off20,
shared20/affine40 -> three100 smoke arms, actual color-aware additions, resume70,
and30-frame validation PNG parity. Smoke has smaller engineering thresholds.
Long training proceeds only after a passed current-source receipt.

Completion selects snapshots on validation only, locks all three before final
test export, and automatically creates official four metrics, alpha diagnostics
and five-snapshot GT/render galleries. A five-second graphical monitor and logs
are under `experiments/color_repair`; new checkpoints under `runs/pidg_color_repair`.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File variants/color_repair/launch.ps1 -Smoke
powershell -NoProfile -ExecutionPolicy Bypass -File variants/color_repair/launch.ps1
```
