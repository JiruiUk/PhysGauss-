# Ordered foreground coverage ablation

This independent PIDG fork follows the requested order. The completed V3
scratch control is reused as C0. Each new candidate receives **50k total steps
from the identical random iteration-zero state**, including empty optimizer
histories, camera order and RNG. No candidate loads trained C0 weights.

| Arm | Foreground mask | Weak local affine | Coverage densification |
|---|---:|---:|---|
| C0_V3_reference | 0 | 0 | off; completed control reused |
| C1_mask | 0.1 | 0 | off |
| C2_mask_affine | 0.1 | 0.0001 | off |
| C3_mask_affine_density | 0.1 | 0.0001 | on |

All arms use the same baseline W256_D0 architecture, V3 image objective and
shared `(original_frame_id-1)/199` deformation/physics time. Iso and kinematic
consistency weights remain zero. The original 50k scratch learning rates,
warm-up3000 and image-gradient densification through15000 are unchanged.
Compare C1-C0, C2-C1 and C3-C2. This cumulative study does not independently
identify interactions or replace a full factorial ablation.

## 1. Training-only original alpha

`data.py` verifies every original training PNG has explicit, nonconstant RGBA
alpha, records file hashes, and reads alpha separately from the existing RGB
compositor. It never substitutes the motion/SAM mask. The RGB loader is
unchanged. Alpha is resized to camera resolution with PIL bicubic, kept on CPU
and copied for the current training view. No held-out alpha enters a loss.

`L_mask = mean_pixels |alpha_render - alpha_GT|`, weighted0.1 from step1. This
includes foreground and background. It supervises silhouette/opacity, not depth.

## 2. Weak affine neighborhood agreement

For128 deterministic anchors, choose up to8 nearest canonical neighbors within
0.1 world units. Fit a full3x3 local map from canonical offsets to deformed
offsets with a detached pseudoinverse. Penalize the residual normalized by mean
squared canonical edge length. Translation cancels. Rotation, stretch and shear
are permitted; there is no fixed neighbor-distance, volume or determinant target.
Weight1e-4 ramps over2000 steps after3000, sampled every10 steps with interval
compensation. Rank-deficient neighborhoods are skipped. Gradients act on the
deformation trajectory with canonical inputs detached.

This experiment explicitly asserts a **single-object Rubber Duck scene** and
refuses silent multi-object use. The pure helper supports explicit object IDs;
tests show no cross-ID edges. The real dataset has no object-ID bank propagated
through Gaussian densification, so contact scenes require that additional work.
Euclidean canonical proximity does not guarantee surface-geodesic adjacency.

## 3. Conservative coverage-guided late additions

Only C3 samples training renders every20 steps from16000 through35999. In GT
foreground eroded by3 pixels, pixels with alpha<0.7 receive deficit
`max(0,0.95-alpha)`. Sample on a4-pixel grid and accumulate onto the renderer's
actual footprint contributors (up to20 IDs per pixel). Eligibility requires
deficits at **three distinct training times**. Repeating one timestamp is one
vote. Rank eligible points by accumulated deficit within the current window.

At18000,20000,...,36000 add at most512 points, with a200000 total point cap.
Children inherit parent SH, scale, rotation, material/velocity ID and dynamic
flag, with small canonical jitter and opacity capped0.1. All parents remain.
A dedicated seeded generator does not consume the trainer RNG. No extra opacity
reset or late pruning occurs; original pruning stops at15000 in every arm.
This tests coverage-driven additions, not a separate pruning-protection method.

Pixels with no rasterizer contributor cannot establish depth or a3D seed and
are skipped. The method reinforces weak existing footprints; it cannot recover
arbitrary unseen geometry. Scores and per-point/per-time votes are saved in a
checksum-bound checkpoint sidecar. Additions run after the optimizer update so
parameter replacement cannot discard that step's gradients.

## Evaluation and verification

Split:120 train /30 validation /50 official test timestamps, pairwise disjoint.
Four best checkpoints use validation RGB metrics only; final50k is retained.
All candidate selections are locked before new final test exports. Historical
official-test exposure remains a limitation.

All five tags export RGB, GT, alpha and pixel-verified GT/render pairs. The
official four full-image metrics are unchanged. Extra diagnostics: foreground
PSNR (GT alpha>0.5), mask IoU (threshold0.5), mean alpha L1, and the fraction
of3-pixel-eroded GT interior with render alpha<0.5. Diagnostics use8-bit PNGs
and per-frame averages. Coverage does not establish3D watertightness.

Unit tests cover exact mask gradients, train-only membership, flags-off early
returns, affine invariance, non-affine gradients, object gating, degeneracy,
unique-time votes and accumulator restoration. GPU preflight exercises flags-off,
all three candidates, actual additions, full density-state resume, validation
PNG parity and alpha export.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File variants/coverage_tuning/launch.ps1 -Smoke -Hours 2
powershell -NoProfile -ExecutionPolicy Bypass -File variants/coverage_tuning/launch.ps1 -Hours 24
```

- Reports/configs/audit/preflight: `experiments/coverage_tuning`
- Iter counters and validation charts, refresh5s: `experiments/coverage_tuning/monitor.html`
- New outputs: `runs/pidg_coverage_tuning`
- GT/render gallery: `runs/pidg_coverage_tuning/render_comparisons`

Create `runs/pidg_coverage_tuning/stop_queue.request` for a full-state safe pause.
Resume after archiving the request; completed arms are skipped.
