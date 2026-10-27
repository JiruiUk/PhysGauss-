# Independent component ablations

This follow-up adds the two requested standalone components to the V3 scratch
reference. Existing cumulative runs and all their source/configuration files
remain frozen.

| Arm | Mask loss | Spatial affine loss | Coverage-guided additions |
|---|---:|---:|---|
| C0: completed V3 reference | 0 | 0 | off |
| C1: previous mask-only | 0.1 | 0 | off |
| C4_affine_only: new | 0 | 0.0001 | off |
| C5_density_only: new | 0 | 0 | on |

The independent effects are C1-C0, C4-C0 and C5-C0. Earlier C2-C1 and C3-C2
measure conditional effects with preceding components active. These are different
comparisons. A complete factorial interaction study would need further arms.

Both new models have the same W256_D0 architecture, V3 RGB objective, shared
frame clock, seed0 and original scratch optimization schedule. Iso and velocity
consistency remain off. They copy the identical random full iteration-zero
checkpoint used by C0-C3, including empty optimizer states, camera order and RNG.
Each receives50k total steps; neither uses any trained parent's weights.

## Component meaning

C4 uses exactly the weak local affine formula, neighbor radius0.1,128 anchors,
up to8 neighbors, interval10, start3000 and ramp2000 from the cumulative runs.
It loads no separate alpha supervision. Baseline RGBA-to-RGB compositing is
unchanged. A formula/gradient parity test compares the extracted no-mask path
against the original implementation.

The real scene contains multiple ducks and obstacles. **C4 is an unsegmented
spatial-neighbor ablation, not an object-isolated constraint.** Its configuration
explicitly acknowledges that limitation; the disproved single-object declaration
is removed from this new runtime. No object-ID filtering or learned segmentation
is introduced. Cross-object adjacency remains possible, as with the original C2.

C5 uses training-frame alpha only to decide where to add Gaussians. Its mask and
affine loss coefficients are both zero. Thus it is density-only, not mask-free
input data. All thresholds and schedules match C3: samples every20 steps from
16000, three distinct training-time votes, at most512 additions every2000 steps
at18000..36000, total point cap200000. Parents remain; no added late pruning or
opacity reset. No-contributor pixels cannot establish3D seeds. Density window
state uses the inherited checksum-bound checkpoint sidecar.

## Queue and evaluation

The coordinator first waits for the existing C0-C3 queue to finish training and
its render/metric jobs and release its process. It never runs a second GPU
training/render process against that queue. The existing24-hour deadline is
retained rather than silently extending the budget.

After that dependency completes, it automatically runs flags-off20-step and
both component100-step GPU smoke tests. It checks nonzero affine trajectory
gradients, real density additions, full40-step density resume, absence of mask
loss logs, validation PNG parity and alpha export. Only passing preflight allows
the two50k long runs. Unit/configuration tests run before enqueueing.

The same time-disjoint120 train /30 validation /50 test split is used. Validation
alone selects four best checkpoints. Both new selections are locked before test
exports; final50k is kept separately. Reports compare all six arms on official
full-image PSNR/SSIM/LPIPS/MS-SSIM and foreground PSNR, mask IoU, interior low-alpha
rate. All five tags export pixel-verified GT/render pairs.

The official test set has historical and user-authorized early inspection
exposure. This extension is not a fresh untouched test. Single scene/seed and
CUDA nondeterminism remain limitations. No improvement is presumed.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File variants/coverage_isolated/launch.ps1
```

- New source: `variants/coverage_isolated`
- New outputs: `runs/pidg_coverage_isolated`
- Six-arm comparison: `experiments/coverage_isolated/results.md`
- Current iter and5-second charts: `experiments/coverage_isolated/monitor.html`
- GT/render galleries: `runs/pidg_coverage_isolated/render_comparisons`

Create `runs/pidg_coverage_isolated/stop_queue.request` to safely stop this
follow-up queue. This does not pause the original C0-C3 experiment.
