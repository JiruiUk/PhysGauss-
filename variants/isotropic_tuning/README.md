# Optional isotropic normalization for T3

This independent fork adds a training-only Gaussian shape regularizer to the
completed T3 recipe. The original PIDG source, completed metric-tuning fork,
checkpoints and results remain immutable. The network architecture is unchanged.

For Gaussian i, use its actual rendered axis magnitudes
`a_i = abs(exp(log_s_i) + d_scaling_i(t))`, and penalize

`L_iso = mean_i(mean_k(abs(a_ik / mean_j(a_ij) - 1)))`.

The objective is `L_T3 + isotropic_weight * L_iso`. Normalization makes the term
dimensionless and invariant to uniform rescaling of nondegenerate axes. The
implementation first divides by the largest magnitude to avoid overflow; a
dimensionless epsilon protects the mean of zero axes. Equal nonzero axes and
the explicitly handled all-zero degenerate case have zero penalty. This is a
soft shape penalty; it does not force spherical Gaussians, preserve volume,
enforce positive deformed scales or impose an isotropic material law.

PIDG's renderer adds `d_scaling` to activated canonical scales in linear space,
not log space. Signed scales enter a squared covariance, hence the absolute
magnitudes in this term. `isotropic_scope=rendered` follows that actual path.
When `compute_cov3D_python=True`, PIDG uses canonical covariance and ignores
deformation offsets; the penalty also uses canonical axes in that case.
`isotropic_scope=canonical` is an explicit alternative. All Gaussians receive
the penalty at the current sampled training time; there is no test-image loss.

Flags in `metric_config.json`:

- `isotropic_weight`: default **0**, term disabled without accessing Gaussian
  tensors or adding an operation to the loss graph.
- `isotropic_scope`: `rendered` (default) or `canonical`.
- `isotropic_eps`: positive dimensionless denominator floor, default `1e-8`.
- The inherited `enabled=false` disables both tuning and isotropy.

Four runnable 50k refinement configurations share the original W256_D0 parent,
T3's L1/SSIM/LPIPS losses, learning rates, optimizer initialization, seed0,
disabled densification and original physics losses. Only isotropy weight changes:

| Config | Weight |
|---|---:|
| I0_T3_control | 0 |
| I1_iso_1e4 | 0.0001 |
| I2_iso_1e3 | 0.001 |
| I3_iso_1e2 | 0.01 |

Configs are exported to `experiments/isotropic_tuning/configs/`. They start from
the original baseline, allowing a matched T3-only control; they do not overwrite
or restart the completed T3 experiment. A fresh refinement uses a new optimizer;
an existing refinement resumes its complete model/optimizer/RNG/camera state.
Evaluation remains the fixed 50 official test images at 1600x900. Test-based
checkpoint selection follows the existing protocol and does not establish
untouched-test generalization.

Run engineering validation:

```powershell
./variants/isotropic_tuning/start.ps1 -Smoke -Hours 2
```

It runs an original-flags-off 20-step smoke test, a 100-step zero-weight T3
control, and an enabled 20-to-100-step full-checkpoint resume. Step100 also
executes the original Cauchy momentum loss. Loss and gradient audits go to
`isotropic_loss.jsonl` and `isotropic_gradient.jsonl`. These engineering scores
are not a long-training effectiveness comparison.

Run one full experiment (separately, without overlapping GPU jobs):

```powershell
./variants/isotropic_tuning/start.ps1 -Train I0_T3_control -Hours 2
./variants/isotropic_tuning/start.ps1 -Train I2_iso_1e3 -Hours 2
```

Each invocation runs only the selected experiment; the second command must wait
for the first to exit. Training reserves 20 minutes for four snapshot renders
and original PNG-metric verification. If its training budget is reached, a full
checkpoint is saved and the run is marked stopped, not completed at50k. Launch
the same command again to resume. For manual pause, create
`runs/pidg_isotropic_tuning/stop_queue.request`, wait for saved process exit, then
archive that request before relaunching. Do not suspend the GPU process.

New outputs: `runs/pidg_isotropic_tuning/<id>/rubber_duck/`.
New status: `experiments/isotropic_tuning/{monitor.html,results.md,preflight.json}`.
Both monitors and results are separate from the completed metric-tuning study.
The result table also retains the already-rendered T3 reference snapshots, so
a new run cannot be labelled globally best merely by beating the older baseline.

## Live curve monitor

Start the read-only chart sampler with `./scripts/start_isotropic_monitor.ps1`.
Open `http://127.0.0.1:8766/` or `experiments/isotropic_tuning/charts.html`.
It shows PSNR, SSIM, LPIPS, MS-SSIM, logged training EMA loss and normalized
isotropy loss, plus progress and GPU usage. The four test metrics include the
historical T3 curve at the same iteration and the original W256_D0 reference.
Hover reveals sampled values; controls switch axis range, reference visibility
and page refresh. The page samples every five seconds and never sends training
control commands. It warns when its source snapshot is older than20 seconds.

The chart page uses its own `charts_data.js`/`charts_data.json` outputs so the
existing trainer's table publisher cannot overwrite it. It reads complete log
records incrementally, deduplicates resumed iterations and retains loss extrema
when downsampling. No model, optimizer, runtime source freeze or training worker
is modified. Both local-file and HTTP viewing use bundled assets without a CDN.
