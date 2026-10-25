# Shared clock and velocity / trajectory consistency

This optional fork preserves the original PIDG architecture. It uses the same
W256_D0 parent, T3 image losses, seed, optimizer schedule, train/test images and
resolution as the completed T3 refinement. ISO is disabled to isolate the new
time and kinematic changes. No completed baseline is retrained.

## Audit and units

The original renderer uses `fid`, while Cauchy momentum training uses
`fid / len(train_cameras)`. The new `shared_physics_time` flag removes that extra
division. All model branches then use the same normalized scene time `tau`.
Internal hash-grid and Fourier encodings remain their respective encodings of
that shared input; their numerical coordinates need not be identical.

The supplied JSON separately maps the 150 training and 50 test images onto
`[0,1]`. The optional `timeline=frame_index` protocol maps the joint sequence of
verified numeric filenames 0001–0200 onto `(frame_id-1)/199`. This assumes uniform
capture cadence inferred from filenames. It does not recover physical seconds.
Neither JSON files nor image splits are edited. Inference/export applies exactly
the same optional remapping as training and validation.

## Objective

Rendered centers are **rotation plus translation**, not just the warp's last
three channels:

```
Phi(X,tau) = R(q(X,tau)) X + b(X,tau)
L_kin = mean Huber(v_pred(X,tau) - dPhi(X,tau)/dtau; delta=0.1)
L_total = L_T3 + lambda * ramp * interval * L_kin  [on sampled iterations]
```

The derivative is a second-order finite difference with `h=0.001`, centered in
the interior and second-order one-sided at the endpoints. All queries stay in
`[0,1]`. This allows ordinary parameter backpropagation through both trajectory
evaluations without relying on unsupported coordinate second derivatives in the
custom CUDA hash encoder. This is a numerical derivative constraint, rather than
an exact higher-order autodiff implementation.

Each tenth training step uses 256 deterministic particle samples. The interval
factor compensates for applying the loss every ten steps. Long runs use
`lambda=0.001`, start after step 1000, and ramp over 1000 steps. Both velocity and
deformation parameters receive gradients; sampled canonical coordinates are
detached. The stencil disables stochastic feature regularizers and clears the
cached attention gate, then restores the original regularizer flags. It consumes
no additional random samples. A zero weight returns before touching model state.

Velocity units are scene distance per unit normalized time. Acceleration and
stress are still expressed in PIDG's normalized units. The legacy momentum
residual's canonical spatial coordinates, convective contraction and sampled
weighting are retained. Self-consistent velocity is not evidence of correct
physical velocity, material recovery, or forces without ground truth.

## Experiments

| Arm | Time supplied to rendering / velocity | Consistency |
|---|---|---|
| V0 | Historical metadata; physics divides by 150 | Completed T3 reused |
| V1 | Historical metadata; physics shares `fid` | Off |
| V2 | Historical metadata; physics shares `fid` | 0.001 |
| V3 | Common numeric-frame time | Off |
| V4 | Common numeric-frame time | 0.001 |

V1/V2 can be compared with historical T3. V3/V4 must be compared within their
retimed protocol. Their 50 RGB images and poses are unchanged, but supplied test
timestamps differ. Initial retimed weights are copied and hashed from the same
parent; historical metric parity is not expected after changing time inputs.

The initial audit illustrates why protocols must be separated: on the **same
parent weights without any training**, metadata-time test PSNR is 27.25678 dB,
whereas frame-index-time test PSNR is 33.22229 dB. This is a change in timing
inputs, not a learned-constraint improvement. See
`experiments/kinematic_tuning/initial_timing_assessment.json` for all four metrics.
At 100 engineering steps, control and joint runs differ by only about 0.0007 dB;
that does not establish a useful consistency-loss benefit. Long comparisons
retain equal budgets and compare within one timing protocol.

All four new arms target **50k additional iterations**, independently initialized
from the same parent best snapshot with fresh optimizers. A paused arm restores
its full optimizer, model, RNG, camera stack and selection state. Four independent
best snapshots and the final equal-budget 50k snapshot are exported. Metrics are
the official PSNR, SSIM, VGG LPIPS and MS-SSIM on exactly 50 official test images
at 1600x900, quantized equivalently to saved PNGs. Test images are only used in
no-grad evaluation/selection; this follows historical test-selection and is not
an untouched validation protocol. This is one scene/seed, not a multi-seed claim.

## Execution and outputs

```powershell
envs/pidg-cu128/python.exe -m pytest tests/test_kinematic_tuning.py -q
powershell -NoProfile -ExecutionPolicy Bypass -File variants/kinematic_tuning/launch.ps1 -Smoke -Hours 1
powershell -NoProfile -ExecutionPolicy Bypass -File variants/kinematic_tuning/launch.ps1 -Hours 24
```

GPU preflight checks flags-off training, control/joint steps including momentum,
finite nonzero actual grid/head/velocity parameter gradients, save at step 20 and
full continuation to 100, and training/export PNG metric agreement. Smoke scores
do not establish a reconstruction improvement. Long training requires preflight.

`tests/gpu_kinematic_derivative.py` additionally compares the finite-difference
velocity to the real hash encoder's first input derivative on 128 frozen-parent
points. The measured relative RMSE is 0.815% at `h=0.001`, decreasing to 0.397%
at `h=0.0005`. This measures a local approximation error on sampled points; it
does not establish physical velocity accuracy. The flags-off 20-step run matches
the previous same-config evaluated PSNR within 0.000036 dB and matches RNG state;
CUDA atomic gradients do not produce bitwise-identical trained parameters.

- `experiments/kinematic_tuning/results.md`: separately grouped results and
  same-step pairwise metric deltas; final assessment uses exported 50k PNGs.
- `experiments/kinematic_tuning/monitor.html`: 5-second refresh, metric curves.
- `experiments/kinematic_tuning/{plan,preflight,source_freeze,timeline_audit}.json`:
  configuration, provenance, timeline assumptions and engineering checks.
- `runs/pidg_kinematic_tuning/<arm>/rubber_duck`: complete checkpoints,
  per-iteration metrics, consistency RMSE/speed diagnostics, raw renders and GT.
- `runs/pidg_kinematic_tuning/render_comparisons/<arm>/index.html`: 250 GT/render
  pairs per arm, covering four selected snapshots and the final 50k snapshot.
  Full-resolution panels are preserved and verified after PNG saving.

To request a safe pause, create `runs/pidg_kinematic_tuning/stop_queue.request`.
The active trainer saves a complete checkpoint and exits; no GPU suspension is
used. Archive that request explicitly before rerunning the launcher to continue.
The coordinator resumes existing checkpoints and skips completed 50k training.
