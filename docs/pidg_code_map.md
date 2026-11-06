# PIDG upgrade code map — Phase 0

Update for Phase 1–15 (2026-10-15): frozen runtime source is selected by each
variant's `stage.json::code_dir`, including preserved `code_rN` revisions. New
variants use their local format-2 `training_checkpoint.py` with mechanics/spline
state; the original Phase 0/shared format-1 helper is unchanged. The implementation
and long-training protocol are documented in `docs/physics_design.md` and
`variants/upgrade/README.md`.

Audited on 2026-10-15. The upgrade target is the patched official implementation in
`third_party/PIDG`, not the independent CPU reference implementation in
`src/physgauss`. The original five-scene benchmark is `runs/pidg_official`.
The upstream commit and compatibility differences are recorded in
`reproduction/source-lock.json` and `reproduction/README.md`.

## Actual modules

| Plan concept | Actual official implementation |
|---|---|
| Canonical Gaussians, SH appearance, opacity, scale, rotation, particle IDs, densification | `scene_PIDG/gaussian_model.py::GaussianModel` |
| Scene and camera splits | `scene_PIDG/__init__.py::Scene`, `scene_PIDG/dataset_readers.py::readCamerasFromTransforms` |
| Continuous deformation wrapper | `scene_PIDG/deform_model.py::DeformModel.step` |
| Hash representation and deformation heads | `scene_PIDG/network.py::Grid4D`, `DeformNetwork` |
| Hash CUDA autograd implementation | `hashencoder/hashgrid.py`, `hashencoder/src/` |
| Material velocity and six stress components | `motion_utils/pidg.py::VelocityNet`, accessed through `GaussianModel.get_velocity_and_stress` |
| Differentiable RGB/flow rasterizer | `gaussian_renderer/__init__.py::render`, installed `diff_gaussian_rasterization` extension |
| Flow supervision and masks | `utils/flow_utils.py::motion_flow_velocity` |
| Cauchy momentum residual (CMR) | `motion_utils/pidg.py::cauchy_momentum_constraint`, `compute_acceleration`, `compute_stress_divergence` |
| CMR particle sampling and chunk reduction | `train_pidg.py::cauchy_momentum_loss_sampled` |
| Training and best snapshot selection | `train_pidg.py::training`, `training_report` |
| Defaults / scene configuration / merging | `arguments/__init__.py`, `arguments/pidg/*.py`, `merge_config` |
| Inference snapshots | `GaussianModel.save_ply`, `Scene.save`, `DeformModel.save_weights` |
| Full optimizer/RNG checkpoints | `reproduction/scripts/training_checkpoint.py` (format 1) |
| Reproduction launcher and seed control | `reproduction/scripts/run_baseline.py`, `entrypoint.py` |
| Official test-image evaluator | `render.py`, `metrics.py`: PSNR, SSIM, VGG LPIPS, MS-SSIM |

Paths in the table are relative to `third_party/PIDG`, except explicitly qualified
reproduction paths. Upgrade variants carry their own runtime source under
`variants/<phase>/code`; data, CUDA environment, installed rasterizer and the
full-checkpoint helper are shared and fingerprinted.

## Rendering call chain

```text
published transforms_train/test.json frame['time']
  -> CameraInfo.fid -> Camera.fid (float tensor)
  -> training: fid expanded to [N,1], canonical xyz.detach()
  -> DeformModel.step: scaled xyz and affine-mapped continuous time
  -> Grid4D: canonical xyz + xyt/yzt/xzt hash features
  -> DeformNetwork: quaternion+translation [N,7], rotation/scale offsets
  -> render: x = R(q) X + translation [N,3]
  -> CUDA GaussianRasterizer -> RGB
  -> (1-lambda_dssim)*L1 + lambda_dssim*(1-SSIM) + TV
```

`d_xyz` is **not** a position or displacement vector: its first four values are a
quaternion, and its last three are a translation. Later kinematic derivatives must
differentiate the complete rendered position `R(q(X,t)) X + b(X,t)`.
The canonical point is not guaranteed to be an exact identity deformation at t=0.
Scale and rotation offsets are separate renderer inputs.

## Material and physics call chain

```text
every 100 iterations after warm_up
  -> coords = concat(canonical Gaussian xyz, fid / number_of_train_cameras)
  -> random particle subset and chunks (with persistent particle IDs)
  -> VelocityNet(coords, particle_ids) -> v [N,3], stress parameters [N,6]
  -> time derivative of v, spatial gradient of v
  -> symmetric free stress + force_sigma * deviatoric symmetric velocity gradient
  -> stress divergence
  -> rho * (dv/dt + legacy convective contraction) - div(stress)
  -> mean squared residual, chunk average, multiplied by M/N
```

The legacy convective contraction is `einsum('bi,bij->bj', v, grad_v)` with
`grad_v[i,j] = d(v_i)/d(x_j)`; retain this behavior in A0 and audit it separately
before introducing a corrected mode. Density is fixed to 1.0; no body forces are
included. CMR coordinates use canonical positions, not rendered deformed positions.
The stress correction depends on velocity gradients; it is not closure from F.

For flow, the same iteration renders the current and next deformation, then a
velocity-advanced translation `b(t)+v_net*delta_fid`. `v_net` is evaluated at `fid`,
whereas CMR evaluates at `fid/train_count`. The flow loss averages the Gaussian
motion-flow and velocity-flow losses after camera-motion compensation and the
published motion mask. This does not enforce `v = dPhi/dt`.

## Known baseline discrepancies to preserve and measure

- PIDG JSON already contains normalized continuous times. CMR divides by train
  count again; deformation and flow do not. Phase 1 must isolate this change.
- `next_cam` means the next training frame; published flows are between adjacent
  raw frames. Split gaps are recorded in the existing dataset audits.
- CMR sampling multiplies the sample mean by M/N; the implementation comment
  claims the opposite scaling.
- The missing upstream material module was replaced with the released
  `motion_utils.pidg.VelocityNet` as a disclosed compatibility fix.
- Official best snapshot selection uses test PSNR; retain this protocol for a
  comparable benchmark and disclose it in research reporting.
- Hash input first derivatives have a GPU preflight, but trajectory second
  derivatives and mixed higher derivatives need explicit Phase 2/3 tests.

## Checkpoints, seeds, and comparison protocol

Full checkpoints contain Gaussian parameters/buffers, material state, Grid4D and
deformation state, both Adam optimizers, learning rates, iteration, best PSNR,
camera order, Python/NumPy/Torch/CUDA RNG states, and the saved options. A `.json`
sidecar records SHA256 and size. PLY/material/deformation snapshots alone cannot
resume training. The terminal iteration of the legacy trainer skips optimizer
step; this behavior remains unchanged in A0.

Seeds are set before model/data creation. Keep the released seed-0 sampling and
CUDA execution mode; do not claim bitwise determinism from atomic CUDA kernels.

The formal baseline has five completed scenes at 50,000 iterations, width 1600,
seed 0, official splits, best snapshot and official evaluator. Short sanity uses
120 iterations, width 400, warm_up=0, no densification, evaluation at step 120.
Compare short variants only against matching short controls; list 50k metrics as
reference without calculating cross-budget gains. Never overwrite baseline runs.
Freeze code, scene config, dataset split/initialization and evaluator hashes, and
retain commands and protocol in each new run. Reject mismatched comparisons.

Phase 0 introduces no physics equation or training loss. Missing future physical
diagnostics (F, J, acceleration, mass/volume) are unavailable, not fabricated zeros.
