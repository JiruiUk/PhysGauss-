# PIDG repository map — M0

The execution pack in `pidg_codex_plan.zip` is the current plan. The previous Phase
0–15 experiments were removed at the user's request and are not resumed.

The baseline is the released PIDG code at commit
`b77fd2107a50c875d7219bb4f82388d5e5ea1319`, under `third_party/PIDG`, with the
existing, disclosed Windows/release compatibility patches. It is not the separate
CPU research scaffold in `src/physgauss`. No new physics is added to the original tree.

| Concept | Actual file / symbol (relative to `third_party/PIDG`) |
|---|---|
| Training entry | `train_pidg.py::training`, CLI at file end |
| Configuration | `arguments/__init__.py::{ModelParams,OptimizationParams,PipelineParams,merge_config}`, `arguments/{pidg,dnerf,hypernerf,neu3d}/*.py` |
| Canonical Gaussian model | `scene_PIDG/gaussian_model.py::GaussianModel` |
| Deformation field | `scene_PIDG/deform_model.py::DeformModel.step`; `scene_PIDG/network.py::{Grid4D,DeformNetwork}` |
| Velocity field | `motion_utils/pidg.py::VelocityNet.velocity_head`, `GaussianModel.get_velocity_and_stress` |
| Predicted stress/material field | `VelocityNet.stress_head`; `GaussianModel.force_sigma`; `compute_stress_divergence` adds a velocity-gradient stress correction |
| Lagrangian trajectory / flow | `utils/flow_utils.py::motion_flow_velocity`, called in `train_pidg.py` every 100 iterations after warmup when `use_flow` is true |
| Cauchy momentum residual | `motion_utils/pidg.py::{cauchy_momentum_constraint,compute_acceleration,compute_stress_divergence}`; `train_pidg.py::cauchy_momentum_loss_sampled` |
| Renderer | `gaussian_renderer/__init__.py::render`, installed flow-capable CUDA rasterizer |
| Dataset loaders | `scene_PIDG/__init__.py::Scene`, `dataset_readers.py::sceneLoadTypeCallbacks`; `cameras.py::Camera` |
| Evaluation | `render.py::{render_sets,render_set}` then `metrics.py::evaluate` |
| Inference checkpoint | `Scene.save`, `GaussianModel.save_ply/load_ply`, `DeformModel.save_weights/load_weights` |
| Full training checkpoint | `reproduction/scripts/training_checkpoint.py::{save_training_state,restore_training_state}`, called by patched original trainer |

## Verified datasets and protocol

The local five PIDG scene directories are `rubber_duck`, `balls_reaction`,
`mechanics_cloth`, `motion_kuro`, `dry_ice`. The released Rubber Duck **config** is
spelled `arguments/pidg/rubber_dack.py`; the dataset remains `rubber_duck`.
D-NeRF has 8 config files, HyperNeRF 11, and Neu3D 6. Only PIDG and the independent
toy synthetic data are currently installed. A config's existence does not imply
its dataset, material annotations, future trajectory GT, or force GT is available.
The experiment matrix records the exact config names and local availability.

Five completed original baselines exist in `runs/pidg_official/<scene>`: 50,000
iterations, width 1600, seed 0, official train/test splits, original best-test-PSNR
snapshot selection and official PSNR/SSIM/VGG-LPIPS/MS-SSIM evaluation.
On the user's instruction, M0 reuses these runs instead of training again.

## Important implementation conventions

- `d_xyz` contains quaternion + translation, **not** a displacement. Rendered
  centers are `R(q) X + b`; covariance rotation/scale offsets are separate.
- Six stress components are ordered `[xx, yy, zz, xy, xz, yz]`. The full Cauchy
  stress includes the original `force_sigma * dev(sym(grad(v)))` correction.
- CMR uses canonical positions and `fid / train_count`; rendering uses `fid`.
  The legacy convective contraction and sampled `M/N` weighting are retained in A0.
- Rubber Duck's official config has `use_flow=False`. Cloth is for reconstruction
  only in the new solid Neo-Hookean extension; Dry Ice and Motion Kuro are also
  ineligible for that solid loss. These settings must not be silently changed.
- The original release imports a missing material file. Existing compatibility
  patches use the released `motion_utils.pidg.VelocityNet`; identity with the
  missing implementation cannot be proven. Other existing patches support MSVC,
  camera device transfer, released mask paths, material snapshot loading and full
  checkpoints. See `reproduction/README.md` and the captured baseline patch.
- Historical training did not capture runtime file hashes. M0 validates completed
  outputs and captures current source/data hashes; it does not retroactively prove
  historical source identity or claim paper-table reproduction.

Material/force recovery and future-prediction errors remain unavailable without
appropriate ground truth. Synthetic tests must be labeled separately from official
scene measurements.

## Metric tuning (separate fork)

`variants/metric_tuning/code` copies the audited original runtime. Its optional
hooks live in `variants/metric_tuning/{model,runtime,checkpoint}.py`; preparation
is recorded in `source_manifest.json`. The original trainer, deformation network,
renderer and data loaders remain at the locations above. The fork adds a
disabled-by-default residual adapter to `DeformNetwork`, optional training-image
objectives, and four-metric no-grad checkpoint selection. New runs/metrics stay in
`runs/pidg_metric_tuning` and `experiments/metric_tuning`. No baseline files or
previous experiment weights are edited.

## Isotropic Gaussian shape regularization (separate fork)

`variants/isotropic_tuning/model.py` defines the dimensionless principal-axis
isotropy penalty. `runtime.py` subclasses the frozen metric-tuning runtime;
`prepare.py` copies its code into a new `code/` directory and adds one optional
training-loss hook. The renderer remains unchanged: the default loss uses
`abs(gaussians.get_scaling + d_scaling)` and follows the canonical-only path when
Python covariance is enabled. Default weight zero preserves the T3 loss path;
all tuning disabled preserves the baseline path. New configs, validation logs
and results live under `experiments/isotropic_tuning` and
`runs/pidg_isotropic_tuning`. This is a Gaussian shape prior, not a material law.

## Shared time and kinematic consistency (separate fork)

`variants/kinematic_tuning/code` independently copies the frozen metric runtime.
`timeline.py` keeps the legacy `fid/train_count` physics path when disabled;
`shared_physics_time` switches the physics input to the same normalized `fid`
used by deformation/rendering. An additional `timeline=frame_index` option maps
the verified 200 numeric filenames jointly to `(frame_id-1)/199`, without editing
source JSON or changing the 150/50 split. Uniform capture cadence is an explicit
assumption; these retimed tests must be compared separately from historical
metadata-time metrics. Real elapsed seconds remain unknown.

`model.py` differentiates the full rendered center `R(q(X,t)) X+b(X,t)` using
second-order finite differences, including one-sided endpoint stencils.
`runtime.py` adds optional joint Huber consistency with `VelocityNet(X,t)`;
the canonical sample positions are detached, but deformation-grid, deformation
head and velocity-network parameter gradients remain enabled. This avoids
depending on unsupported coordinate second derivatives in the custom hash CUDA
encoder. The legacy canonical-coordinate momentum residual, convective
contraction and sampled weighting are preserved. `kinematic_weight=0` disables
the new loss, and all tuning flags off preserve the original training path.
Runs, complete resumable checkpoints, metrics, diagnostics and GT/render pairs
are isolated in `runs/pidg_kinematic_tuning` and `experiments/kinematic_tuning`.

## Time-disjoint validation / test retraining (separate fork)

`variants/kinematic_resplit/code` copies the frozen kinematic source. Its optional
`runtime.py::Tuning` retains the V3/V4 physics and uses only validation cameras
for checkpoint selection. `data.py` prepares 120 training / 30 validation / the
unchanged 50 official test acquisitions with a common frame-index clock. The
training loader's legacy test slot holds validation; final-test RGB is absent
from that loader's data directory. A separate view is used for final evaluation.
No old weights, released point cloud, flow or depth are reused. A new train-only
50k parent initializes V3/V4, and `run.py::branch` copies an identical complete
1000-step refinement state before activating V4 consistency. Selections are
locked before final test rendering. Historical use of the official test set
remains disclosed; the split does not retroactively remove tuning exposure.
New logs, checkpoint state, metrics and pairs live in
`runs/pidg_kinematic_resplit` and `experiments/kinematic_resplit`.

## V3 plus Gaussian isotropy (separate fork)

`variants/v3_isotropic/code` copies the completed resplit source. Its optional
`runtime.py::Tuning.regularization` reuses the tested normalized principal-axis
penalty from `variants/isotropic_tuning`, while inheriting the validation-only
evaluation and shared clock from `variants/kinematic_resplit/runtime.py`.
Two additional50k continuations start from the same completed V3 weights: iso
weight zero versus 0.001, with a byte-identical full-state branch at extra1000.
The only new trainer operation is the optional isotropic loss hook; kinematic
consistency stays off. Original resplit data and the frozen V3 parent stay
unchanged. New logs/results/charts live in `experiments/v3_isotropic`, and weights,
final renders and pixel-verified GT/render pairs in `runs/pidg_v3_isotropic`.
