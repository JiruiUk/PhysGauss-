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
