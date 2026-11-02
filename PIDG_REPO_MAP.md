# PIDG repository map — M0

## Visible image-fragment / fixed3D surface bank (2026-11-02)

`variants/surface_correspondence/model.py` adds disabled-by-default, non-neural
joint center/footprint/observed-motion support gates, three-observation fixed
canonical row selection, and conservative image-fragment qualification. It
does not edit PIDG paths or assign material/object labels. `output/surface_correspondence`
contains `common.py` for complete RGB color-adjoint contributions (beyond the
first20 diagnostics), `contracts.py` for real GPU/full-checkpoint integrity,
`audit.py` for five local references and all120 frozen-row forward evaluations,
`finalize.py` for per-point unknown retention/unique coverage/actual compositor
images/3D center paths, `bind_banks.py` for exact full-checkpoint/data/row SHA
binding, and `verify.py` for independent CPU flow and gate checks.
Protocols/results are in `experiments/surface_correspondence`; fixed banks and
NPZ trajectories/assignments in `output/surface_correspondence`; seed evidence
caches in `runs/pidg_surface_correspondence/cache`. Tests cover disabled behavior,
missing support, ambiguity and fixed-row selection. No new Gaussian training,
shape/contact/force/material objective or cross-occlusion physical-ID merge.

## Motion-history correspondence prototype (2026-11-02)

`variants/motion_correspondence/model.py` supplies an optional non-neural image
position/velocity tracker with irregular normalized time, covariance prediction,
one-to-one multiple hypotheses, merge rejection, expiry, persistent ambiguity
and JSON state restore. Flags off delegates exactly to the prior visible-only
`output.surface_support.core.propagate`. Candidate image seed identities are not
Gaussian/material labels. No PIDG optimizer or training/inference source changes.
`run.py` evaluates fixed synthetic hidden pass/bounce ambiguity, 36 artificial-gap
trials against frozen visible FB pseudo-references, and all120 train frames.
`output/motion_correspondence/rolling_test.py` adds local rolling visible tracklet
checks across the sequence; these do not provide global physical identities.
`render_support.py` checks complete C1 early-patch alpha compositing and centers;
`finalize.py` creates figures/browser; `verify_results.py` independently checks
inputs, metrics, masks, state replay and protected C1 hashes. Protocol, initial
failed uncertainty probe, corrected results and assessment are retained in
`experiments/motion_correspondence`; tests in `tests/test_motion_correspondence.py`.
This diagnostic does not resolve the rejected persistent-surface gate or resume P1.

## Full train-sequence surface-support audit (2026-11-02)

`output/surface_support/{core,audit,contracts,finalize,verify}.py` adds read-only
support diagnostics for historical C1 final50k and rejected P1 at16k. It checks
all120 train frames, full center trajectories, effective additive covariance,
full compositor group contributions, isolated/fixed-footprint/fixed-camera
counterfactuals and conservative raw bidirectional image tracking. The CUDA
RGB/alpha compositor uses all contributors but its diagnostic IDs/offsets store
only the first20 per pixel; metadata-normalized coverage is not full coverage.
No frozen renderer or trainer was changed. The early patch graph still fails
persistent visible support under complete compositing; unknown IDs stay unknown.
`experiments/surface_support/assessment.md` records limitations and required
identity/representation evidence before any new shape/contact training.

## C1 local shape-coherence candidate (2026-11-02)

`variants/shape_coherence/prepare.py` copies C1 and changes only the optional
Tuning import. `model.py` estimates local relative F with rank-gated FP64 fits,
separately switchable affine and broad stretch/collapse terms, and no material
law. `observations.py` lifts manually reviewed early train-image components to
partial reference patch labels using corrected alpha*T contributions. Unknown
points never connect, and these labels are not physical material IDs.
`runtime.py` builds a frozen graph only after original densification and binds
its state to complete checkpoints; disabled flags delegate to original C1.
`audit.py`, `verify_update.py`, and `tests/test_shape_coherence.py` check the
train-only evidence, intended gradients, actual optimizer update, affine/rigid
behavior, label isolation and checkpoint integrity. `run.py` pauses at16k for
manual graph review before enabling the objective. Historical C1 is reused.
`output/shape_coherence/{continue_run,review_graph,finalize,verify_results}.py`
provide launch receipts, a coverage gate, locked-selection comparison and media
verification. No collision force, constitutive law or learned segmentation is
introduced. Early points lose visible contribution later in C1, so this weak
partial prior must not be presented as complete persistent object tracking.

## C1 optical-flow-only ablation (2026-11-02)

User correction: reuse historical C1; do not retrain the original. The former
new O0 was safely stopped and excluded. `output/c1_flow_ablation/reuse_control.py`
trains only O1 in a separate coordinator queue; `compare_reuse.py` reuses the
historical locked best/final snapshots and existing PNGs without altering them.
`reuse_control_plan.json` supersedes the original two-new-arms protocol below.

`variants/c1_flow_ablation/prepare.py` copies the frozen C1_repaired trainer and
renderer, changing only the optional runtime import and additive observed-flow
hook. `runtime.py` inherits C1 image/alpha losses, frame-index clock, independent
velocity/stress and original Cauchy momentum unchanged. Its single treatment
flag `observed_flow_enabled` toggles corrected raw train-only geometric and
independent-velocity flow supervision; no M1/N1 structural or momentum changes
are inherited. `run.py` branches the identical full pre-flow3k checkpoint into
two newly trained50k arms. `tests/test_c1_flow_ablation.py`, CUDA smoke/recovery
and `verify_update.py` validate flag isolation, parent delegation, and actual
flow-only optimizer updates. `results.py` locks validation selection before test,
exports GT|Render, evaluates the frozen V2 validation observation cache, checks
camera schedules, full checkpoints and rendered artifacts. Reports reside in
`experiments/c1_flow_ablation`; historical C1 is a descriptive reference only.

## Coherent discrete trajectory momentum (2026-11-02)

`variants/coherent_momentum/operator.py` derives interval chord velocities,
acceleration and the physics rollout from the same complete center trajectory.
It retains stress_pred and the original deviatoric rate term, using one
normalized quadratic current-space fit for the stress Jacobian / velocity
Hessian instead of successive linear fits. `runtime.py` gates the new operator
behind `coherent_momentum`, preserving M1's gradient policy and exact new-flag-off
path. `tests/test_coherent_momentum.py` checks analytic fields, degeneracy,
gradients and parent parity. `prepare.py` creates an independent trainer copy;
`run.py`, `evaluate.py`, `export.py` control recovery, validation selection,
common-anchor dual-operator diagnostics and GT|Render. The frozen old PIDG,
L1 and M1 trees are unchanged. A shared trajectory was already present in M1;
the new objective improves discrete compatibility rather than claiming a new
independent-velocity unification or exact physical integration.

## Momentum-gradient repair follow-up (2026-11-02)

`variants/momentum_guard/runtime.py` optionally guards the inherited shared
momentum loss with same-batch RGB gradients. `balance.py` defines the explicit
first-order per-group policy, leaving private stress/rate-coefficient gradients
unchanged; `tests/test_momentum_guard.py` checks the policy and exact guard-off
parent path. `prepare.py` copies the frozen unified architecture and changes only
the trainer runtime import. `run.py` / `verify_update.py` provide one matched
full-state experiment, smoke/recovery/flags-off and real optimizer checks.
`experiments/momentum_repair/{diagnosis,plan}.json` record evidence and prospective
rules. This adds no constitutive or external-force module.

## Unified trajectory/stress follow-up (2026-11-02)

`variants/unified_flow/code` is an independent frozen copy of the flow-comparison
trainer. Its optional `DeformNetwork.shared_stress` is Linear(width,6), stored in
deformation snapshots/full state. `model.predict_stress` evaluates the same
deformation Grid4D and exact shared trunk latent as `gaussian_warp`; full centre
velocity remains in `variants/flow_comparison/temporal.py`. `runtime.Tuning`
selects shared versus original predicted stress in the same local momentum
operator and integrates corrected raw source→target-camera flow. New optimizer
group, checkpoint extension, input hashes, matched controls and validation-only
selection are in `checkpoint.py`, `prepare_data.py`, `run.py` and
`experiments/unified_flow/plan.json`. Flags off preserve the original paths.
Original PIDG/K1/K2/source locks and weights remain unchanged. The additional
head has no constitutive or external-force interpretation.

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

## Fresh V3 / V3+iso initialization (separate fork)

`variants/v3_iso_scratch/code` copies the optional-iso source and adds a flagged
step-zero capture hook before the training loop. `parent_run` is empty; new
Gaussian and neural parameters are initialized from random geometry and seed0.
The capture contains zero optimizer updates, verified by empty optimizer state
dictionaries. Both arms copy identical complete checkpoint bytes before step1;
only iso weight differs. They each train50k total, with original initial-training
warm-up, densification and learning rates, V3 image objectives/shared time, and
iso weight0 or0.001 from step1. Existing validation-only selection and final-test
locking are reused. The prior continuation is saved and stopped at the user's
request; no trained weights from it are used. New outputs are isolated in
`experiments/v3_iso_scratch` and `runs/pidg_v3_iso_scratch`.

## Ordered foreground coverage ablation (separate fork)

`variants/coverage_tuning/code` copies the frozen scratch source. Only optional
loss and post-optimizer density hooks are added to its trainer; an environment
flag enables diagnostic alpha PNG export in its renderer. The original RGB
compositor and all existing source trees remain unchanged. `data.py` audits and
reads original RGBA alpha from the120 training frames only. `model.py` defines
foreground alpha L1 and a weak local affine neighborhood fit allowing stretch
and shear. `density.py::CoverageAccumulator` collects unique-training-time
coverage evidence. `runtime.py::Tuning` integrates the losses, conservative late
point additions, and checksum-bound coverage checkpoint state. Render inspection
confirmed multiple ducks and obstacles; the earlier single-object assumption was
incorrect. Current real-scene affine neighborhoods have distance gating but no
object-ID filtering. Explicit known IDs are required for object-isolated contact
constraints. `run.py` orders mask, mask+affine,
and mask+affine+density50k scratch runs from the same random step-zero checkpoint,
reuses the completed V3 control, selects on validation only and locks selections
before final testing. Iso stays off. New diagnostics include foreground PSNR,
mask IoU and eroded-interior low-alpha rate. New outputs live exclusively in
`experiments/coverage_tuning` and `runs/pidg_coverage_tuning`.

## Standalone component effects (separate follow-up fork)

`variants/coverage_isolated/code` copies the frozen coverage trainer with only
the Tuning import changed. `runtime.py` retains the exact affine formula and
inherited density module, sets mask loss to zero, removes separate alpha targets
from the affine-only path, and replaces the disproved single-object declaration
with explicit acknowledgment of unsegmented neighbors. C4 enables affine only;
C5 enables density only and still uses training alpha to select additions.
`run.py` waits for the existing GPU queue, runs GPU preflight, then trains both
arms50k from the identical random zero state. Reports compare C4-C0 and C5-C0
alongside all existing cumulative arms. New files remain in
`experiments/coverage_isolated` and `runs/pidg_coverage_isolated`.

## Whole-scene foreground centroid (separate fork)

`variants/centroid_tuning/code` copies the frozen coverage hook fork and changes
only its Tuning import. `model.py` computes differentiable normalized 2D alpha
centroids; `runtime.py` applies an optional train-only centroid-distance loss.
The default original-PIDG recipe has no perceptual training loss, shared physics
clock, mask L1, affine prior, isotropic term, velocity consistency, or late
coverage additions. It uses the established clean120/30/50 data protocol and
its metadata rendering clock; original conventional densification remains.
This is not a physical3D center-of-mass estimator and has no real object-ID bank.

`run.py` captures a random full step0 state, trains a common3k prefix, then
branches identical tensors/optimizers/RNG/camera stacks into G0 and G1. Only
centroid_weight changes, with a2k ramp after3k and50k total steps per arm.
Validation-only selection is locked before final-test rendering. Completion
exports four image metrics, alpha/centroid diagnostics and pixel-verified GT
pairs. New outputs remain in `experiments/centroid_tuning` and
`runs/pidg_centroid_tuning`; historical controls are preserved.

## Coverage color repair (separate fork)

`variants/color_repair/prepare.py` copies the frozen coverage runtime, changes
the Tuning import, and inserts one opt-in renderer color branch. `color.py`
evaluates SH in deformed viewing directions and `model.py` maps to smooth
positive radiance. Flags-off retains the original CUDA SH path. `runtime.py`
adds delayed/ramped mask and train-foreground RGB losses, weaker spatial-affine
regularization and conservative training-color-aware coverage children;
`evidence.py` stores nonlearned color moments in checkpoint-bound sidecars.
No object-ID isolation, learned segmentation or physical mass claim is added.

`diagnose.py` only reads original final checkpoints and predetermined training
frames. `run.py` trains repaired C1/C2/C3 to50k from new random initialization,
with a shared3k prefix and exact shared15k C2/C3 pre-density state. Validation
selects snapshots before locked final testing. Original models, data split and
completed reports remain preserved. New outputs live in `experiments/color_repair`
and `runs/pidg_color_repair`.
# C1 shared effective-stiffness ablation

`variants/c1_material/prepare.py` copies frozen C1 repair code and changes only
the optional Tuning import. `model.py::SharedYoung` adds one bounded material-
shared E with fixed Poisson ratio. `runtime.py` reuses `pidg_ext` F estimation,
Neo-Hookean stress and robust consistency while retaining PIDG predicted stress.
Its canonical basis is detached; trajectory, stress and shared E receive gradients.
The new flag-off path returns the C1 runtime loss without new RNG/optimizer work.

All three runs branch the same complete pre-material C1 random-init3k state and
finish at total50k. Without known material IDs the scope is explicitly a global
effective prior, not real material identification. E is in normalized units, not
Pa. E/optimizer/canonical graph are checksum-bound to full checkpoints. Conditional
profiles cannot certify joint identifiability. New code/tests, reports and outputs
are isolated in `variants/c1_material`, `experiments/c1_material`, and
`runs/pidg_c1_material`. Old source, weights and test-selection files are preserved.

## Object-center candidate review (offline, no training integration)

`variants/object_centers/audit.py` reads frozen C1 repair final50000 and only
12 predetermined training frames. `grouping.py` compares fixed canonical,
trajectory/appearance and motion-dominant K-means candidates. IDs are fixed per
kernel; unknown kernels remain -1. `model.py` provides separately tested center,
zero-mean residual, piecewise-linear trajectory, topology inheritance and soft
overlap primitives. These are not inserted into C1 training or inference.

`experiments/object_centers/grouping_review.json` and the local gallery under
`runs/pidg_object_centers/grouping_review/c1_final50000` record the source hashes,
candidate IDs, GT/C1/group panels and measured center curves. Visual review
finds merged/split objects; clusters are not accepted as contact labels.
There is no new center-model long training or material segmentation.

`variants/object_centers/motion_grouping.py` additionally compares pure
displacement, displacement plus time-correct trajectory finite-difference
velocity, and motion-group refinement using persistent spatial connectivity.
`motion_review.py` reuses the original train-only evidence with SHA checks;
rendered C1 RGB and deformed poses are verified against that evidence before
exporting a separate motion-method gallery. Both baseK=5 and6 are inspected.
Spatial refinement can yield more components than baseK; these remain
unverified candidates, not accepted object/contact labels or new training.

`variants/object_centers/adaptive_grouping.py` removes the requested count
entirely: displacement HDBSCAN, displacement/velocity HDBSCAN, and a direct
motion/spatial graph with no base clustering. Minimum cluster sizes128/256,
min_samples and spatial/motion thresholds remain explicit parameters.
`adaptive_review.py` exports these no-K candidates to their own gallery using
the same train-only cache and immutable C1 snapshot. Shared export helpers now
accept an output directory and comparison field; previous default baseK paths
are retained. No center/contact loss is integrated into training.

`variants/object_centers/color_features.py` supplies an opt-in cached predicted
mean-RGB block for offline grouping. Density color_weight=0 and graph
color_radius=None retain the no-color algorithm paths. `color_review.py`
compares the three no-K methods with color enabled against their saved
motion-only candidates. Features use existing C1 predictions at training
views, not GT/test colors. A new gallery includes per-method before/after GT/C1
panels; C1 model weights, architecture and training losses remain unchanged.

## Trainable object identity / motion bottleneck (separate C1 fork)

`variants/object_id/prepare.py` copies frozen repaired C1 and inserts opt-in
deformation/state hooks. `model.py::ObjectMotion` maps canonical XYZ to soft
candidate membership and normalized time to group-shared SE3 center motion.
The original nonrigid trajectory is retained with a ramped25% grouped blend;
covariance rotation/scale, PIDG velocity/stress and C1 image/alpha losses remain.
The16-candidate ceiling is not a target object count; occupancy/pruning are
explicit diagnostics, with no GT instance or offline cluster label targets.

`runtime.py` adds separate optimization, RGB-gradient audits, auxiliary motion/
assignment losses and SHA-bound full-checkpoint sidecars. Selected deformation
snapshots contain matching object state. Disabled hooks return the original
warp and allocate no object optimizer. `run.py` compares C1, frozen identity
and learned identity from the identical retained full3k random-init checkpoint,
total50k per arm, validation-only selection and locked final testing. Reports
and new runs are isolated in `experiments/object_id` and `runs/pidg_object_id`.
This is a custom paper-inspired reconstruction ablation, not physical instance
ground truth, material segmentation or guaranteed nonpenetration.

## Frozen trajectory / velocity-only diagnostic (standalone, 2026-11-01)

`variants/frozen_velocity_probe/run.py::{export,train}` reads the H3 final50k
PLY/material/deformation snapshot, caches detached derivatives of the full
rendered center using `variants/kinematic_tuning/model.py`, and fits diagnostic
copies of the existing PIDG VelocityNet. No original training hook is changed.
Canonical coordinates and deformation/grid states stay frozen; the only fit
objective is velocity MSE, without momentum, force, image or constitutive loss.
`core.py` partitions cloned particle-ID families before selecting held-out
Gaussian rows, freezes the unused stress-head parameters, and reports amplitude,
direction and RMS alignment. Shared-feature updates can alter unused stress
outputs, so these velocity-only copies are not drop-in production stress models.
`report.py` and `collisions.py` inspect fitting/generalization and scalar-time
encoding aliases; the latter is a post-hoc frozen-teacher mechanism check.
`clock.py::apply_clock` defaults to a legacy no-op; its opt-in fixed monotonic
clock is tested in `time_control.py` with the same cached labels and fit budget.
Config/protocol/results live in `experiments/frozen_velocity_probe`; cached
targets and independent velocity checkpoints in `runs/pidg_frozen_velocity_probe`.
Disabled execution exits without model imports, checkpoint access or optimization.

## Full monotonic-clock / kinematic training (independent H3 fork)

`variants/monotonic_training/prepare.py` copies H3 and changes only the Tuning
import, the VelocityNet internal clock call, and the Gaussian constructor's
configuration hook. The renderer, deformation field, data loader and camera
sampler remain byte-identical to H3. `model.py::time_coordinate` preserves the
original Fourier/projected path by default; opt-in `velocity_clock_mode` selects
`tanh(2*t-1)`, retaining every original parameter and optimizer group while
freezing unused learned clock parameters. Both velocity and predicted stress
heads use the same existing features. Full resume retains the mode through the
checksum-bound run config; snapshots require that config when loading physics.
`runtime.py` reuses H3 and its existing joint full-center kinematic loss.
`run.py` performs GPU smoke/resume/flags-off checks, copies the identical full3k
state, trains to total50k, locks validation selection, then renders final test.
`export.py` checks pixel-identity GT|Render concatenations and independently
recomputes PNG PSNR before producing videos/gallery/archive. Experiments are in
`experiments/monotonic_training`; runs in `runs/pidg_monotonic_training`; user
deliverables in `output/monotonic_training`. No external force is introduced.
